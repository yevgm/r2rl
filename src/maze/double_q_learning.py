import sys
import os
import numpy as np
import math
import random
import argparse
import time
import pickle
import gym
import wandb
import inspect

from datetime import datetime


# Add script directory to sys.path.
def GetScriptDirectory():
    module_path = ""
    if hasattr(GetScriptDirectory, "dir"):
        return GetScriptDirectory.dir
    try:
        # The easy way. Just use __file__.
        # Unfortunately, __file__ is not available when cx_Freeze is used or in IDLE.
        module_path = __file__
    except NameError:
        if len(sys.argv) > 0 and len(sys.argv[0]) > 0 and os.path.isabs(sys.argv[0]):
            module_path = sys.argv[0]
        else:
            module_path = os.path.abspath(inspect.getfile(GetScriptDirectory))
            if not os.path.exists(module_path):
                # If cx_Freeze is used the value of the module_path variable at this point is in the following format.
                # {PathToExeFile}\{NameOfPythonSourceFile}.
                # This makes it necessary to strip off the file name to get the correct
                # path.
                module_path = os.path.dirname(module_path)
    GetScriptDirectory.dir = os.path.dirname(module_path)
    return GetScriptDirectory.dir


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
sys.path.append(os.path.join(GetScriptDirectory(), "lib"))

from config import parse_args_maze
from src import classic_control
from src.utils.optimization import *


class ReplayBuffer(object):
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))

        # Init buffer with uniform distribution
        # if not init_zeros:
        #     self.state = self.state + np.random.uniform(size=(max_size, state_dim))
        #     self.action = self.action + np.random.uniform(size=(max_size, 1))
        #     self.reward = self.reward + np.random.uniform(size=(max_size, 1))
        #     self.next_state = self.next_state + np.random.uniform(size=(max_size, state_dim))

    def add(self, state, action, reward, next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.max_size
        # self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(low=0, high=self.max_size, size=batch_size)
        return (self.state[ind], self.action[ind], self.reward[ind],
                self.next_state[ind])


class DoubleQLearning:

    def __init__(self, args):

        self.args = args
        self.output_dir = self.args.output_dir
        self.env = gym.make("marsrover-v0",
                            width=self.args.grid_size,
                            height=self.args.grid_size,
                            goal=self.args.goal,
                            risk_zone=self.args.risk_zone,
                            rwd_success=self.args.rwd_success,
                            rwd_step=self.args.rwd_step,
                            rwd_fail=self.args.rwd_fail,
                            random_start=self.args.random_start,
                            random_rate=self.args.random_rate)
        self.seed = self.set_seed(self.args.seed)

        '''
        Defining the environment related constants
        '''
        self.maze_size = (self.env.domain_width, self.env.domain_height)
        self.state_space_size = np.prod(self.maze_size)
        self.num_actions = self.env.action_space.n  # ["N", "S", "E", "W"]  # Number of discrete actions
        self.state_bounds = list(zip([0, 0], self.maze_size))  # Bounds for each discrete state

        '''
        Learning related constants
        '''
        self.decay_factor = np.prod(self.maze_size, dtype=float) / 10.0
        self.alpha_p = args.alpha_p
        self.alpha_r = args.alpha_r


        '''
        Defining the simulation related constants
        '''
        self.wandb_dict = {}
        self.num_episodes = args.num_episodes
        self.max_t = self.state_space_size * 100
        self.solved_t = self.state_space_size * args.solved_episode_length_factor
        self.fp = []  # trained model future location
        self.save_model = True if self.args.save_model else False

        self.use_wandb = args.use_wandb
        if self.use_wandb:
            self.wandb_run_name = args.wandb_run_name

        '''
        Creating a Q-Table for each state-action pair
        '''
        self.reward_vec = []
        self.q_table_a = np.zeros(self.maze_size + (self.env.action_space.n,), dtype=float)
        self.q_table_b = np.zeros(self.maze_size + (self.env.action_space.n,), dtype=float)

        '''
        Load R2 Q table if given and calculate best norm and policy
        '''
        if self.args.best_r2_q_name != '':
            model_path = os.path.join(self.args.output_dir, self.args.best_r2_q_name)
            with open(model_path, 'rb') as file:
                self.best_r2_q = pickle.load(file, encoding='latin1')
            # calculate norm
            v_t_true = np.amax(self.best_r2_q, axis=2).reshape(-1)
            self.optimal_r2_norm = np.linalg.norm(v_t_true, ord=self.args.r2_norm_p)
            self.optimal_r2_policy = np.argmax(self.best_r2_q, axis=2)
        else:
            self.best_r2_q = None
            self.optimal_r2_norm = None
            self.optimal_r2_policy = None

        '''Init Buffer'''
        self.buffer = ReplayBuffer(state_dim=len(self.maze_size),
                                   max_size=int(args.buffer_size))
        self.old_norm_estimate = 0
        # self.args.streak_to_end *= 10  # to be able to see decreasing norm before it cv

    def simulate(self, verbose=False):

        # Instantiating the maze related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        num_streaks = 0

        # Reset the environment
        obv = self.env.reset()
        # the initial state
        state_0 = tuple(obv)
        # Measure start time
        start_time = time.time()

        for _ in range(int(self.args.buffer_size)):
            action = self.select_action(state_0, 1)
            obv, reward, done, info = self.env.step_grid(action_command=action)
            state = tuple(obv)
            self.buffer.add(state=state_0, action=action, reward=reward, next_state=state)
            state_0 = state
            if done:
                # Reset the environment
                obv = self.env.reset()
                state_0 = tuple(obv)

        obv = self.env.reset()
        state_0 = tuple(obv)
        total_reward = 0

        for it in range(self.args.num_iterations):

            # Select an action
            action = self.select_action(state_0, explore_rate)
            # execute the action
            obv, reward, done, info = self.env.step_grid(action_command=action)
            # Observe the result
            state = tuple(obv)
            total_reward += reward
            # Add four tuple to buffer
            self.buffer.add(state=state_0, action=action, reward=reward, next_state=state)
            # Update Q tables
            self.update_step(state_0, state, action, reward, learning_rate, done)
            # Setting up for the next iteration
            state_0 = state
            # Print data
            self.print_episode_step(it,  action, state, reward, learning_rate,
                                    explore_rate, done, num_streaks)

            if done:
                self.wandb_dict['reward'] = total_reward
                self.wandb_dict['explore_rate'] = explore_rate
                self.wandb_dict['num_episodes'] = self.env.epoch
                self.wandb_dict['buffer_ptr'] = self.buffer.ptr

                if self.args.r2_norm_estimate:
                    self.calculate_r2_metrics()

                if self.args.use_wandb:
                    # wandb.log({"reward": total_reward, 'explore_rate': explore_rate, "num_episodes": self.env.epoch})
                    # if self.robust_q is not None:
                    #     l2_q_difference = self.compare_r2_q_table_to_robust_setting(self.q_table)
                    #     wandb.log({"l2_q_difference": l2_q_difference})
                    self.report_to_wandb()

                num_streaks = num_streaks + 1 if (not self.env.is_game_over and self.env.step_ep <= self.solved_t) \
                    else 0
                if it % 10 == 0:
                    print("Episode %d finished after %d time steps with total reward = %f (streak %d)."
                          % (self.env.epoch, self.env.step_ep, total_reward, num_streaks))

                # Reset the environment
                obv = self.env.reset()
                state_0 = tuple(obv)
                total_reward = 0

            # It's considered done when it's solved over streak_to_end times consecutively
            if num_streaks > self.args.streak_to_end:
                break

            # Update parameters
            explore_rate = self.get_explore_rate(it)
            learning_rate = self.get_learning_rate(it)
            # Learning time limit for robust setting
            if (time.time() - start_time) / 3600 > self.args.time_threshold:
                break

        if self.save_model:
            self.save_model_to_disk()

        return total_reward

    '''Choose a from s based on Q1 and Q2 (using epsilon greedy policy in Q1 + Q2)'''
    def select_action(self, state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            qa_value = self.q_table_a[state]
            qb_value = self.q_table_b[state]
            q_sum = qa_value + qb_value
            action = int(np.argmax(q_sum))

        return action

    '''Update rule for Q tables'''
    def update_step(self, state_0, state, action, reward, learning_rate, done):
        s, a, r, s_, lr = state_0, action, reward, state, learning_rate
        if np.random.random() < 0.5:
            # update q1
            return self.update_q_table(self.q_table_a, self.q_table_b, s, a, r, done, s_, lr)
        else:
            # update q2
            return self.update_q_table(self.q_table_b, self.q_table_a, s, a, r, done, s_, lr)

    '''Auxiliary function for update rule'''
    def update_q_table(self, q1_table, q2_table, state_0, action, reward, done,
                       state, learning_rate):
        maze_size = self.maze_size[0]
        # Take Q_target based on Q current argmax
        greedy_q1 = np.argmax(q1_table, axis=2)
        best_q = q2_table[state][greedy_q1[state]]

        q2_table_ = q2_table.reshape(maze_size ** 2, -1)
        q_argmax_ = greedy_q1.reshape(-1, 1)
        v_t = np.take_along_axis(q2_table_, q_argmax_, axis=1).reshape(-1)

        if self.args.algo_type == 'robust':
            p_support, _ = p_support_function(v=v_t, alpha=self.args.alpha_p, proxy_type=self.args.p_proxy_type) if self.args.alpha_p > 0 else 0,0
            r_support, _ = r_support_function(y=1, alpha=self.args.alpha_r) if self.args.alpha_r > 0 else 0, 0
            td_error = reward + (1 - done) * self.args.discount_factor * (best_q + p_support) + r_support - q1_table[state_0 + (action,)]
        elif self.args.algo_type == 'r2':
            if not self.args.r2_norm_estimate:
                if self.args.p_proxy_type == 'inner_product':
                    transition_norm = np.linalg.norm(v_t)  # l2 norm
                elif self.args.p_proxy_type == 'l1_norm':
                    transition_norm = np.max(np.abs(v_t))  # infinity norm
            else:
                four_tuple_batch = self.buffer.sample(batch_size=self.args.batch_size)
                transition_norm = self.estimate_norm_r2(four_tuple_batch, q1_table, q2_table)

            td_error = reward - self.args.alpha_r + (1 - done) * self.args.discount_factor * (best_q - self.args.alpha_p * transition_norm) \
                       - q1_table[state_0 + (action,)]
        else:
            td_error = reward + (1 - done) * self.args.discount_factor * best_q - q1_table[state_0 + (action,)]

        q1_table[state_0 + (action,)] += learning_rate * td_error


    '''Estimates the current Q table norm using a sampled batch of (s,a,r,s')'''
    def estimate_norm_r2(self, four_tuple_batch, q1_table, q2_table):
        state = four_tuple_batch[0]  # (batch, state)
        maze_dim = self.maze_size[0]

        # reshape to (maze_size ** 2, actions)
        q1_table_ = q1_table.reshape(maze_dim ** 2, -1)
        q2_table_ = q2_table.reshape(maze_dim ** 2, -1)
        # transform to (state_index)
        state_index = state[:, 0] * maze_dim + state[:, 1]
        uniques_states, counts_states = np.unique(state_index, return_counts=True)
        weights = dict(zip(uniques_states, counts_states / float(len(state_index))))

        w = np.array([weights[s] for s in state_index])

        state = state_index.reshape(-1, 1)
        actions_batch_q1 = np.take_along_axis(q1_table_, state.astype(int), 0)  # (batch, actions)
        actions_batch_q2 = np.take_along_axis(q2_table_, state.astype(int), 0)  # (batch, actions)

        q1_argmax = np.argmax(actions_batch_q1, axis=1).reshape(-1, 1)  # (batch)
        qmax = np.take_along_axis(actions_batch_q2, q1_argmax, axis=1)  # choose the argmax from the target Q

        # calculate norm p for qmax
        norm_estimate = (np.sum(np.multiply(qmax, w) ** self.args.r2_norm_p))**(1/float(self.args.r2_norm_p))
        new_norm_estimate = (1 - self.args.beta) * self.old_norm_estimate + self.args.beta * norm_estimate
        self.old_norm_estimate = new_norm_estimate
        return new_norm_estimate

    def print_episode_step(self, it, action, state, reward,
                           learning_rate, explore_rate, done, num_streaks):

        if self.args.debug_mode == 2:
            print("\nIteration = %d" % it)
            print("Action: %d" % action)
            print("State: %s" % str(state))
            print("Reward: %f" % reward)
            print("Explore rate: %f" % explore_rate)
            print("Learning rate: %f" % learning_rate)
            print("Streaks: %d" % num_streaks)
            print("")

        elif self.args.debug_mode == 1:
            if done or it >= self.max_t - 1:
                print("\nIteration = %d" % it)
                print("Steps in episode = %d" % self.env.step_ep)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("Total reward: %f" % reward)
                print("")

    def report_to_wandb(self):
        wandb.log(self.wandb_dict)

    def calculate_r2_metrics(self):
        four_tuple_batch = self.buffer.sample(batch_size=self.args.batch_size)
        norm_est = self.estimate_norm_r2(four_tuple_batch, self.q_table_a, self.q_table_a)
        r2_norm_distance = abs(norm_est - self.optimal_r2_norm)

        greedy_policy = np.argmax(self.q_table_a, axis=2)
        policy_distance = np.sum(greedy_policy != self.optimal_r2_policy) / float(self.maze_size[0] ** 2)

        self.wandb_dict['r2_norm_distance'] = r2_norm_distance
        self.wandb_dict['r2_policy_distance'] = policy_distance
        self.wandb_dict['norm_est'] = norm_est
        self.wandb_dict['true_norm'] = self.optimal_r2_norm

    def get_explore_rate(self, t):
        return max(self.args.min_explore_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(self.args.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def set_seed(self, seed):
        s = random.seed() if seed is None else seed
        self.seed = s
        self.env.seed(s)
        np.random.seed(s)
        random.seed(s)
        self.env.action_space.seed(s)
        return s

    def save_model_to_disk(self):
        out_dir = os.path.join(self.output_dir, 'maze')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        if self.args.use_wandb:
            self.fp = os.path.join(out_dir, self.args.algo_type + '_q_learning_seed_' + str(self.seed) + '_' + self.args.wandb_run_name + '.pkl')
        else:
            now = datetime.now()
            date_string = now.strftime("%b-%d-%Y_%H-%M-%S")
            self.fp = os.path.join(out_dir, self.args.algo_type + '_q_learning_seed_' + str(self.seed) + '_' + date_string + '.pkl')
        print(self.fp)
        with open(self.fp, 'wb') as file:
            pickle.dump(self.q_table_a, file)

    def load_model_from_disk(self, model_path):
        with open(model_path, 'rb') as file:
            self.q_table = pickle.load(file)

    def compare_r2_q_table_to_robust_setting(self, r2_q):
        q_norm_dist = np.linalg.norm(r2_q - self.robust_q)
        return q_norm_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_parser = parse_args_maze(parser)
    args, _ = model_parser.parse_known_args()
    # print("running with args: ", args)
    # args.algo_type = 'r2'
    # args.alpha_p = 1e-5
    # args.alpha_r = 1e-5
    # args.p_proxy_type = 'inner_product'


    if args.use_wandb:
        wandb.init(project="robust_rl", config=args, allow_val_change=True)
        # wandb.init(project="robustrl_sweeps", entity="robustrl", config=args)
        # x = vars(args)
        # args = argparse.Namespace(**wandb.config)
        # args = wandb.config
        args.wandb_run_name = wandb.run.name
        args.best_r2_q_name = "maze/r2_q_learning_seed_" + str(args.seed) + ".pkl"
        wandb.config.update(args, allow_val_change=True)

    maze_ins = DoubleQLearning(args)
    '''
    Begin simulation
    '''
    total_reward = maze_ins.simulate(verbose=True)




