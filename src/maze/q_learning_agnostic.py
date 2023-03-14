import sys
import os
import math
import random
import argparse
import time
import pickle
import gym
import wandb
import copy
import inspect
import pandas as pd
import csv

from tqdm import tqdm
from datetime import datetime


# Add script directory to sys.path.
def GetScriptDirectory():
    if hasattr(GetScriptDirectory, "dir"):
        return GetScriptDirectory.dir
    module_path = ""
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


class QLearning:

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

        # self.seed = None
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
        self.decay_factor = self.state_space_size / 10.0
        if self.args.alpha != -1:
            self.alpha_p = self.alpha_r = self.args.alpha
        else:
            self.alpha_p = self.args.alpha_p
            self.alpha_r = self.args.alpha_r
        '''
        Defining the simulation related constants
        '''
        self.max_t = self.state_space_size * 100
        self.solved_t = self.state_space_size * self.args.solved_episode_length_factor
        self.fp = []  # trained model future location
        self.save_model = True if self.args.save_model else False

        '''
        Creating a Q-Table for each state-action pair
        '''
        # self.reward_vec = []
        self.q_table = np.zeros(self.maze_size + (self.env.action_space.n,), dtype=float)

        '''
        Load Robust Q table if given
        '''
        if self.args.robust_q_name != '':
            model_path = os.path.join(self.args.output_dir, self.args.robust_q_name)
            with open(model_path, 'rb') as f:
                self.robust_q = pickle.load(f)
        else:
            self.robust_q = None

    def simulate(self, verbose=False):

        # Instantiating the maze related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        num_streaks = 0

        list_it_time = []

        # Render the maze
        # TODO: debug rendering
        if self.args.render_maze:
            self.env.render()

        # Reset the environment
        obv = self.env.reset()
        # the initial state
        state_0 = self.state_to_bucket(obv)
        total_reward = 0

        # Measure start time
        start_time = time.time()
        for it in range(self.args.num_iterations):

            # Select an action
            action = self.select_action(state_0, explore_rate)
            obv_0 = obv

            # execute the action
            obv, reward, done, info = self.env.step_grid(action_command=action)

            # Observe the result
            state = self.state_to_bucket(obv)
            total_reward += reward
            # print(obv_0, state_0, action, reward, obv, state)

            # Update the Q based on the result
            best_q = np.amax(self.q_table[state])
            v_t = np.amax(self.q_table, axis=2).reshape(-1)
            # Robust setting for reward uncertainty and transition uncertainty
            # assign self.alpha_p, self.alpha_r = 0, 0 : for vanilla Q learning

            start_it_time = time.time()
            if self.args.algo_type == 'robust':
                if self.alpha_p > 0:
                    p_support, _ = p_support_function(v=v_t, alpha=self.alpha_p, proxy_type=self.args.p_proxy_type)
                else:
                    p_support = 0
                if self.alpha_r > 0:
                    r_support, _ = r_support_function(y=1, alpha=self.alpha_r)
                else:
                    r_support = 0

                td_error = reward + (1-done) * self.args.discount_factor * (best_q + p_support) + r_support\
                           - self.q_table[state_0 + (action,)]
            elif self.args.algo_type == 'r2':
                # l2 norm or infinity norm
                transition_norm = np.linalg.norm(v_t) if self.args.p_proxy_type == 'inner_product' else np.max(np.abs(v_t))

                td_error = reward - self.alpha_r + (1-done) * self.args.discount_factor * (best_q - self.alpha_p * transition_norm)\
                           - self.q_table[state_0 + (action,)]
            else:
                td_error = reward + (1 - done) * self.args.discount_factor * best_q - self.q_table[state_0 + (action,)]

            it_time = time.time() - start_it_time
            if it < 1000:
                list_it_time.append(it_time)
            # else:
                # print(np.mean(list_it_time), np.std(list_it_time))

            self.q_table[state_0 + (action,)] += learning_rate * td_error

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if self.args.debug_mode == 2:
                print("\nEpisode = %d" % it)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif self.args.debug_mode == 1:
                # if done:
                print("\nIteration = %d" % it)
                print("Steps in episode = %d" % self.env.step_ep)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("Total reward: %f" % total_reward)
                print("")

            # Render the maze
            if self.args.render_maze:
                self.env.render()

            if done:
                if self.args.use_wandb:
                    wandb.log({"reward": total_reward, 'explore_rate': explore_rate, "num_episodes": self.env.epoch})
                    if self.robust_q is not None:
                        l2_q_difference = self.compare_r2_q_table_to_robust_setting(self.q_table)
                        wandb.log({"l2_q_difference": l2_q_difference})

                num_streaks = num_streaks + 1 if (not self.env.is_game_over and self.env.step_ep <= self.solved_t) \
                        else 0
                if it % 10 == 0:
                    print("Episode %d finished after %d time steps with total reward = %f (streak %d)."
                      % (self.env.epoch, self.env.step_ep, total_reward, num_streaks))

                # Reset the environment
                obv = self.env.reset()
                state_0 = self.state_to_bucket(obv)
                total_reward = 0

            # It's considered done when it's solved over 120 times consecutively
            if num_streaks > self.args.streak_to_end:
                print("Streaks reached threshold. Task solved!")
                break

            # Update parameters
            explore_rate = self.get_explore_rate(it)
            learning_rate = self.get_learning_rate(it)

            if (time.time() - start_time) / 3600 > self.args.time_threshold:
                print("too long training time. Stopping")
                break

        if self.save_model:
            self.save_model_to_disk()

        return total_reward

    def evaluate(self, model_path):

        if model_path is not None:
            self.load_model_from_disk(model_path)

        # Reset the environment
        obv = self.env.reset()
        # print("Env parameters for evaluation: ", self.env.rwd_fail, self.env.rwd_success, self.env.rwd_step, self.env.random_rate)

        state_0 = self.state_to_bucket(obv)
        total_reward = 0
        reward_vec = []

        for t in range(self.args.num_it_eval):

            action = int(np.argmax(self.q_table[state_0]))
            obv, reward, done, _ = self.env.step_grid(action_command=action)
            state = self.state_to_bucket(obv)
            total_reward += reward
            state_0 = state

            if done:
                # print("Finished after %f time steps with total reward = %f." % (t, total_reward))
                reward_vec.append(total_reward)
                obv = self.env.reset()
                state_0 = self.state_to_bucket(obv)
                total_reward = 0

        return np.mean(reward_vec), np.std(reward_vec)

    def select_action(self, state, explore_rate):
        action = self.env.action_space.sample() if random.random() < explore_rate else int(np.argmax(self.q_table[state]))
        return action

    def get_explore_rate(self, t):
        return max(self.args.min_explore_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(self.args.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.maze_size[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.maze_size[i] - 1) * self.state_bounds[i][0] / bound_width
                scaling = (self.maze_size[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def set_seed(self, seed):
        s = random.seed() if seed is None else seed
        self.seed = s
        self.env.seed(s)
        np.random.seed(s)
        random.seed(s)
        self.env.action_space.seed(s)
        return s

    def save_model_to_disk(self):

        out_dir = os.path.join(self.output_dir, 'maze/evaluations')
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
        with open(self.fp, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_model_from_disk(self, model_path):

        with open(model_path, 'rb') as file:
            self.q_table = pickle.load(file)

    def compare_r2_q_table_to_robust_setting(self, r2_q):
        q_norm_dist = np.linalg.norm(r2_q - self.robust_q)
        return q_norm_dist


def train_maze(config, verbose=False):
    maze_ins = QLearning(config)
    # print(maze_ins.q_table)
    total_reward = maze_ins.simulate(verbose)
    print(maze_ins.fp)
    if not config.save_model:
        print('Model is not saved')

    return total_reward, maze_ins.fp


def evaluate_robustness(config_args, sweep_vector, sweep_str, model_name):
    # config = copy.deepcopy(config_args)  # copy the 'Namespace' object - very important to copy!
    config = config_args
    config_dict = vars(config)
    model_path = os.path.join(config.output_dir, model_name)

    avg_rwd = []
    std_rwd = []

    for param_value in tqdm(sweep_vector):
        config_dict[sweep_str] = param_value  # in-place change of given parameter (Hence the copy)
        print('New param', config.random_rate)
        maze_ins = QLearning(config)
        avg, std = maze_ins.evaluate(model_path=model_path)
        avg_rwd.append(avg)
        std_rwd.append(std)

    return avg_rwd, std_rwd, sweep_vector, sweep_str


def train_and_evaluate(config, sweep_params, model_name=None):
    # cond = (model_name is None and seed is not None) or (model_name is not None and seed is None)
    # assert cond, 'only one of model_name or seed must be passed'

    # define an evaluation seed
    # seed_vec = np.arange(1, n_iter + 1)

    # train model
    if model_name is None:
        model_fps = []
        # for s in config.seeds:
        #     config.seed = s
        assert config.save_model, 'save_model must be True to evaluate model'
        # config.save_model = True
        out, model_fp = train_maze(config=config, verbose=False)
        model_fps.append(model_fp)
    else:
        model_fps = ['maze/evaluations' + x for x in model_name]

    rews = {}
    errs = {}
    for fp in model_fps:
        # update saved model location
        # wandb.config["key"] = value
        # config.model_name = wandb.config.update({'model_name': os.path.join(*fp.split('/')[-2:])})

        print(fp)
        print(os.path.join(*fp.split('/')[-2:]))
        model_name_evaluate = os.path.join(*fp.split('/')[-2:]) if not config.use_wandb else fp

        for key in sweep_params.keys():
            rew = evaluate_robustness(config_args=config, sweep_vector=sweep_params[key], sweep_str=key, model_name=model_name_evaluate)
            rews[key] = rew
            # errs[key] = err

            if config.use_wandb:
                data = [[x, y] for (x, y) in zip(sweep_params[key], rew)]
                table = wandb.Table(data=data, columns=["env param " + key, "reward"])
                wandb.log({"Robustness w.r.t. " + key: wandb.plot.line(table, "env param " + key, "reward",
                                                                title="Testing robustness")})

    return rews


def evaluate_policy(config_args, num_seeds, param_str, param_val):
    config = copy.deepcopy(config_args)  # copy the 'Namespace' object - very important to copy!
    config_dict = vars(config)
    model_path = os.path.join(config.output_dir, config.model_name)

    iter_reward_list = []
    config_dict[param_str] = param_val  # in-place change of given parameter (Hence the copy)

    for ins in range(num_seeds):
        # config_dict['seed'] = config_args.seed
        maze_ins = QLearning(config)
        total_reward = maze_ins.evaluate(model_path=model_path)
        iter_reward_list.append(total_reward)

    return np.mean(iter_reward_list), np.std(iter_reward_list)


def train_and_evaluate_policy(config, num_seeds, param_str, param_val, model_name=None):
    # cond = (model_name is None and seed is not None) or (model_name is not None and seed is None)
    # assert cond, 'only one of model_name or seed must be passed'

    # define an evaluation seed
    # seed_vec = np.arange(1, n_iter + 1)

    # train model
    if model_name is None:
        model_fps = []
        for s in range(num_seeds):
            config.seed = random.seed()
            config.save_model = True
            out, model_fp = train_maze(config=config, verbose=False)
            model_fps.append(model_fp)
    else:
        model_fps = ['maze/evaluations' + x for x in model_name]

    rews = []
    errs = []
    for fp in model_fps:
        # update saved model location
        config.model_name = os.path.join(*fp.split('/')[-2:])
        rew, err = evaluate_policy(config_args=config, num_seeds=num_seeds, param_str=param_str, param_val=param_val)
        rews.append(rew)
        errs.append(err)

    return np.mean(rews), np.mean(errs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    model_parser = parse_args_maze(parser)
    args, _ = model_parser.parse_known_args()
    if args.use_wandb:
        wandb.init(project="robust_rl", config=args, allow_val_change=True)
        # args = wandb.config
        # args = argparse.Namespace(**wandb.config)
        args.wandb_run_name = wandb.run.name
        wandb.config.update(args, allow_val_change=True)

    train_maze(config=args, verbose=False)

    # # # Parameters for evaluating robustness after training
    # evaluate_params = dict({'random_rate': np.arange(0., 1, .025)})
    #     # 'rwd_success': np.arange(1., 0., -.1),
    #     #                   'rwd_step': np.arange(-0.01, -0.1, -0.01),
    #     #                   'rwd_fail': np.arange(0.5, -0.5, -0.05),
    #
    # for algo in ['vanilla', 'r2','robust']:
    #     args.algo_type = algo
    #     print('Algo type: ', algo)
    #     df = train_and_evaluate_csv(config=args, sweep_params=evaluate_params, seed_list=range(10))
    #
