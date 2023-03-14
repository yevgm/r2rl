import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pickle
from src.utils.optimization import p_support_function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_net(layer_shape, activation, output_activation):
    """build net with for loop"""
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q


class DQN_Agent(object):
    def __init__(
            self,
            env_with_dw,
            state_dim,
            action_dim,
            use_stochastic_reward,
            use_wandb,
            gamma=0.99,
            hid_shape=(100, 100),
            lr=1e-3,
            batch_size=256,
            exp_noise=0.2,
            double_dqn=True,
            wandb_run_name=None,
            env_idx=0,
            env=None,
            scheduler_step=int(1.5e5),
    ):

        self.q_net = Q_Net(state_dim, action_dim, hid_shape).to(device)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.q_net_optimizer, step_size=int(scheduler_step))
        self.q_target = copy.deepcopy(self.q_net)
        # Freeze target networks with respect to optimizers
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.env_with_dw = env_with_dw
        self.gamma = gamma
        self.tau = 0.005  # used for exponential averaging in ddqn algorithm
        self.batch_size = batch_size
        self.exp_noise = exp_noise
        self.action_dim = action_dim
        self.double_dqn = double_dqn
        self.use_stochastic_reward = use_stochastic_reward
        self.env = env
        self.env_idx = env_idx
        self.use_wandb = use_wandb
        self.wandb_dict = {}
        if self.use_wandb:
            self.wandb_run_name = wandb_run_name

        self.step_list = []

    def report_to_wandb(self):
        if self.use_wandb:
            wandb.log(self.wandb_dict)

    def to_one_hot_tensor(self, s, size):
        '''
        In place chnage of s
        '''
        batch_size = s.size()[0]
        out_state = torch.zeros(batch_size, size ** 2)
        idx = s[:, 0] * size + s[:, 1]
        idx = idx.long()
        src = torch.ones(batch_size, 1)
        out_state.scatter_(index=idx.reshape(-1, 1), dim=1, src=src)

        return out_state

    def select_action(self, state, deterministic):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if deterministic:
                a = self.q_net(state).argmax().item()
            else:
                if np.random.rand() < self.exp_noise:
                    a = np.random.randint(0, self.action_dim)
                else:
                    a = self.q_net(state).argmax().item()
        return a

    def step_timer(self, start_time, end_time):
        if len(self.step_list) <= 1000:
            self.step_list.append(end_time - start_time)

    def train(self, replay_buffer, step=None):
        s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
        # if it's maze, transform to one-hot representation
        if self.env_idx == 3:
            maze_size = self.env.observation_space.high[0] + 1
            s_prime = self.to_one_hot_tensor(s_prime, maze_size)
            s = self.to_one_hot_tensor(s, maze_size)

        '''Compute the target Q value'''
        with torch.no_grad():
            # write starting step time
            start_step_time = time.time()

            if self.double_dqn:
                argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
                max_q_prime = self.q_target(s_prime).gather(1, argmax_a)
            else:
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

            '''Avoid impacts caused by reaching max episode steps'''
            if self.env_with_dw:
                target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime  # dw: die or win
            else:
                target_Q = r + self.gamma * max_q_prime

            # calculate average step time
            self.step_timer(start_step_time, end_time=time.time())

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        # if self.use_wandb: wandb.log({'loss': q_loss})

        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        self.scheduler.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item()

    def save(self, algo, EnvName, steps, subdir=''):
        use_stochastic_reward_spec = "" if not self.use_stochastic_reward else "_stochasticreward"
        wandb_run_name_spec = "" if not self.use_wandb else f"_{self.wandb_run_name}"
        if subdir is not None:
            dir = f"./model/{subdir}"
            if not os.path.exists(dir):
                os.makedirs(dir)

            torch.save(self.q_net.state_dict(), f"./model/{subdir}/{algo}_{EnvName}_{steps}"
                                                f"{use_stochastic_reward_spec}"
                                                f"{wandb_run_name_spec}"
                                                f".pth")
        else:
            torch.save(self.q_net.state_dict(), f"./model/{algo}_{EnvName}_{steps}"
                                                f"{use_stochastic_reward_spec}"
                                                f"{wandb_run_name_spec}"
                                                f".pth")

    def load(self, model_path):
        self.q_net.load_state_dict(torch.load(f"./model/{model_path}", map_location=device))
        self.q_target.load_state_dict(torch.load(f"./model/{model_path}", map_location=device))

    def load_in_notebook(self, model_path, sub_folder=''):
        file_dir = os.path.dirname(__file__)
        fp = os.path.abspath(os.path.join(file_dir, 'model', sub_folder, model_path))
        self.q_net.load_state_dict(torch.load(fp, map_location=device))
        self.q_target.load_state_dict(torch.load(fp, map_location=device))


class RobustDQNAgent(DQN_Agent):
    def __init__(self, kwargs, env, dic_cardinal_uncertainty, dic_nominal_uncertainty, var=1):
        super().__init__(**kwargs)
        self.env = env
        self.model_dicts = generate_us(dic_cardinal_uncertainty, dic_nominal_uncertainty, var)

    def batch_step_model(self, a, s, s_prime, dw_mask, model_dict):
        mod_next_state = torch.zeros(s.shape, dtype=s.dtype,
                                     device=s.device, requires_grad=s.requires_grad)
        a_ = a.squeeze().detach().cpu().numpy()
        dw_mask = dw_mask.detach().cpu().squeeze().numpy()
        for i in range(len(a_)):
            if not dw_mask[i]:
                mod_s_prime, reward, done, info = self.env.step_model(a_[i], s[i], model_dict)
                mod_next_state[i] = torch.tensor(mod_s_prime, dtype=s.dtype, device=s.device)
            else:
                mod_next_state[i] = s_prime[i]  # s_prime original

        return mod_next_state

    def train(self, replay_buffer, step=None):
        s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)

        '''Compute the target Q value'''
        with torch.no_grad():
            if self.double_dqn:
                    min_model_max_q_prime = None
                    for model_dict in self.model_dicts:
                        mod_next_state = self.batch_step_model(a, s, s_prime, dw_mask, model_dict)
                        mod_argmax_a = self.q_net(mod_next_state).argmax(dim=1).unsqueeze(-1)
                        mod_max_q_prime = self.q_target(mod_next_state).gather(1, mod_argmax_a)
                        # min on q_target for discrete models
                        if min_model_max_q_prime is None:
                            min_model_max_q_prime = mod_max_q_prime
                        else:
                            condition = (mod_max_q_prime < min_model_max_q_prime)
                            min_model_max_q_prime[condition] = mod_max_q_prime[condition]  # new minimum value
            else:
                raise NotImplemented

            '''Avoid impacts caused by reaching max episode steps'''
            if self.env_with_dw:
                target_Q = r + (1 - dw_mask) * self.gamma * min_model_max_q_prime  # dw: die or win
            else:
                target_Q = r + self.gamma * min_model_max_q_prime

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        # if self.use_wandb: wandb.log({'loss': q_loss})

        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        self.scheduler.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item()


class RobustOptDQNAgent(DQN_Agent):
    def __init__(self, kwargs, alpha_p, alpha_r, p_proxy, args):
        super().__init__(**kwargs)
        self.al_p = alpha_p
        self.al_r = alpha_r
        self.p_proxy = p_proxy
        # Only valid in case of maze - loading of previous r2 tables
        if self.env_idx == 3:
            self.maze_size = self.env.observation_space.high[0] + 1
            if args.best_r2_q_name_without_batch != '':
                model_path = os.path.join(args.output_dir, args.best_r2_q_name)
                with open(model_path, 'rb') as file:
                    self.best_r2_q_without_batch = pickle.load(file)

                self.estimate_r2_norm = True
            if args.best_r2_q_name_with_batch != '':
                model_path = os.path.join(args.output_dir, args.best_r2_q_name)
                with open(model_path, 'rb') as file:
                    self.best_r2_q_with_batch = pickle.load(file)

    def train(self, replay_buffer, step=None):
        s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)

        '''if it's maze, transform to one-hot representation'''
        if self.env_idx == 3:
            s_prime = self.to_one_hot_tensor(s_prime, self.maze_size)
            s = self.to_one_hot_tensor(s, self.maze_size)

        '''Compute the target Q value'''
        with torch.no_grad():
            # write starting step time
            start_step_time = time.time()

            if self.double_dqn:
                argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
                max_q_prime = self.q_target(s_prime).gather(1, argmax_a)
            else:
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

            '''Make robust optimization step'''
            v_t = self.q_net(s_prime).max(dim=1)[0]
            v_t = v_t.double().cpu().numpy()
            p_support, _ = p_support_function(v=v_t, alpha=self.al_p, proxy_type=self.p_proxy)
            '''Avoid impacts caused by reaching max episode steps'''
            if self.env_with_dw:
                target_Q = r + (1 - dw_mask) * self.gamma * (max_q_prime + p_support)  # dw: die or win
            else:
                target_Q = r + self.gamma * (max_q_prime + p_support)

            # calculate average step time
            self.step_timer(start_step_time, end_time=time.time())

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        # if self.use_wandb: wandb.log({'loss': q_loss})

        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        self.scheduler.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item()


class R2DQNAgent(DQN_Agent):
    def __init__(self, kwargs, args=None):
        super().__init__(**kwargs)
        self.al_p = args.alpha_p
        self.al_r = args.alpha_r
        self.p_proxy = args.p_proxy
        self.norm_estimate = 0
        self.beta = args.beta  # moving avg parameter

        self.best_r2_q_without_batch = None
        self.best_r2_q_with_batch = None
        self.estimate_r2_norm = False
        # Only valid in case of maze - loading of previous r2 tables
        if self.env_idx == 3:
            self.maze_size = self.env.observation_space.high[0] + 1
            if args.best_r2_q_name_without_batch != '':
                model_path = os.path.join(args.output_dir, args.best_r2_q_name)
                with open(model_path, 'rb') as file:
                    self.best_r2_q_without_batch = pickle.load(file)

                self.estimate_r2_norm = True
            if args.best_r2_q_name_with_batch != '':
                model_path = os.path.join(args.output_dir, args.best_r2_q_name)
                with open(model_path, 'rb') as file:
                    self.best_r2_q_with_batch = pickle.load(file)

    '''Estimates the current Q table norm using a sampled batch of (s,a,r,s')'''
    def estimate_norm_r2(self, four_tuple_batch, q_net, q_target):
        state = four_tuple_batch[0]  # (batch, state)
        q_net_val = q_net(state)  # (batch, actions)
        q_net_argmax = q_net_val.argmax(dim=1)  # (batch, )
        q_target_val = q_target(state)  # (batch, actions)
        qmax = q_target_val.gather(dim=1, index=q_net_argmax.unsqueeze(-1))  # (batch, )

        # calculate norm p for qmax
        if self.p_proxy == 'inner_product':
            norm_estimate = (sum(qmax ** 2)) ** (1/2)  # dual norm (itself)
        elif self.p_proxy == 'l1-norm':
            norm_estimate = max(abs(qmax))  # dual norm (l_infinity)

        return norm_estimate, q_net_argmax

    def train(self, replay_buffer, step=None):
        s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
        '''maze r2 comparison'''
        self.calculate_r2_metrics((s, a, r, s_prime))
        '''if it's maze, transform to one-hot representation'''
        if self.env_idx == 3:
            s_prime = self.to_one_hot_tensor(s_prime, self.maze_size)
            s = self.to_one_hot_tensor(s, self.maze_size)

        '''Compute the target Q value'''
        with torch.no_grad():
            # write starting step time
            start_step_time = time.time()

            if self.double_dqn:
                argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
                max_q_prime = self.q_target(s_prime).gather(1, argmax_a)
            else:
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

            current_norm_estimate, _ = self.estimate_norm_r2((s, a, r, s_prime), self.q_net, self.q_target)
            norm_estimate = self.beta * self.norm_estimate + (1 - self.beta) * current_norm_estimate  # moving avg
            self.norm_estimate = norm_estimate  # update last norm

            '''Avoid impacts caused by reaching max episode steps'''
            if self.env_with_dw:
                target_Q = r - self.al_r + (1 - dw_mask) * self.gamma * (max_q_prime - self.al_p * norm_estimate)  # dw: die or win
            else:
                target_Q = r - self.al_r + self.gamma * (max_q_prime - self.al_p * norm_estimate)

            # calculate average step time
            self.step_timer(start_step_time, end_time=time.time())

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)

        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        self.scheduler.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item()

    def calculate_r2_metrics(self, four_tuple_batch):
        self.calculate_r2_metrics_(four_tuple_batch, self.best_r2_q_without_batch, 'r2_no_batch')
        self.calculate_r2_metrics_(four_tuple_batch, self.best_r2_q_with_batch, 'r2_batch')

    def calculate_r2_metrics_(self, four_tuple_batch, q_table, q_table_name):
        if self.env_idx == 3:
            if self.estimate_r2_norm:
                (s_, a, r, s_prime_) = four_tuple_batch
                '''transform to one-hot representation'''
                s_prime = self.to_one_hot_tensor(s_prime_.clone(), self.maze_size)
                s = self.to_one_hot_tensor(s_.clone(), self.maze_size)

                norm_est, policy_est = self.estimate_norm_r2((s, a, r, s_prime), self.q_net, self.q_target)
                v_t = np.amax(q_table, axis=2).reshape(-1)
                norm_best = np.linalg.norm(v_t)
                r2_norm_distance = np.linalg.norm(norm_est - norm_best)

                # batch-wise best policy
                greedy_policy = policy_est.cpu().numpy()
                best_policy_full = np.argmax(q_table, axis=2).flatten()
                idx = s_[:, 0] * self.maze_size + s_[:, 1]
                idx = idx.long().cpu().numpy()
                best_policy = best_policy_full.take(indices=idx)
                policy_distance = np.sum(greedy_policy != best_policy) / (self.maze_size ** 2)

                self.wandb_dict[q_table_name + '_norm_distance'] = r2_norm_distance
                self.wandb_dict[q_table_name + '_policy_distance'] = policy_distance


def generate_us(dic_cardinal_uncertainty, dic_nominal_uncertainty, var):
    '''
    Robust dqn general method for generating synthetic uncertainty sets
    :param dic_cardinal_uncertainty: dict with the structure - key: permutations per key
    :param dic_nominal_uncertainty: nominal value dictionary
    :param var: gaussian variance for sampling near nominal
    :return: list of dicts (permutations of nominal)
    '''

    # create uncertainty values
    dic_uncertainty_values = dic_cardinal_uncertainty.copy()

    for key in dic_cardinal_uncertainty:
        values = []
        for v in range(dic_cardinal_uncertainty[key]):
            values.append(np.abs(np.random.normal(loc=dic_nominal_uncertainty[key], scale=var))) # no negative values allowed

        dic_uncertainty_values[key] = [dic_cardinal_uncertainty[key], values]

    dict_list = generate_us_inner(dic_uncertainty_values, dic_nominal_uncertainty)

    return dict_list


def generate_us_inner(dic_uncertainty_values, dic_nominal_uncertainty):
    '''
    Inner recursive function of generate_us.
    Recursive due to the dynamic property of parameters to loop on.

    :param dic_uncertainty_values: dict with the structure
        key: list(#permutations per key, list(samples of values for this key) )
    :param dic_nominal_uncertainty: nominal dictionary which will be modified every permutation
    :return: list of dicts (permutations of nominal)
    '''

    if any(dic_uncertainty_values) == False:
        return [dic_nominal_uncertainty]

    dict_list = []

    for key in dic_uncertainty_values:
        for i in range(dic_uncertainty_values[key][0]):
            dic_nominal_uncertainty_copy = dic_nominal_uncertainty.copy()
            dic_nominal_uncertainty_copy[key] = dic_uncertainty_values[key][1][i]  # new nominal
            dic_uncertainty_values_copy = dic_uncertainty_values.copy()
            dic_uncertainty_values_copy.pop(key, None)   # remove this loop

            list_of_dicts = generate_us_inner(dic_uncertainty_values_copy, dic_nominal_uncertainty_copy)
            dict_list = dict_list + list_of_dicts

        break  # choose only one key

    return dict_list


if __name__ == '__main__':
    cardinal = {'test1': 2,
                'test2': 3,
                'test3': 2}

    nominal = {'abc': 22,
                'test1': 41,
                'test2': 0.123,
                'test3': 152,
               'a': 235}

    dict_list = generate_us(cardinal, nominal, var=1)
    a=1