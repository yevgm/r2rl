import sys
import os
import wandb
import numpy as np
sys.path.insert(0, os.path.abspath('./src'))
sys.path.insert(0, os.path.abspath('.'))
import gym
from tqdm import tqdm
from src.dqn.env_init import *
from src.dqn.evaluate import evaluate_policy, evaluate_policy_shock
from config import get_sweep_params


def evaluate_one_param_robustness(config_args, seed_vec, sweep_vector, sweep_str, model):
    EnvName = ['LocalCartPole-v1', 'LocalAcrobot-v0', 'LocalMountainCar-v0']

    config = copy.deepcopy(config_args)  # copy the 'Namespace' object - very important to copy!
    config_dict = vars(config)

    iter_reward_list = []
    reward_list = []
    std_error_list = []

    for param_value in tqdm(sweep_vector):
        config_dict[sweep_str] = param_value  # in-place change of given parameter (Hence the copy)

        for seed in seed_vec:
            config_dict['seed'] = int(seed)

            EnvIdex = config_dict['EnvIdex']
            if EnvIdex == 0:  # cartpole
                updater = UpdateCartpoleParams()
                env_kwargs = {
                    'use_stochastic_reward': config_args.use_stochastic_reward,
                    'reward_x_weight': config_args.reward_x_weight,
                    'reward_angle_weight': config_args.reward_angle_weight
                }
                env = gym.make('LocalCartPole-v1', **env_kwargs)
            elif EnvIdex == 1:  # acrobot
                updater = UpdateAcrobotParams()
                env = gym.make(EnvName[EnvIdex])
            elif EnvIdex == 2:  # mountain car
                updater = UpdateMountaincarParams()
                env = gym.make(EnvName[EnvIdex])
            else:
                raise NotImplementedError(
                    f'EnvIdex not implemented, expected 0 for cartpole or 1 for acrobot, got {EnvIdex}')

            env = updater.update_environment(env, config)

            total_reward = evaluate_policy(env=env, model=model, render=config.render, turns=20, env_idx=EnvIdex)
            iter_reward_list.append(total_reward)

        rewards_vec = np.array(iter_reward_list)
        reward_list.append(np.mean(rewards_vec))
        std_error_list.append(np.std(rewards_vec))

        iter_reward_list = []

    rew = np.array(reward_list)
    error = np.array(std_error_list)
    return rew, error


def evaluate_one_shock_params(config_args, seed_vec, model, param_name, param_values):
    EnvName = ['LocalCartPole-v1', 'LocalAcrobot-v0', 'LocalMountainCar-v0']

    config = copy.deepcopy(config_args)  # copy the 'Namespace' object - very important to copy!
    config_dict = vars(config)

    iter_reward_list = []
    for seed in seed_vec:
        config_dict['seed'] = int(seed)

        EnvIdex = config_dict['EnvIdex']
        if EnvIdex == 0:  # cartpole
            updater = UpdateCartpoleParams()
            env_kwargs = {
                'use_stochastic_reward': config_args.use_stochastic_reward,
                'reward_x_weight': config_args.reward_x_weight,
                'reward_angle_weight': config_args.reward_angle_weight
            }
            env = gym.make(EnvName[EnvIdex], **env_kwargs)
            shock_updater = UpdateCartpoleFixedParams(config, param_str_list=param_name, param_new_values=param_values)
        elif EnvIdex == 1:  # acrobot
            updater = UpdateAcrobotParams()
            env = gym.make(EnvName[EnvIdex])
            shock_updater = UpdateAcrobotFixedParams(config, param_str_list=param_name, param_new_values=param_values)
        elif EnvIdex == 2:  # mountain car
            updater = UpdateMountaincarParams()
            env = gym.make(EnvName[EnvIdex])
            shock_updater = UpdateMountaincarFixedParams(config, param_str_list=param_name, param_new_values=param_values)
        else:
            raise NotImplementedError(
                f'EnvIdex not implemented, expected 0 for cartpole or 1 for acrobot, got {EnvIdex}')

        env = updater.update_environment(env, config)
        total_reward = evaluate_policy_shock(env=env, model=model,
                                             env_updater=shock_updater, turns=20,
                                             shock_time=20)
        iter_reward_list.append(total_reward)

    rewards_vec = np.array(iter_reward_list)
    rew = np.mean(rewards_vec)
    error = np.std(rewards_vec)
    return rew, error


def train_and_evaluate(config, sweep_param, model_inst, n_iter=50):
    config.evaluate = True
    config.render = False
    config.Loadmodel = True

    # define an evaluation seed config_args, seed_vec, sweep_vector, sweep_str, model):
    seed_vec = np.arange(1, n_iter + 1)

    rews= []
    errors = []
    for model in tqdm(model_inst.get_model()):
        # run evaluation on one parameter parameters
        rew, error = evaluate_one_param_robustness(config_args=config,
                                                   seed_vec=seed_vec,
                                                   sweep_vector=sweep_param[0],
                                                   sweep_str=sweep_param[1],
                                                   model=model)

        rews.append(rew)
        errors.append(error)

    rew = np.array(rews)
    error = np.array(errors)

    return rew, error


def evaluate_shock_matrix(config, sweep_param_str, sweep_param_vecs, model_inst, n_iter=50):
    config.evaluate = True
    config.render = False
    config.Loadmodel = True

    # define an evaluation seed config_args, seed_vec, sweep_vector, sweep_str, model
    seed_vec = np.arange(1, n_iter + 1)

    score_max_shape = (len(sweep_param_vecs[0]), len(sweep_param_vecs[1]))
    rew_mat = np.zeros(score_max_shape + (len(model_inst), ))
    error_mat = np.zeros(score_max_shape + (len(model_inst), ))
    with tqdm(total=len(model_inst), file=sys.stdout) as pbar:
        for idx, model in enumerate(model_inst.get_model()):
            for i in range(score_max_shape[0]):
                for j in range(score_max_shape[1]):
                    rew, error = evaluate_one_shock_params(config_args=config, seed_vec=seed_vec,
                                                           model=model, param_name=sweep_param_str,
                                                           param_values=[sweep_param_vecs[0][i], sweep_param_vecs[1][j]])
                    rew_mat[i, j, idx] = rew
                    error_mat[i, j, idx] = error

            pbar.set_description('processed models: %d' % (1 + idx))
            pbar.update(1)

    return rew_mat, error_mat


def evaluate_robustness(opt, model, rw_list, err_list, total_steps, done=False):
    model_gen = TrainModel(opt, model)
    sweep_str = opt.evaluate_robustness_name
    sweep_vector = get_sweep_params(env=opt.EnvIdex, sweep_name=sweep_str)
    sweep_param = [sweep_vector, sweep_str]
    rew, error = train_and_evaluate(opt, sweep_param, model_gen)

    if ~done:
        label = "{:.3f}".format(total_steps)
    else:
        label = "End"

    rw_list[0].append(rew.tolist()[0])
    rw_list[1].append(label)
    err_list.append(error.tolist()[0])


def plot_robustness(opt, rw_list, err_list):
    if opt.use_wandb:
        sweep_str = opt.evaluate_robustness_name
        sweep_vector = get_sweep_params(env=opt.EnvIdex, sweep_name=sweep_str)

        xs = sweep_vector
        wandb.log({"Reward": wandb.plot.line_series(
            xs=xs,
            ys=rw_list[0],
            keys=rw_list[1],
            title="Reward")})
        wandb.log({"Error": wandb.plot.line_series(
            xs=xs,
            ys=err_list,
            keys=rw_list[1],
            title="Error")})



