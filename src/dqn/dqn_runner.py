import argparse
import os
import sys
import wandb
import gym
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import parse_args_ddqn, parse_global_args
from src.dqn.dqn_visualize import evaluate_robustness, plot_robustness
from src.utils.utils import ExpMovingAvg, set_seed
from src.utils.wandb import init_wandb, report_step_mean
from src.dqn.env_init import *
from src.dqn.evaluate import to_one_hot
from src.dqn.dqn import DQN_Agent, RobustDQNAgent, R2DQNAgent, RobustOptDQNAgent
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.evaluate import evaluate_policy


def main():
    # Set seed first
    parent = argparse.ArgumentParser()
    global_parser = parse_global_args(parent)
    seed = global_parser.parse_known_args()[0].seed
    set_seed(seed)

    parser = parse_args_ddqn(global_parser)
    EnvIdex = parser.parse_known_args()[0].EnvIdex

    EnvName = ['LocalCartPole-v1', 'LocalAcrobot-v0', 'LocalMountainCar-v0', 'maze-sample-stochastic-10x10-v0']
    BriefEnvName = ['CPV1', 'ACV0', 'MCV0', 'MAV0']

    if EnvIdex == 0:  # cartpole
        updater = UpdateCartpoleParams()
        opt = updater.update_parser(parser)[0]
        if opt.use_wandb:
            opt = init_wandb(parameters=opt)
        env_with_dw = True
        env_kwargs = {
            'use_stochastic_reward': opt.use_stochastic_reward,
            'reward_x_weight': opt.reward_x_weight,
            'reward_angle_weight': opt.reward_angle_weight
        }
        env = gym.make(EnvName[EnvIdex], **env_kwargs)
        env.seed(seed)
        env.action_space.seed(seed)
        eval_env = gym.make(EnvName[EnvIdex], **env_kwargs)
        eval_env.seed(seed)
        eval_env.action_space.seed(seed)

        env = updater.update_environment(env, opt)
        eval_env = updater.update_environment(eval_env, opt)
    elif EnvIdex == 1:  # acrobot
        updater = UpdateAcrobotParams()
        opt = updater.update_parser(parser)[0]
        if opt.use_wandb:
            opt = init_wandb(parameters=opt)
        env_with_dw = False
        env = gym.make(EnvName[EnvIdex])
        env.seed(seed)
        env.action_space.seed(seed)
        eval_env = gym.make(EnvName[EnvIdex])
        eval_env.seed(seed)
        eval_env.action_space.seed(seed)

        env = updater.update_environment(env, opt)
        eval_env = updater.update_environment(eval_env, opt)
    elif EnvIdex == 2:  # mountain car
        updater = UpdateMountaincarParams()
        opt = updater.update_parser(parser)[0]
        if opt.use_wandb:
            opt = init_wandb(parameters=opt)
        env_with_dw = True
        env = gym.make(EnvName[EnvIdex])
        env.seed(seed)
        env.action_space.seed(seed)
        eval_env = gym.make(EnvName[EnvIdex])
        eval_env.seed(seed)
        eval_env.action_space.seed(seed)

        env = updater.update_environment(env, opt)
        eval_env = updater.update_environment(eval_env, opt)
    elif EnvIdex == 3:  # maze (discrete)
        updater = UpdateMazeParams()
        opt = updater.update_parser(parser)[0]
        if opt.use_wandb:
            opt = init_wandb(parameters=opt)
        config_dict = {'transition_rate': opt.transition_rate,
                       'reward_rate_mean': opt.reward_rate_mean,
                       'reward_rate_std': opt.reward_rate_std,
                       'enable_render': opt.render_maze}

        env_with_dw = False
        env = gym.make(EnvName[EnvIdex], **config_dict)
        env.seed(seed)
        env.action_space.seed(seed)
        eval_env = gym.make(EnvName[EnvIdex], **config_dict)
        eval_env.seed(seed)
        eval_env.action_space.seed(seed)
        opt.use_stochastic_reward = False  # not used, just for compatibility
    else:
        raise NotImplementedError(f'EnvIdex not implemented, expected 0 for cartpole or 1 for acrobot, got {EnvIdex}')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_e_steps = env._max_episode_steps

    # Use DDQN or DQN
    algo_name = opt.algo_type
    ddqn = True if ((algo_name == 'DDQN') or (algo_name == 'R_DDQN') or (algo_name == 'R2_DDQN')) else False

    print('Algorithm:', algo_name, '  Env:', BriefEnvName[EnvIdex], '  state_dim:', state_dim, '  action_dim:',
          action_dim, '  Random Seed:', seed, '  max_e_steps:', max_e_steps)
    print('\n')

    # if it's maze, transform to one-hot representation
    if EnvIdex == 3:
        network_input_dim = (env.observation_space.high[0] + 1) ** 2
    else:
        network_input_dim = state_dim

    kwargs = {
        "env_with_dw": env_with_dw,
        "state_dim": network_input_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width, opt.net_width),
        "lr": opt.lr,
        "batch_size": opt.batch_size,
        "exp_noise": opt.exp_noise,
        "double_dqn": ddqn,
        'use_stochastic_reward': opt.use_stochastic_reward,
        'use_wandb': opt.use_wandb,
        'wandb_run_name': opt.wandb_run_name if opt.use_wandb else None,
        'env_idx': EnvIdex,
        'env': env,
        'scheduler_step': opt.scheduler_step
    }
    if not os.path.exists('model'): os.mkdir('model')
    if algo_name == 'R_DDQN':
        dic_cardinal_uncertainty, dic_nominal_uncertainty = create_robust_dicts(BriefEnvName[EnvIdex], opt)
        model = RobustDQNAgent(kwargs, env, dic_cardinal_uncertainty, dic_nominal_uncertainty, var=opt.var)

    elif algo_name == 'RO_DDQN':
        model = RobustOptDQNAgent(kwargs, alpha_p=opt.alpha_p, alpha_r=opt.alpha_r, p_proxy=opt.p_proxy, args=opt)

    elif (algo_name == 'R2_DDQN') or (algo_name == 'R2_DQN'):
        model = R2DQNAgent(kwargs, args=opt)
    else:
        model = DQN_Agent(**kwargs)

    if opt.Loadmodel: model.load(opt.model_path)

    buffer = ReplayBuffer(state_dim, max_size=int(1e6))

    print("running with the following args: ", opt)
    train_threshold = opt.stop_train_threshold
    rw_list = [[], []]
    err_list = []
    exp_avg = ExpMovingAvg()
    if opt.evaluate:
        score = evaluate_policy(eval_env, model, env_idx=EnvIdex, render=opt.render, turns=20)
        if opt.use_wandb:
            wandb.log({'score': score})
        print('EnvName:', BriefEnvName[EnvIdex], 'seed:', seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < int(opt.Max_train_steps):
            s, done, ep_r, steps = env.reset(), False, 0, 0
            while not done:
                steps += 1  # steps in current episode
                if buffer.size < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    # if it's maze, transform to one-hot representation
                    if EnvIdex == 3:
                        maze_size = env.observation_space.high[0] + 1
                        s_ = to_one_hot(s.copy(), maze_size)
                    else:
                        s_ = s

                    a = model.select_action(s_, deterministic=False)
                s_prime, r, done, info = env.step(a)

                '''Avoid impacts caused by reaching max episode steps'''
                if done and steps != max_e_steps:
                    dw = True  # dw: dead and win
                else:
                    dw = False

                buffer.add(s, a, r, s_prime, dw)
                s = s_prime.copy()
                ep_r += r

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        loss = model.train(buffer, total_steps)
                        exp_avg.update(loss)
                    if opt.use_wandb:
                        wandb.log({'loss': loss})

                # Evaluate robustness
                if (opt.evaluate_robustness_every != 0) and \
                    (total_steps > 0) and \
                    (total_steps % opt.evaluate_robustness_every == 0):
                    evaluate_robustness(opt, model, rw_list, err_list, total_steps)

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    model.exp_noise *= opt.noise_decay
                    score = evaluate_policy(eval_env, model, opt.render, EnvIdex)
                    if opt.use_wandb:
                        wandb.log({'score': score})
                    print('EnvName:', BriefEnvName[EnvIdex], 'seed:', seed,
                          'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)

                    if (total_steps != 0) and (np.abs(exp_avg.value()) < train_threshold):
                        # stop training
                        report_step_mean(opt, time_list=model.step_list)
                        model.save(algo_name, BriefEnvName[EnvIdex], total_steps, subdir=opt.model_subdir)
                        if opt.evaluate_robustness_every != 0:
                            evaluate_robustness(opt, model, rw_list, err_list, total_steps, done=True)
                            plot_robustness(opt, rw_list, err_list)
                        env.close()
                        return
                total_steps += 1

        if opt.evaluate_robustness_every != 0:
            evaluate_robustness(opt, model, rw_list, err_list, total_steps, done=True)
            plot_robustness(opt, rw_list, err_list)
        report_step_mean(opt, time_list=model.step_list)
        model.save(algo_name, BriefEnvName[EnvIdex], total_steps, subdir=opt.model_subdir)
    env.close()


if __name__ == '__main__':
    main()
