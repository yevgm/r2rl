import argparse
import os
import numpy as np


def str2bool(v):
    """transfer str to bool for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_global_args(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    repo_root = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(repo_root, 'output')
    parser.add_argument('--repo-root', default=repo_root, type=str,
                        help='repository main dir')
    parser.add_argument('--output-dir', default=output_folder, type=str,
                        help='output dir')

    parser.add_argument('--save-model', default=True, type=str2bool,
                        help='save model to output folder or not')
    parser.add_argument('--use-wandb', default=False, type=str2bool,
                        help='whether to use weights and biases or not')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for everything')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='set gpu number')

    return parser


def parse_args_maze(parent):
    """
    Parse commandline arguments.
    """
    parser = parse_global_args(parent)

    # mars rover env specific args
    parser.add_argument('--grid_size', default=10,
                        type=int, help='size of the grid (square)')
    parser.add_argument('--goal', default=(2, 9),
                        type=tuple, help='goal_state')
    parser.add_argument('--risk_zone', default=[(4,5), (5,4), (2,5), (3,4), (4,3),(4,8), (2,7), (2,3),(5, 7), (3,7)],
                        type=list, help='risky zone that ends episode -- must be list of tuples')
    parser.add_argument('--rwd_success', default=1,
                        type=float, help='rwd success (goal), rwd step (no goal no risk zone), rwd fail (risk zone)')
    parser.add_argument('--rwd_step', default=-0.01,
                        type=float, help='rwd success (goal), rwd step (no goal no risk zone), rwd fail (risk zone)')
    parser.add_argument('--rwd_fail', default=-0.5,
                        type=float, help='rwd success (goal), rwd step (no goal no risk zone), rwd fail (risk zone)')
    parser.add_argument('--random_rate', default=0.,
                        type=float, help='probability of taking random action instead of prescripted one')
    parser.add_argument('--random_start', default=False,
                        type=bool, help='whether to start from random state in the grid')
    # parser.add_argument('--sweep_params', default={}, help='parameters to test robustness on - after training')
    parser.add_argument('--min_explore_rate', default=0.01,
                        type=float, help='minimal epsilon for exploration')
    parser.add_argument('--min_learning_rate', default=0.2,
                        type=float, help='minimal learning rate')
    parser.add_argument('--alpha', default=-1, type=float,
                        help='alpha rate for both transition and reward uncertainty. Default (-1) uses separate values')
    parser.add_argument('--alpha_p', default=0., type=float,
                        help='alpha rate for transition uncertainty in the robust MDP Q learning setting')
    parser.add_argument('--alpha_r', default=0., type=float,
                        help='alpha rate for reward uncertainty in the robust MDP Q learning setting')
    parser.add_argument('--p-proxy-type', default='inner_product', type=str,
                        help='proxy type for transition uncertainty in the robust MDP Q learning setting')
    parser.add_argument('--discount-factor', default=0.99, type=float,
                        help='bellman discount factor')
    parser.add_argument('--algo-type', default='vanilla', type=str,
                        help='choose between q-learning algorithm type -- vanilla, robust, r2')
    parser.add_argument('--best-r2-q-name', default='', type=str,
                        help='given r2 q table') # maze/r2_q_learning_seed_4_Dec-19-2022_12-04-06.pkl # q_learning_seed_42_swift-brook-310.pkl "maze/q_learning_seed_42_Mar-25-2022_14-39-45.pkl"
    parser.add_argument('--robust_q_name', default='', type=str,
                        help='best robust q table')

    parser.add_argument('--num_episodes', default=500, type=int,
                        help='maximal number of episodes')
    parser.add_argument('--num_it_eval', default=int(1e3), type=int,
                        help='number of evaluation steps')
    parser.add_argument('--num_iterations', default=1000000, type=int,
                        help='maximal number of episodes')
    parser.add_argument('--solved-episode-length-factor', default=1, type=float,
                        help='multiplier of the maximal steps in an episode to count for a streak')
    parser.add_argument('--time-threshold', default=4, type=int,
                        help='maximal time in hours to run')
    parser.add_argument('--streak_to_end', default=100, type=int,
                        help='how many goal reaches to end training')
    parser.add_argument('--debug-mode', default=0, type=int,
                        help='debug mode')
    parser.add_argument('--render-maze', default=False, type=bool,
                        help='render maze')
    parser.add_argument('--enable-recording', default=False, type=bool,
                        help='enable recording')

    # double q learning
    parser.add_argument('--buffer-size', default=1e5, type=int,
                        help='buffer size of four tuples')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size in the replay buffer')
    parser.add_argument('--r2-norm-estimate', default=False, type=bool,
                        help='r2 algorithm - to estimate the norm or use the true norm')
    parser.add_argument('--r2-norm-p', default=2, type=int,
                        help='norm type to estimate (p=2 is euclidean) for r2 algorithm')
    parser.add_argument('--beta', default=1, type=float,
                        help='moving average - mixture parameter (for double q learning only)')

    # only dqn - this is the comparison to previous versions of r2
    parser.add_argument('--best_r2_q_name_with_batch', default='', type=str,
                        help='given r2 q table')  # maze/q_learning_seed_42_sweet-water-383.pkl
    parser.add_argument('--best_r2_q_name_without_batch', default='', type=str,
                        help='given r2 q table')  # maze/q_learning_seed_42_Mar-25-2022_14-39-45.pkl
    return parser


def parse_args_cartpole(parent):
    """
    Parse commandline arguments.
    """
    parser = parent
    parser.add_argument('--model-name', default=os.path.join('cartpole', ''),
                        type=str, help='model name')
    parser.add_argument('--gravity', default=9.8, type=float,
                        help='gravity of the world')
    parser.add_argument('--force-mag', default=10, type=float,
                        help='force magnitude applied at each timestep')
    parser.add_argument('--length', default=0.5, type=float,
                        help='pole length')
    parser.add_argument('--masscart', default=1, type=float,
                        help='cart mass')
    parser.add_argument('--masspole', default=0.1, type=float,
                        help='pole mass')
    parser.add_argument('--use-stochastic-reward', default=False, type=bool,
                        help='whether to use stochastic reward or not')
    parser.add_argument('--reward-x-weight', default=1, type=float,
                        help='weight of x distance from x_threshold in stochastic reward')
    parser.add_argument('--reward-angle-weight', default=1, type=float,
                        help='weight of theta distance from theta_threshold in stochastic reward')
    # robust: 'force_mag', 'length'
    parser.add_argument('--cartpole_sensitive_param', default='force_mag', type=str,
                        help='which parameter to use for robust discrete setting')
    return parser


def parse_args_acrobot(parent):
    """
    Parse commandline arguments.
    """
    parser = parent
    parser.add_argument('--model-name', default=os.path.join('acrobot', ''),
                        type=str, help='model name')
    parser.add_argument('--link-length-1', default=1.0,
                        type=float, help='length of first link [meters]')
    parser.add_argument('--link-length-2', default=1.0,
                        type=float, help='length of second link [meters]')
    parser.add_argument('--link-mass-1', default=1.0,
                        type=float, help='mass of first link [kg]')
    parser.add_argument('--link_mass_2', default=1.0,
                        type=float, help='mass of second link [kg]')
    parser.add_argument('--link-com-pos-1', default=1.0,
                        type=float, help='pos of center of mass of first link [m]')
    parser.add_argument('--link_com_pos_2', default=1.0,
                        type=float, help='pos of center of mass of second link [m]')
    parser.add_argument('--link-moi', default=1.0,
                        type=float, help='moments of inertia for both links')

    parser.add_argument('--use-stochastic-reward', default=False, type=bool,
                        help='whether to use stochastic reward or not')
    # robust setting: 'link-mass-2', 'link-com-pos-1', 'link-com-pos-2', 'link-length-1'
    parser.add_argument('--acrobot_sensitive_param', default='link_com_pos_2', type=str,
                        help='which parameter to use for robust discrete setting')
    return parser


def parse_args_mountaincar(parent):
    """
    Parse commandline arguments.
    """
    parser = parent

    parser.add_argument('--model-name', default=os.path.join('mountaincar', ''),
                        type=str, help='model name')
    parser.add_argument('--force', default=0.001,
                        type=float, help='applied force to mountain car')
    parser.add_argument('--gravity', default=0.0025,
                        type=float, help='gravity in mountain car')

    parser.add_argument('--use-stochastic-reward', default=False, type=bool,
                        help='whether to use stochastic reward or not')
    # robust setting: 'gravity'
    parser.add_argument('--mountaincar_sensitive_param', default='gravity', type=str,
                        help='which parameter to use for robust discrete setting')
    return parser


def parse_args_ddqn(parent):
    parser = parent
    parser.add_argument('--EnvIdex', type=int, default=1, help='CP-v1, AC-v0, MC-v0')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or not')
    parser.add_argument('--evaluate', type=str2bool, default=False, help='Evaluate or train')
    # [0] cartpole: length, force_mag. [1] link_com_pos_2
    parser.add_argument('--evaluate_robustness_name', type=str, default='link_com_pos_2')
    parser.add_argument('--evaluate_robustness_every', type=float, default=0)
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or not')
    parser.add_argument('--model_path', type=str, default='DDQN_CPV1_100000.pth', help='Model file name in model dir')
    parser.add_argument('--model_subdir', type=str, default='', help='subdirectory name in model dir')

    parser.add_argument('--Max_train_steps', type=float, default=4e5, help='Max training steps')
    # cartpole 0.035, acrobot 0.001, mountaincar 1e-6
    parser.add_argument('--stop_train_threshold', type=float, default=0, help='training moving exponent threshold')
    parser.add_argument('--save_interval', type=float, default=1.5e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=3e3, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')  # was 200
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='length of sliced trajectory')
    parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
    parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
    parser.add_argument('--scheduler_step', type=float, default=1.5e5, help='step size of scheduler')
    # robust algorithm additions
    parser.add_argument('--algo_type', type=str, default='DDQN', help='one of: DQN, DDQN, R_DDQN (Robust), RO_DDQN (Robust), R2_DDQN, R2_DQN')
    parser.add_argument('--var', type=float, default=1, help='discrete model sampling gaussian variance')
    parser.add_argument('--num_model_samples', type=int, default=5, help='number of discrete models')
    # r2 params
    parser.add_argument('--alpha_p', type=float, default=1e-4, help='alpha^P in the algorithm')
    parser.add_argument('--alpha_r', type=float, default=1e-4, help='alpha^R in the algorithm')
    parser.add_argument('--p_proxy', type=str, default='inner_product', help='inner_product or l1-norm')
    parser.add_argument('--beta', type=float, default=0.99, help='norm estimate moving avg parameter')

    # dynamically update environments
    parser.add_argument('--update_env_every', type=float, default=1e3, help='update every T training steps')  # 3e4
    return parser


def get_sweep_params(env, sweep_name):
    if env == 0:
        # cartpole
        if sweep_name == 'force_mag':
            return [1, 3, 5, 7, 9, 10, 50, 100, 125, 200, 250]
        elif sweep_name == 'length':
            return np.linspace(0.01, 0.45, 7).tolist() + np.linspace(0.5, 3, 10).tolist()
        elif sweep_name == 'masspole':
            return [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5]
        elif sweep_name == 'masscart':
            return [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7, 10, 50]
        elif sweep_name == 'gravity':
            return [0.1, 1, 4, 7, 9.8, 11, 15, 20, 50]
        else:
            raise NotImplemented
    elif env == 1:
        # acrobot
        if sweep_name == 'link_com_pos_2':
            return [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 5]
        elif sweep_name == 'link_mass_2':
            return [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    elif env == 2:
        # mountaincar
        if sweep_name == 'gravity':
            return [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01, 0.02]
        if sweep_name == 'force':
            return [1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2]
    else:
        raise NotImplemented


def get_shock_params(env, sweep_name):
    if env == 0:
        # cartpole
        if sweep_name == 'force_mag':
            return [1, 5, 10, 50, 100, 125]
        if sweep_name == 'length':
            return [0.1, 0.2, 0.5, 0.75, 1, 1.25, 1.5]
        if sweep_name == 'masspole':
            return [0.1, 0.2, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
        raise NotImplemented
    elif env == 1:
        # acrobot
        if sweep_name == 'link_com_pos_2':
            return [0.1, 0.25, 0.5, 1, 1.2]
        if sweep_name == 'link_mass_2':
            return [0.1, 0.25, 0.5, 1, 1.2]
        raise NotImplemented
    elif env == 2:
        # mountaincar
        if sweep_name == 'gravity':
            return [0.001, 0.002, 0.0025, 0.004, 0.006, 0.008]
        if sweep_name == 'force':
            return [0.005e-2, 0.1e-2, 0.15e-2, 0.2e-2, 0.4e-2]
        raise NotImplemented
    else:
        raise NotImplemented
