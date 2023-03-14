from abc import abstractmethod
from config import parse_args_cartpole, parse_args_acrobot, parse_args_mountaincar, parse_args_maze
import copy
from src.dqn.dqn import DQN_Agent


class UpdateParams:
    def update_parser(self, parent_parser):
        """
        This function combines both parent parser (dqn) and child parser of the specific environment
        :param parent_parser: argument parser
        :return: the same gym-env
        """
        return NotImplemented

    @staticmethod
    def update_environment(env, args):
        """
        This function updates the gym-env environment variables
        :param env: gym env
        :param args: argparse config
        :return: the same gym-env
        """
        return NotImplemented


class UpdateParamsDynamically:
    def __init__(self, config, param_str_list, param_new_values):
        self.param_str_list = param_str_list
        self.param_new_values = param_new_values
        self.nominal_config = copy.deepcopy(config)
        self.perturbed_config = None
        self.env_updater = None

    def get_perturbed_config(self):
        raise NotImplemented


class UpdateFixedParams(UpdateParamsDynamically):
    def __init__(self, config, param_str_list, param_new_values):
        super().__init__(config, param_str_list, param_new_values)
        self.perturbed_config = self.get_perturbed_config()
        self.curr_config = 0

    def update_environment(self, env):
        if self.curr_config == 0:
            self.env_updater.update_environment(env, self.perturbed_config)
            self.curr_config = 1
        else:
            self.env_updater.update_environment(env, self.nominal_config)
            self.curr_config = 0

    def get_perturbed_config(self):
        perturbed_config = copy.deepcopy(self.nominal_config)
        for idx, param_str in enumerate(self.param_str_list):
            config_dict = vars(perturbed_config)
            config_dict[param_str] = self.param_new_values[idx]

        return perturbed_config

    def reset(self):
        self.curr_config = 0


class UpdateCartpoleParams(UpdateParams):
    def update_parser(self, parent_parser):
        combined_parser = parse_args_cartpole(parent_parser)
        args = combined_parser.parse_known_args()
        return args

    @staticmethod
    def update_environment(env, args):
        env.env.gravity = args.gravity
        env.env.force_mag = args.force_mag
        env.env.length = args.length
        env.env.masscart = args.masscart
        env.env.masspole = args.masspole
        return env


class UpdateAcrobotParams(UpdateParams):
    def update_parser(self, parent_parser):
        combined_parser = parse_args_acrobot(parent_parser)
        args = combined_parser.parse_known_args()
        return args

    @staticmethod
    def update_environment(env, args):
        env.env.LINK_LENGTH_1 = args.link_length_1
        env.env.LINK_LENGTH_2 = args.link_length_2
        env.env.LINK_MASS_1 = args.link_mass_1
        env.env.LINK_MASS_2 = args.link_mass_2
        env.env.LINK_COM_POS_1 = args.link_com_pos_1
        env.env.LINK_COM_POS_2 = args.link_com_pos_2
        env.env.LINK_MOI = args.link_moi
        return env


class UpdateMountaincarParams(UpdateParams):
    def update_parser(self, parent_parser):
        combined_parser = parse_args_mountaincar(parent_parser)
        args = combined_parser.parse_known_args()
        return args

    @staticmethod
    def update_environment(env, args):
        env.env.force = args.force
        env.env.gravity = args.gravity
        return env


class UpdateMazeParams(UpdateParams):
    def update_parser(self, parent_parser):
        combined_parser = parse_args_maze(parent_parser)
        args = combined_parser.parse_known_args()
        return args


class UpdateCartpoleFixedParams(UpdateFixedParams):
    def __init__(self, config, param_str_list, param_new_values):
        super(UpdateCartpoleFixedParams, self).__init__(config, param_str_list, param_new_values)
        self.env_updater = UpdateCartpoleParams


class UpdateAcrobotFixedParams(UpdateFixedParams):
    def __init__(self, config, param_str_list, param_new_values):
        super(UpdateAcrobotFixedParams, self).__init__(config, param_str_list, param_new_values)
        self.env_updater = UpdateAcrobotParams


class UpdateMountaincarFixedParams(UpdateFixedParams):
    def __init__(self, config, param_str_list, param_new_values):
        super(UpdateMountaincarFixedParams, self).__init__(config, param_str_list, param_new_values)
        self.env_updater = UpdateMountaincarParams


def create_robust_dicts(env, args):
    if env == 'CPV1':
        dic_nominal_uncertainty = {'gravity': args.gravity,
                                   'masscart': args.masscart,
                                   'masspole': args.masspole,
                                   'length': args.length,
                                   'force_mag': args.force_mag}
        dic_cardinal_uncertainty = {args.cartpole_sensitive_param: args.num_model_samples}
    elif env == 'ACV0':
        dic_nominal_uncertainty = {'link_length_1': args.link_length_1,
                                   'link_length_2': args.link_length_2,
                                   'link_mass_1': args.link_mass_1,
                                   'link_mass_2': args.link_mass_2,
                                   'link_com_pos_1': args.link_com_pos_1,
                                   'link_com_pos_2': args.link_com_pos_2,
                                   'link_moi': args.link_moi}
        dic_cardinal_uncertainty = {args.acrobot_sensitive_param: args.num_model_samples}
    elif env == 'MCV0':
        dic_nominal_uncertainty = {'gravity': args.gravity,
                                   'force': args.force}
        dic_cardinal_uncertainty = {args.mountaincar_sensitive_param: args.num_model_samples}

    return dic_cardinal_uncertainty, dic_nominal_uncertainty


class ModelGenerator:
    def __init__(self, config, subfolder, model_name:list = []):
        self.config = config
        self.subfolder = subfolder
        self.model_list = model_name
        self.num_models = len(model_name)

    def __len__(self):
        return self.num_models

    @abstractmethod
    def get_model(self):
        pass


class LoadModel(ModelGenerator):
    def get_model(self):
        for model_name in self.model_list:
            self.config.model_path = model_name
            if self.config.EnvIdex == 0:
                state_dim = 4
                action_dim = 2
                env_with_dw = True
            elif self.config.EnvIdex == 1:
                state_dim = 6
                action_dim = 3
                env_with_dw = False
            elif self.config.EnvIdex == 2:
                state_dim = 2
                action_dim = 3
                env_with_dw = True
            else:
                raise NotImplemented

            model_kwargs = {
                "env_with_dw": env_with_dw,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "gamma": self.config.gamma,
                "hid_shape": (self.config.net_width, self.config.net_width),
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "exp_noise": self.config.exp_noise,
                "double_dqn": True,
                'use_wandb': False,
                'use_stochastic_reward': self.config.use_stochastic_reward
            }
            model = DQN_Agent(**model_kwargs)
            model.load_in_notebook(model_path=self.config.model_path, sub_folder=self.subfolder)

            yield model


class TrainModel(ModelGenerator):
    def __init__(self, config, model):
        super().__init__(config, subfolder='', model_name=[])
        self.model = model

    def get_model(self):
        yield self.model

