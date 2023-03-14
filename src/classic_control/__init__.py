from gym.envs.registration import register

# Classic
# ----------------------------------------


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Cartpole
# ----------------------------------------------------------------------------------------------------------------------#

register(
    id="LocalCartPole-v1",
    entry_point="classic_control.envs:CartPoleEnvLocal",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="LocalCartPoleStochasticReward-v1",
    entry_point="classic_control.envs.cartpole:CartPoleEnvLocalStochasticReward",
    max_episode_steps=500,
    reward_threshold=475.0,
)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Acrobot
# ----------------------------------------------------------------------------------------------------------------------#


register(
    id="LocalAcrobot-v0",
    entry_point="classic_control.envs.acrobot:AcrobotEnvLocal",
    reward_threshold=-100.0,
    max_episode_steps=500,
)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                 Mountain Car
# ----------------------------------------------------------------------------------------------------------------------#

register(
    id="LocalMountainCar-v0",
    entry_point="classic_control.envs.mountain_car:MountainCarEnvLocal",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                 MarsRover
# ----------------------------------------------------------------------------------------------------------------------#

register(
    id='marsrover-v0',
    entry_point='src.classic_control.envs.marsrover:MarsRover',
    max_episode_steps=500,
)
