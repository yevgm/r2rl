import gym
from gym import spaces

import numpy as np
from numpy import random as random
# import random
import matplotlib.pyplot as plt
# import matplotlib.animation

# import IPython
# from IPython.display import HTML
from typing import Tuple


class MarsRover(gym.Env):
    def __init__(self, width=10, height=10, goal=(2, 9), risk_zone=None,
                 rwd_success=0, rwd_step=-0.01, rwd_fail=-0.02, random_rate=0.05, random_start=False, rwd_variance=0.0001):
        if risk_zone is None:
            self.risk_zone = [(random.randrange(1, int(width/2)+1), random.randrange(1, height)) for _ in range(10)]
            # Original zone from RCPO
            # self.risk_zone = [(1, 20), (2, 7), (2, 11), (2, 17), (2, 19),
            #              (3, 13), (4, 10), (4, 16), (4, 24), (5, 6),
            #              (5, 20), (6, 12), (7, 17), (7, 22), (8, 9),
            #              (8, 14), (9, 7), (9, 18), (10, 22), (12, 12),
            #              (12, 19), (13, 15), (15, 20), (16, 12), (17, 16),
            #                   (17, 22), (19, 5), (22, 10), (28, 19)]
        else:
            self.risk_zone = risk_zone
        # print("Risk zone: ", self.risk_zone)
        self.risk_zone = list(map(tuple, self.risk_zone))
        self.random_rate = random_rate

        self.x_location = self.y_location = None

        self.domain_width = width
        self.domain_height = height
        self.goal = tuple(goal)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.domain_width, self.domain_height])
        self.rwd_success = rwd_success
        self.rwd_step = rwd_step
        self.rwd_fail = rwd_fail
        self.rwd_variance = rwd_variance

        self.history = []  # full grid
        self.action_probabilities = []

        self.epoch = self.step_ep = 0
        self.is_train = bool
        self.random_start = random_start
        self.is_game_over = False

    def reset(self):
        self.epoch += 1
        self.step_ep = 0

        if self.is_train and np.random.rand() < max(0, 1 - (self.epoch * 1e-5)) and self.random_start:
            # print('random starting state')
            self.x_location, self.y_location = self.goal
            while ((self.x_location, self.y_location) == self.goal or
                   (self.x_location, self.y_location) in self.risk_zone):
                self.x_location = np.random.randint(0, self.domain_width)
                self.y_location = np.random.randint(0, self.domain_height)
        else:
            self.x_location = self.y_location = 0   # 1 older version

        self.history = []
        self.action_probabilities = []
        self.is_game_over = False
        # self.create_state()
        return [self.x_location, self.y_location]

    def create_state(self):
        grid = np.zeros((1, 1, self.domain_width + 2, self.domain_height + 2))
        for (x, y) in self.risk_zone:
            grid[0, 0, x, y] = 1
        (x, y) = self.goal
        grid[0, 0, x, y] = 2
        grid[0, 0, self.x_location, self.y_location] = 3
        for x in range(0, self.domain_width + 2):
            grid[0, 0, x, 0] = 4
            grid[0, 0, x, self.domain_height + 1] = 4
        for y in range(0, self.domain_height + 2):
            grid[0, 0, 0, y] = 4
            grid[0, 0, self.domain_width + 1, y] = 4

        self.history.append(grid * 1.0 / 4)
        grid_state = grid * 1.0 / 4

        fig = plt.figure(figsize=(6, 3))
        fig_grid = fig.add_subplot(111)
        fig_grid.matshow(grid_state[0, 0], vmin=-1, vmax=1, cmap='jet')
        plt.show()

        return grid_state

    def step_grid(self, action_command):
        assert self.action_space.contains(action_command), "%r (%s) invalid" % (action_command, type(action_command),)
        a = np.random.rand()
        # print(a, self.random_rate)
        if a < self.random_rate:
            # print('random action')
            action_command = self.action_space.sample()

        if action_command == 0:  # move forward
            self.y_location = min(self.y_location + 1, self.domain_height-1)
        elif action_command == 1:  # move backward
            self.y_location = max(self.y_location - 1, 0)
        elif action_command == 2:  # move right
            self.x_location = min(self.x_location + 1, self.domain_width-1)
        else:  # action_command == 3:  # move left
            self.x_location = max(self.x_location - 1, 0)

        n_state = [self.x_location, self.y_location]
        n_state_t = tuple(n_state)

        info = {'success': True} if n_state_t == self.goal else {'success': False}
        done = True if (n_state_t in self.risk_zone or n_state_t == self.goal) else False
        # print(type(n_state_t), type(n_state), type(self.goal), type(self.risk_zone[0]))
        # print(n_state_t, self.risk_zone)
        if n_state_t in self.risk_zone:
            self.is_game_over = True
            rwd = random.normal(self.rwd_fail, self.rwd_variance)
            # if self.epoch % 10 == 0:
            #     print("Game over! Reached a failure state")
        elif n_state_t == self.goal:
            rwd = random.normal(self.rwd_success, self.rwd_variance)
            # if self.epoch % 10 == 0:
            # print("Success! Reached the goal state")
        else:
            rwd = random.normal(self.rwd_step, self.rwd_variance)
        # print("current reward: ", rwd)
        self.step_ep += 1
        return n_state, rwd, done, info
