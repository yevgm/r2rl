import os
import numpy as np
import torch
from src.dqn.dqn import device


class ReplayBuffer(object):
    def __init__(self, state_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dw = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        with torch.no_grad():
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.Tensor(self.action[ind]).long().to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.dw[ind]).to(self.device)
            )

    def save(self):
        if not os.path.exists('buffer'):
            os.mkdir('buffer')
        scaller = np.array([self.max_size, self.ptr, self.size], dtype=np.uint32)
        np.save("buffer/scaller.npy", scaller)
        np.save("buffer/state.npy", self.state)
        np.save("buffer/action.npy", self.action)
        np.save("buffer/reward.npy", self.reward)
        np.save("buffer/next_state.npy", self.next_state)
        np.save("buffer/dw.npy", self.dw)

    def load(self):
        scaller = np.load("buffer/scaller.npy")

        self.max_size = scaller[0]
        self.ptr = scaller[1]
        self.size = scaller[2]

        self.state = np.load("buffer/state.npy")
        self.action = np.load("buffer/action.npy")
        self.reward = np.load("buffer/reward.npy")
        self.next_state = np.load("buffer/next_state.npy")
        self.dw = np.load("buffer/dw.npy")
