import random
import numpy as np
import torch


class ExpMovingAvg:
    def __init__(self, alpha:float = 0.99):
        self._alpha = alpha
        self._cntr = 0
        self._value = 0

    def update(self, current_val:float):
        if self._cntr == 0:
            self._value = current_val
        else:
            self._value = self._value * self._alpha + (1 - self._alpha) * current_val

        self._cntr += 1

    def value(self):
        return self._value


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
