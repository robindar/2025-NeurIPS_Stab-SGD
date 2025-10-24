from . import Algorithm
import numpy as np
import torch

class SGD(Algorithm):
    def __init__(self, eta):
        self.eta = eta

    def update(self, x, gradient, s):
        x = x - self.eta * gradient
        return x, s

    def init_state(self, x):
        return None


class SgdWithPowerScheduler(Algorithm):
    shortname = "sgd-sched"

    variants = {
            "sgd-sched-1": { 'alpha': 1.0 },
            "sgd-sched-1p2": { 'alpha': 1 / 2 },
            "sgd-sched-2p3": { 'alpha': 2 / 3 },
            }

    def __init__(self, eta, alpha):
        self.eta = eta
        self.alpha = alpha

    def update(self, x, grad, t):
        rescale = 1 / (1 + t) ** self.alpha
        eta_t = self.eta * rescale
        new_x = x - eta_t * grad
        return new_x, t + 1

    def init_state(self, x):
        return 0
