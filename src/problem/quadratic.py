import torch
import numpy as np

from . import Problem

class QuadraticRebuttal(Problem):
    variants = {
            'QWC': { 'd': 25,  'starting': 'beta', 'decay': 'fast' },
            'QSC': { 'd': 250, 'starting': 'even', 'decay': 'slow' },
    }

    def __init__(
            self,
            d,
            starting,
            decay,
            noise_variance = 0,
            seed = 1,
            zeta = 1,
            ):
        self.d = d
        self.noise_variance = noise_variance
        self.rng = np.random.default_rng(seed=seed)

        _d = np.arange(d)
        assert decay in [ 'fast', 'slow' ]
        if decay == 'fast':
            self.beta = torch.Tensor(1 / (np.float64(2) ** (_d / zeta)))
        elif decay == 'slow':
            self.beta = torch.Tensor(1 / (1 + _d * zeta)).double()

        assert starting in [ 'beta', 'even', 'normal' ]
        if starting == 'beta':
            self.starting_point = self.beta
        elif starting == 'even':
            self.starting_point = (0 * self.beta + 1)
        elif starting == 'normal':
            self.starting_point = self.rng.normal(size=(d,))

    def init_point(self):
        return torch.Tensor(self.starting_point)

    def sample_gradient(self, x):
        grad = self.mean_gradient(x)
        noise = self.rng.normal(size=grad.shape) * np.sqrt(self.noise_variance)
        return grad + noise

    def stab_oracle(self, x):
        grad = self.mean_gradient(x)
        grad_quad = (grad ** 2).sum()
        snr = grad_quad / ( self.noise_variance * self.d + grad_quad )
        return snr

    def mean_gradient(self, x):
        x = x.detach() # Removes information about the gradient
        x.requires_grad = True
        output = self.mean_value(x)
        output.backward()
        grad = x.grad.clone()
        return grad

    def sample_value(self, x):
        noise = self.rng.normal() * np.sqrt(self.noise_variance)
        return self.mean_value(x) + noise

    def mean_value(self, x):
        p = x.double()
        return (self.beta * x ** 2).sum() / 2
