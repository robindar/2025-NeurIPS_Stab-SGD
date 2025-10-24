from . import Algorithm, FirstOrderAlgorithm
import numpy as np
import torch


class RLS(FirstOrderAlgorithm):
    shortname = "rls"

    variants = {
            'rls-b0': { 'beta': 0.0 },
            'rls-b1': { 'beta': 1.0 },
            }

    def __init__(self, eta, beta=0.9):
        self.eta = eta
        self.beta = beta

    def update(self, x, value, gradient, state):
        z, x, y, t = state

        z = z - self.eta * gradient
        x = x + (z - x) / (t + 1)
        y = (1 - self.beta) * z + self.beta * x

        return y, x, (z, x, y, t+1)

    def init_state(self, x):
        y, z = x, x
        return (z, x, y, 0)


class DAdapt(FirstOrderAlgorithm):
    shortname = "dadapt"

    variants = {
            'dadapt-d4': { 'd0': 4 },
            'dadapt-d200': { 'd0': 200 },
            }

    def __init__(self, eta, d0):
        self.eta = eta
        self.d0 = d0
        self.G = 1 / eta

    def update(self, x, value, gradient, state):
        s, d, x, a, weight, gg, lg, t = state

        gg = gg + (gradient ** 2).sum()
        l = d / np.sqrt(gg)
        s = s + l * gradient
        lg = lg + (l ** 2) * (gradient ** 2).sum()
        d_hat = ((s ** 2).sum() - lg) / (2 * np.sqrt((s ** 2).sum()))
        d = max(d, d_hat)
        x = x - l * gradient

        a = a + (x - a) * l / (weight + l)

        return x, a, (s, d, x, a, weight + l, gg, lg, t+1)

    def init_state(self, x):
        s = 0
        d = self.d0
        x = x
        a = x
        gg = self.G ** 2
        lg = 0
        return (s, d, x, a, 0, gg, lg, 0)


class Cocob(FirstOrderAlgorithm):
    shortname = "cocob"

    def __init__(self, eta):
        self.eta = eta
        self.L = 1 / eta

    def zeros_like(self, arr):
        if isinstance(arr, np.ndarray):
            return np.zeros_like(arr)
        else:
            return torch.zeros_like(arr)

    def sigmoid(self, arr):
        if isinstance(arr, np.ndarray):
            return 1 / (1.0 + np.exp(- arr))
        else:
            return torch.sigmoid(arr)

    def update(self, x, value, gradient, state):
        w, init_w, a, g, r, theta, t = state
        neg_gradient = - gradient

        g = g + np.abs(neg_gradient)
        r = r + (w - init_w) * neg_gradient
        theta = theta + neg_gradient
        beta = (2 * self.sigmoid(2 * theta / (g + self.L)) - 1) / self.L
        w = init_w + beta * (self.L + r)

        a = a + (w - a) / (t + 1)

        return w, a, (w, init_w, a, g, r, theta, t+1)

    def init_state(self, x):
        w = x
        init_w = x
        a = x
        g = self.zeros_like(x) + self.L
        r = self.zeros_like(x)
        theta = self.zeros_like(x)
        return (w, init_w, a, g, r, theta, 0)



class CocobBackprop(FirstOrderAlgorithm):
    shortname = "cocob-backprop"

    def __init__(self, eta):
        self.eta = eta

    def zeros_like(self, arr):
        if isinstance(arr, np.ndarray):
            return np.zeros_like(arr)
        else:
            return torch.zeros_like(arr)

    def abs(self, arr):
        if isinstance(arr, np.ndarray):
            return np.abs(arr)
        else:
            return torch.abs(arr)

    def maximum(self, a, b):
        if isinstance(a, np.ndarray):
            return np.maximum(a, b)
        else:
            return torch.maximum(a, b)

    def update(self, x, value, gradient, state):
        w, init_w, g, r, theta, L, t = state
        neg_gradient = - gradient

        L = self.maximum(L, self.abs(neg_gradient))
        alpha = 1 / self.eta

        g = g + np.abs(neg_gradient)

        r = self.maximum(r + (w - init_w) * neg_gradient, self.zeros_like(r))

        theta = theta + neg_gradient
        beta = theta / (L * self.maximum(g + L, alpha * L))
        w = init_w + beta * (L  + r)

        return w, w, (w, init_w, g, r, theta, L, t+1)

    def init_state(self, x):
        w = x
        init_w = x
        g = self.zeros_like(x)
        r = self.zeros_like(x)
        theta = self.zeros_like(x)
        L = self.zeros_like(x)
        return (w, init_w, g, r, theta, L, 0)
