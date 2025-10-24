from . import Algorithm, FirstOrderAlgorithm
import torch

class Adam(Algorithm):
    shortname = "adam"

    variants = {}

    def __init__(self, eta, beta1=0.9, beta2=0.9, eps=1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, x, gradient, s):
        m, v, t = s
        beta1, beta2, eta = self.beta1, self.beta2, self.eta
        m = beta1 * m + (1-beta1) * gradient
        v = beta2 * v + (1-beta2) * gradient ** 2
        m_bar = m / (1 - beta1 ** (t+1))
        v_bar = v / (1 - beta2 ** (t+1))
        direction = m_bar / (v_bar + self.eps) ** (1./2)
        x = x - eta * direction
        return x, (m, v, t + 1)

    def init_state(self, x):
        m = 0 * x
        v = 0 * x
        return (m, v, 0)
