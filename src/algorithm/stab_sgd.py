from torchvision.utils import math
from . import Algorithm, StabOracleAlgorithm
import numpy as np
import torch


class StabSgdOracle(StabOracleAlgorithm):
    shortname = "stab-sgd-oracle"

    def __init__(self, eta):
        self.eta = eta

    def update(self, x, gradient, stab_oracle, s):
        t, = s
        eta_new = self.eta * stab_oracle
        x = x - eta_new * gradient
        return x, (t + 1,)

    def init_state(self, x):
        return (0,)


class StabSgdInline(Algorithm):
    shortname = "stab-sgd-inline"

    variants = {}

    def __init__(self, eta, zeta = 50, gamma = 1, kappa = 1, s0=1, t0=0):
        self.eta = eta
        self.s0 = s0
        self.t0 = t0
        self.zeta = zeta
        self.kappa = kappa
        self.gamma = float(gamma)
        self.eps = 1e-12

    def update(self, x, gradient, s):
        stab, m, v, t, t_next, n_samples, is_stab_time, t_real = s
        t = t + 1
        if is_stab_time:
            m = m + gradient
            v = v + (gradient ** 2).sum()
        else:
            t_real = t_real + 1
            x = x - self.eta * stab * gradient

        if t == t_next:
            if is_stab_time:
                stab1 = ((m**2).sum() / (n_samples * v + self.eps)).item()
                stab = (stab1 - 1/n_samples) / (1 - 1/n_samples)

                clip_low, clip_high = 1e-8, 1.0
                stab = min(max(stab, clip_low), clip_high)
            is_stab_time = not is_stab_time
            m = (0 * x)
            v = 0
            if is_stab_time:
                if not np.isfinite(stab):
                    print(f"ERROR: Got invalid StabRatio {stab}")
                    stab = 1e-8
                n_samples = int(math.ceil(self.zeta / stab))
                t_next = t + n_samples
            else:
                old_t = t_next
                _t_real = max(t_real, self.t0)
                t_next = int(math.ceil(t_next + self.kappa * _t_real ** self.gamma))
                t_next = t_next + 1 if t_next <= old_t else t_next
        return x, (stab, m, v, t, t_next, n_samples, is_stab_time, t_real)

    def init_state(self, x):
        snr = 1
        m = 0 * x
        v = 0
        t = 0
        n_samples = int(math.ceil(self.zeta / self.s0))
        t_next = t + n_samples
        t_real = 1
        is_stab_time = True
        return (snr, m, v, t, t_next, n_samples, is_stab_time, t_real)
