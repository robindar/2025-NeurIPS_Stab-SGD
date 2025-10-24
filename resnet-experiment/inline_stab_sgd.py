import numpy as np
import torch
from torch.optim import Optimizer



class InlineStabSGD(Optimizer):
    """Stability Ratio guided gradient descent optimizer.

    This implementation alternates between:
    - an stability ratio computation phase (accumulate gradients over many steps)
    - a gradient descent phase (use estimated Stability Ratio to scale updates)

    Parameters
    ----------
    parameters: iterable
        Parameters to optimize (same as torch.optim.Optimizer).
    lr: float
        Base learning rate.
    weight_decay: float, optional
        L2 weight decay (default: 0).
    zeta_start: int, optional
        Initial number of steps used to compute the Stability Ratio (default: 100).
    zeta: float, optional
        Factor controlling how many samples to use for next Stability Ratio computation (default: 100).
    kappa: float, optional
        Factor controlling how many GD steps until next scheduling point (t_next = kappa * t^gamma) (default: 0.1).
    gamma: float, optional
        Factor controlling how many GD steps until next scheduling point (t_next = kappa * t^gamma) (default: 1.).
    eps: float, optional
        Small value to avoid division by zero (default: 1e-8).
    stab_eps: float, optional
        Minimum value for the Stability Ratio (default: 1e-8).
    """

    def __init__(self, parameters, lr, weight_decay=0, zeta_start=100., zeta=100., kappa=0.1, gamma=1., eps=1e-8, stab_eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "zeta_start": zeta_start,
            "zeta": zeta,
            "kappa": kappa,
            "gamma": gamma,
            "eps": eps,
            "stab_eps": stab_eps
        }
        super().__init__(parameters, defaults)

        self.stab_history = []
        self.kurtosis_history = []
        self.total_compute_samples = 0

    def _tensors_to_vector(self, key):
        vector = []
        for group in self.param_groups:
            for param in group['params']:
                tensor = getattr(param, key, None)
                if tensor is not None:
                    vector.append(tensor.view(-1))
        if len(vector) == 0:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(vector)

    def _vector_to_tensors(self, key, vector):
        start_idx = 0
        for group in self.param_groups:
            for param in group['params']:
                tensor = getattr(param, key, None)
                if tensor is not None:
                    numel = tensor.numel()
                    tensor.copy_(vector[start_idx:start_idx + numel].view_as(tensor))
                    start_idx += numel

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = group['lr']
        weight_decay = group['weight_decay']
        zeta = group['zeta']
        zeta_start = group['zeta_start']
        kappa = group['kappa']
        gamma = group['gamma']
        eps = group['eps']
        stab_eps = group['stab_eps']

        state = self.state.setdefault("param", {})

        if len(state) == 0:
            state["stab"] = 1.0
            state["m"] = torch.zeros_like(self._tensors_to_vector('data'))
            state["v"] = 0.0
            state["m4"] = 0.0
            state["t"] = 0
            state["t_gd"] = 0
            state["n_samples"] = group['zeta_start']
            state["t_next"] = group['zeta_start']
            state["is_compute_time"] = True

        data = self._tensors_to_vector('data')
        grad = self._tensors_to_vector('grad')

        t = int(state['t'])
        t_gd = int(state.get('t_gd', 0))
        t_next = int(state['t_next'])
        n_samples = int(state['n_samples'])
        is_compute_time = bool(state['is_compute_time'])
        m = state['m']
        v = float(state['v'])
        m4 = float(state['m4'])

        t += 1

        if is_compute_time:
            m = m + grad
            norm = grad.norm()
            v = v + (norm ** 2).item()
            m4 = m4 + (norm ** 4).item()
            self.total_compute_samples += 1
        else:
            if weight_decay != 0:
                grad = grad + weight_decay * data 
            
            data = data - lr * state['stab'] * grad
            t_gd += 1

        if t == t_next:
            if is_compute_time:
                stab1 = (m.norm() ** 2).item() / (n_samples * v + eps)
                state['stab'] = (stab1 - 1.0 / n_samples) / (1.0 - 1.0 / n_samples)
                state['stab'] = max(state['stab'], stab_eps)
                kurtosis = (m4 / n_samples) / ((v / n_samples) ** 2 + eps)
                state['kurtosis'] = float(kurtosis)
                self.stab_history.append((t, float(state['stab'])))
                self.kurtosis_history.append((t, float(state['kurtosis'])))
                print(f"Step: {t}, Eff. lr: {lr * state['stab']:.3e}, "
                      f"Next compute: t={t_next + int(np.ceil(kappa * t_next**gamma))} "
                      f"with {int(np.ceil(zeta / state['stab']))} samples")
            is_compute_time = not is_compute_time
            m = torch.zeros_like(data)
            v = 0.0
            m4 = 0.0
            if is_compute_time:
                n_samples = int(np.ceil(zeta / state['stab']))
                t_next = t + n_samples
            else:
                t_next += int(np.ceil(kappa * t_next**gamma))

        self._vector_to_tensors('data', data)

        state['t'] = t
        state['t_gd'] = t_gd
        state['t_next'] = t_next
        state['n_samples'] = n_samples
        state['is_compute_time'] = is_compute_time
        state['m'] = m
        state['v'] = v
        state['m4'] = m4

        return loss

