import numpy as np
from numpy import ndarray
from typing import List


class Optimizer:

    def __init__(self, lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str =None) -> None:

        self.lr = lr
        self.first = True
        self.final_lr = final_lr
        self.decay_type = decay_type

    def _setup_decay(self) -> None:

        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr/self.lr, 1.0/(self.max_epochs-1))
        elif self.decay_type == "linear":
            self.decay_per_epoch = (
                self.lr - self.final_lr) / (self.max_epochs-1)

    def _decay_lr(self) -> None:

        if not self.decay_type:
            return
        if self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch
        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01,
                 momentum: float = 0,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:

        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self) -> None:

        if self.momentum:
            if self.first:
                self.velocities: List[ndarray] = [
                    np.zeros_like(param) for param in self.net.params()]
                self.first = False

            for (param, param_grad, velocity) in zip(self.net.params(), self.net.param_grads(), self.velocities):
                self._update_rule(
                    param=param, grad=param_grad, velocity=velocity)

        else:
            for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
                self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

        if self.momentum:
            kwargs["velocity"] *= self.momentum
            kwargs["velocity"] += self.lr * kwargs["grad"]
            kwargs["param"] -= kwargs["velocity"]

        else:
            kwargs["param"] -= self.lr * kwargs["grad"]

   
