import numpy as np
from numpy import ndarray


class Optimizer:

    def __init__(self, lr: float = 0.01) -> None:

        self.lr = lr

    def step(self) -> None:

        pass


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:

        super().__init__(lr)

    
    def step(self):

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad