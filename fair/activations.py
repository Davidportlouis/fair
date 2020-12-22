import numpy as np
from numpy import ndarray
from .operations import Operation


class Sigmoid(Operation):


    def __init__(self) -> None:

        pass

    
    def _output(self) -> ndarray:

        return 1.0/(1.0 + np.exp(-self.input_))


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return self.output * (1.0 - self.output) * output_grad


class Linear(Operation):


    def __init__(self) -> None:

        pass


    def _output(self) -> ndarray:

        return self.input_


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad