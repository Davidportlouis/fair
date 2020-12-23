import numpy as np
from numpy import ndarray
from .operations import Operation


class Sigmoid(Operation):


    def __init__(self) -> None:

        super().__init__()
    
    def _output(self) -> ndarray:

        return 1.0/(1.0 + np.exp(-self.input_))


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return self.output * (1.0 - self.output) * output_grad


class Linear(Operation):


    def __init__(self) -> None:

        super().__init__()


    def _output(self, inference) -> ndarray:

        return self.input_


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad


class ReLU(Operation):

    def __init__(self) -> None:

        super().__init__()


    def _output(self, inference) -> ndarray:

        return np.maximum(0, self.input_)


    def _input_grad(self, output_grad: ndarray) -> ndarray: 
        return (self.output >= 0)  * output_grad 



class Tanh(Operation):

    def __init__(self) -> None:

        super().__init__()

    
    def _output(self, inference) -> ndarray:

        return np.tanh(self.input_)

    
    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return  (1.0-np.power(self.output, 2)) * output_grad


class LeakyReLU(Operation):

    def __init__(self, alpha: float =  0.2) -> None:

        super().__init__()
        self.alpha: float = alpha

    
    def _output(self, inference) -> ndarray:

        return np.maximum(self.alpha * self.input_, self.input_)

    
    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return (np.ones_like(self.input_) if self.output.all() >= 0 else (np.ones_like(self.input_) * self.alpha)) * output_grad
