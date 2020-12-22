import numpy as np
from numpy import ndarray

from fair.utils.np_utils import assert_same_shape


class Operation:
    
    """
    base class for Operation
    """

    def __init__(self) -> None:

        pass


    def forward(self, input_: ndarray) -> ndarray:

        self.input_ = input_
        self.output = self._output()

        return self.output


    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad


    def _output(self) -> ndarray:

        raise NotImplementedError()


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()



class ParamOperation(Operation):

    """
    base class for Parameterized Operation
    """

    def __init__(self, param: ndarray) -> None:

        super().__init__()
        self.param = param

    
    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    
    def _input_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    """
    weight multiplication operation for neural network
    """

    def __init__(self, W: ndarray) -> None:

        super().__init__(W)


    def _output(self) -> ndarray:

        assert self.input_.shape[1] == self.param.shape[0]
        return np.dot(self.input_, self.param)


    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(output_grad, np.transpose(self.param, (1,0)))

    
    def _param_grad(self, output_grad: ndarray) -> ndarray:

        return np.dot(np.transpose(self.input_, (1,0)), output_grad)


class BiasAdd(ParamOperation):

    """
    bias addition operation for neural network
    """

    def __init__(self, B: ndarray) -> None:

        assert B.shape[0] == 1
        super().__init__(B)

    
    def _output(self) -> ndarray:

        return self.input_ + self.param

    
    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.ones_like(self.input_) * output_grad


    def _param_grad(self, output_grad: ndarray) -> ndarray:

        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])