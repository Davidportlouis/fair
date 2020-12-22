import numpy as np
from numpy import ndarray
from typing import List, Tuple
from .operations import Operation, ParamOperation, WeightMultiply, BiasAdd
from .activations import Linear
from fair.utils.np_utils import assert_same_shape

class Layer:

    def __init__(self, neurons: int) -> None:

        self.neurons: int = neurons
        self.first: bool = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []


    def _setup_layer(self, input_: ndarray) -> None:

        raise NotImplementedError()


    def forward(self, input_: ndarray) -> ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_
        
        return self.output


    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        assert_same_shape(self.input_, input_grad)
        self._param_grads()
        
        return input_grad


    def _param_grads(self) -> ndarray:

        self.param_grads: List[ndarray] = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)


    def _param(self) -> ndarray:

        self.param: List[ndarray] = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param.append(operation.param)


class Dense(Layer):


    def __init__(self, neurons: int, activation: Operation = Linear()) -> None:

        super().__init__(neurons)
        self.activation = activation

    
    def _setup_layer(self, input_: ndarray) -> None:

        if self.seed:
            np.random.seed(self.seed)
        self.params: List[ndarray] = []
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        self.params.append(np.random.randn(1, self.neurons))
        self.operations: List[Operation] = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]

    
