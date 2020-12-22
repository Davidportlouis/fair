import numpy as np
from numpy import ndarray
from fair.utils.np_utils import assert_same_shape, softmax

class Loss:

    def __init__(self):

        pass

    
    def forward(self, prediction: ndarray, target: ndarray) -> float:

        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        
        return loss_value

    
    def backward(self) -> ndarray:

        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad


    def _output(self) -> float:

        raise NotImplementedError()

    def _input_grad(self) -> ndarray:

        raise NotImplementedError()


class MSE(Loss):


    def __init__(self):

        super().__init__()


    def _output(self) -> float:

        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]


    def _input_grad(self) -> ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
        
class SCE(Loss):


    def __init__(self, eps: float = 1e-9):
        
        super().__init__()
        self.eps = eps
        self.single_output = False


    def _output(self) -> float:

        softmax_preds = softmax(self.prediction, axis=1)
        # to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1-self.eps)
        loss = - (self.target * np.log(self.softmax_preds) + (1 - self.target) * np.log(1 - self.softmax_preds))

        return np.sum(loss)

    def _input_grad(self) -> ndarray:

        return self.softmax_preds - self.target 