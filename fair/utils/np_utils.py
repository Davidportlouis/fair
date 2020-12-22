import numpy as np
from numpy import ndarray
from typing import Tuple

def assert_same_shape(arr: ndarray, arr_grad: ndarray) -> None:

    assert arr.shape == arr_grad.shape


def to_2d(arr: ndarray, type_: str = "col") -> ndarray:

    assert arr.ndim == 1
    if type_ == "col":
        return arr.reshape(-1, 1)
    elif type_ == "row":
        return arr.reshape(1, -1)

def permute_data(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:

    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]
