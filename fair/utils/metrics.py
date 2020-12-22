import numpy as np
from numpy import ndarray

def mae(y_preds: ndarray, y_actual: ndarray) -> float:
    """
    Mean Absoulte Error
    """

    return np.mean(np.abs(y_preds - y_actual))


def rmse(y_preds: ndarray, y_actual: ndarray) -> float:
    """
    Root Mean Squared Error
    """

    return np.sqrt(np.mean(np.power(y_preds - y_actual, 2)))