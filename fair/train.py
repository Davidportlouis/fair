import numpy as np
from numpy import ndarray
from typing import Tuple
from copy import deepcopy
from fair.utils.np_utils import permute_data
from .network import NeuralNetwork
from .optimizers import Optimizer


class Trainer:

    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:

        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(optim, "net", self.net)


    def generate_batches(self, X: ndarray, y: ndarray, size: int = 32) -> Tuple[ndarray, ndarray]:

        assert X.shape[0] == y.shape[0]
        N = X.shape[0]

        for i in range(0, N, size):
            X_batch, y_batch = X[i:i+size], y[i:i+size]
            yield X_batch, y_batch


    def fit(self, X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray, epochs: int = 100, eval_every: int = 10, batch_size: int = 32, seed: int = 1, restart: bool = True) -> None:

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for i, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Epoch: {e+1} Validation loss: {loss:.3f}")
                else:
                    print(f"Loss increased from {self.best_loss:.3f} to {loss:.3f}")
                    self.net = last_model
                    setattr(optim, "net", self.net)
                    break



    