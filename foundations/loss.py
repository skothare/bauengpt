import numpy as np
from numpy.typing import NDArray
import torch


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        assert y_true.shape == y_pred.shape
        
        # Clip y_pred to prevent log(0) without permanently mutating the input
        # clipping prevents values from dropping below 1e-15 or above (1 - 1e-15)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss_per_sample = -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

        return round(float(np.mean(loss_per_sample)),4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        assert y_true.shape == y_pred.shape
        
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss_per_sample = -np.sum(y_true * np.log(y_pred), axis=-1)
        
        #Take the average across the batch samples
        return round(float(np.mean(loss_per_sample)), 4)