import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(np.matmul(X, weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        
        MSE = np.mean(((model_prediction - ground_truth) ** 2)) # If we provide axis=0 to np.mean,
        # the output will be an array of size (1,) and we would need to index the element to extract the float.
        #MSE = np.sum(((model_prediction - ground_truth) ** 2), axis =0) / ground_truth.shape[0] # calculate the squared errors

        # now, add up the errors and calculate the mean


        return np.round(float(MSE), 5)
