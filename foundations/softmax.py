import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        #METHOD 1 (potentially, max(z) is inefficient since it unpacks the entire vector)
        # z_maxsubtract = z - max(z)
        # exp_z = np.exp(z_maxsubtract)
        # softmax = exp_z / np.sum(exp_z)

        

        # METHOD 2:
        # 1. Make this function z-dimension agnostic
        # 2. Runtime efficient
        
        # Sum over the last dimensional axis and keep the structural shape the same
        # 
        
        shifted = z - np.max(z)
        exps = np.exp(shifted)
        return np.round(exps / np.sum(exps), 4)
