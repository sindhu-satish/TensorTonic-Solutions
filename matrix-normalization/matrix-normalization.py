import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # axis = 1 row wise, axis = 0 col wise
    # without keepdims = [3, 7] shape = (2,) 
    # with keepdims = [[3],[7]] shape = (2, 1)
    matrix = np.asarray(matrix, dtype=float)
    if norm_type == "l1":
        divisor_ = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == "l2":
        divisor_ = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
    else:
        divisor_ = np.max(np.abs(matrix), axis=axis, keepdims=True)
    divisor_ = np.where(divisor_ == 0, 1.0, divisor_)
    return matrix / divisor_