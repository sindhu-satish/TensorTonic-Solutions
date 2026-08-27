import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    a = np.asarray(A, dtype=float)
    m, n = a.shape
    t = np.zeros((n, m), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            t[j, i] = a[i, j]
    return t
    
