import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    X_mean = X - np.mean(X, axis=0)
    cov_ = X_mean.T @ X / (X.shape[0] - 1)
    # diagonal of covariance matrix is variance
    std_dev = np.sqrt(np.diag(cov_))
    divisor_ = np.outer(std_dev, std_dev)
    # do the division, and if some correlations are undefined because of zero standard deviation, return nan without throwing errors
    with np.errstate(divide="ignore", invalid="ignore"):
        return cov_ / divisor_