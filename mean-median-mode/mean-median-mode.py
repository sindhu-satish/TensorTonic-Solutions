from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    median = np.median(x)
    freq = Counter(x)
    highest_freq = max(freq.values())
    mode = min(value for value, count in freq.items() if count == highest_freq)
    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode),
    }