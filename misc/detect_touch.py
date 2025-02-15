import numpy as np
import scipy.sparse as sp
import os


def detect_touch(npz_path, window_size, threshold=10):
    """
    Checks whether any sliding window along the border of the matrix contains more than `threshold` nonzero values.

    Parameters:
        npz_path (str): Path to the .npz file containing the sparse matrix.
        window_size (int): Size of the sliding window.
        threshold (int): The number of nonzero elements that must be exceeded in a window to register a touch.

    Returns:
        int: 1 if any sliding window along the border has more than `threshold` nonzero values, 0 otherwise.
    """
    
    mat = sp.load_npz(npz_path)
    mat = mat.toarray()
    height, width = mat.shape
    
    
    for c in range(width - window_size + 1):
        patch = mat[:window_size, c:c + window_size]
        if np.count_nonzero(patch) > threshold:
            return 1

    for c in range(width - window_size + 1):
        patch = mat[height - window_size:, c:c + window_size]
        if np.count_nonzero(patch) > threshold:
            return 1

    for r in range(height - window_size + 1):
        patch = mat[r:r + window_size, :window_size]
        if np.count_nonzero(patch) > threshold:
            return 1

    for r in range(height - window_size + 1):
        patch = mat[r:r + window_size, width - window_size:]
        if np.count_nonzero(patch) > threshold:
            return 1

    return 0
