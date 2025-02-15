def detect_touch(npz_path, window_size, intensity_threshold=100):
    """
    Checks whether any sliding window along the border of the matrix has a cumulative intensity
    (sum of values) greater than `intensity_threshold`.

    Parameters:
        npz_path (str): Path to the .npz file containing the sparse matrix.
        window_size (int): Size of the sliding window.
        intensity_threshold (float): Threshold for the cumulative intensity of a patch.

    Returns:
        int: 1 if any sliding window along the border exceeds the intensity threshold, 0 otherwise.
    """

    mat = sp.load_npz(npz_path).toarray()
    height, width = mat.shape

    def get_patches(edge):

        """Yield sliding window patches from the specified edge."""

        if edge == "top":
            for c in range(width - window_size + 1):
                yield mat[:window_size, c:c + window_size]
        elif edge == "bottom":
            for c in range(width - window_size + 1):
                yield mat[height - window_size:, c:c + window_size]
        elif edge == "left":
            for r in range(height - window_size + 1):
                yield mat[r:r + window_size, :window_size]
        elif edge == "right":
            for r in range(height - window_size + 1):
                yield mat[r:r + window_size, width - window_size:]
        else:
            raise ValueError("Invalid edge specified.")

    # Check each edge for patches with intensity greater than the threshold.
    for edge in ["top", "bottom", "left", "right"]:
        for patch in get_patches(edge):
            intensity = np.sum(patch)
            if intensity > intensity_threshold:
                print(f"{edge.capitalize()} edge intensity: {intensity}")
                return 1

    return 0
