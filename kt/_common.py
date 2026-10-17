"""Shared helpers. Weights follow Schrab et al. (MIT reference code) in the same
floating-point evaluation order, which keeps results bit-identical to the references."""
import numpy as np

WEIGHTS = ("uniform", "decreasing", "increasing", "centred")


def weights(N: int, weights_type: str) -> np.ndarray:
    if weights_type not in WEIGHTS:
        raise ValueError(f"weights_type must be one of {WEIGHTS}")
    if weights_type == "uniform":
        return np.array([1 / N] * N)
    if weights_type == "decreasing":
        z = sum(1 / k for k in range(1, N + 1))
        return np.array([1 / (k * z) for k in range(1, N + 1)])
    if weights_type == "increasing":
        z = sum(1 / k for k in range(1, N + 1))
        return np.array([1 / ((N + 1 - k) * z) for k in range(1, N + 1)])
    off = 1 if N % 2 == 1 else 0.5
    z = sum(1 / (abs((N + 1) / 2 - k) + off) for k in range(1, N + 1))
    return np.array([1 / ((abs((N + 1) / 2 - k) + off) * z) for k in range(1, N + 1)])
