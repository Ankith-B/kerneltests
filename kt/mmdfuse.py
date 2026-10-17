"""MMD-FUSE two-sample test (Biggs, Schrab, Gretton; NeurIPS 2023).

NumPy re-implementation adapted from the MIT-licensed JAX reference package
https://github.com/antoninschrab/mmdfuse (Copyright (c) 2023 Antonin Schrab).
The statistic, bandwidth collection and permutation calibration follow the
reference in the same evaluation order. JAX's random-number generator cannot be
reproduced in NumPy, so the permutations are drawn from ``np.random.RandomState(seed)``;
given the same permutations (private ``_permutations`` hook) the results agree with
the reference up to floating-point rounding (see ``check_mmdfuse_vs_reference.py``).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

_L1 = ("laplace",) + tuple(f"matern_{i}.5_l1" for i in range(5))
_L2 = ("imq", "rq", "gaussian") + tuple(f"matern_{i}.5_l2" for i in range(5))
KERNELS = _L1 + _L2


def _kernel(pairwise: np.ndarray, kernel: str, bandwidth: float, rq_exponent: float = 0.5) -> np.ndarray:
    d = pairwise / bandwidth
    if kernel == "gaussian":
        return np.exp(-(d ** 2) / 2)
    if kernel == "laplace":
        return np.exp(-d * np.sqrt(2))
    if kernel == "rq":
        return (1 + d ** 2 / (2 * rq_exponent)) ** (-rq_exponent)
    if kernel == "imq":
        return (1 + d ** 2) ** (-0.5)
    base = kernel[:10]
    if base == "matern_0.5":
        return np.exp(-d)
    if base == "matern_1.5":
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    if base == "matern_2.5":
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-np.sqrt(5) * d)
    if base == "matern_3.5":
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(-np.sqrt(7) * d)
    return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3
            + (6 ** 4) / 1680 * d ** 4) * np.exp(-3 * d)  # matern_4.5


def _bandwidths(distances: np.ndarray, number: int) -> np.ndarray:
    median = np.median(distances)
    distances = distances + (distances == 0) * median
    dd = np.sort(distances)
    lam_min = dd[int(np.floor(len(dd) * 0.05))] / 2
    lam_max = dd[int(np.floor(len(dd) * 0.95))] * 2
    return np.linspace(lam_min, lam_max, number)


def mmdfuse(X, Y, alpha=0.05, kernels=("laplace", "gaussian"), lambda_multiplier=1,
            number_bandwidths=10, number_permutations=2000, seed=42, return_details=False,
            _permutations=None):
    """Return 1 if MMD-FUSE rejects H0: P_X = P_Y at level alpha, else 0.

    MMD-FUSE fuses normalised MMD estimates over ``kernels`` x ``number_bandwidths``
    bandwidths with a soft maximum (log-mean-exp, temperature ``lambda_multiplier``) and
    calibrates the fused statistic with ``number_permutations`` permutations.

    kernels: a kernel name or tuple of names from ``"laplace"``, ``"gaussian"``, ``"imq"``,
        ``"rq"``, ``"matern_{0..4}.5_l1"``, ``"matern_{0..4}.5_l2"``.
    seed: seeds ``np.random.RandomState`` for the permutations.
    return_details: also return a dict with ``statistic``, ``threshold`` (reject iff
        ``statistic > threshold``), ``p_value``, ``permuted_statistics``, ``kernel``,
        ``bandwidth`` and the normalised single-kernel ``mmd`` values.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2-D arrays with the same number of columns")
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m, n = X.shape[0], Y.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples in each of X and Y")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not lambda_multiplier > 0:
        raise ValueError("lambda_multiplier must be positive")
    if not (isinstance(number_bandwidths, int) and number_bandwidths > 1):
        raise ValueError("number_bandwidths must be an int > 1")
    if not (isinstance(number_permutations, int) and number_permutations > 0):
        raise ValueError("number_permutations must be a positive int")
    if isinstance(kernels, str):
        kernels = (kernels,)
    kernels = tuple(kernels)
    if not kernels or any(k not in KERNELS for k in kernels):
        raise ValueError(f"kernels must be a non-empty collection from {KERNELS}")

    B = number_permutations
    if _permutations is None:
        idx = np.random.RandomState(seed).rand(B, m + n).argsort(axis=1)
    else:
        idx = np.asarray(_permutations)
        if idx.shape != (B, m + n):
            raise ValueError("_permutations must have shape (number_permutations, m + n)")

    def perm(v):
        V = np.empty((B + 1, m + n)); V[:B] = v[idx]; V[B] = v
        return V.T

    V11 = perm(np.r_[np.ones(m), -np.ones(n)])
    V10 = perm(np.r_[np.ones(m), np.zeros(n)])
    V01 = perm(np.r_[np.zeros(m), -np.ones(n)])

    # same ordering as the reference: all l1 kernels first, then l2
    groups = [("cityblock", [k for k in kernels if k in _L1]),
              ("euclidean", [k for k in kernels if k in _L2])]
    N = number_bandwidths * len(kernels)
    M = np.empty((N, B + 1)); bws = np.empty(N); names = []
    Z = np.vstack((X, Y))
    row = 0
    for metric, ks in groups:
        if not ks:
            continue
        P = cdist(Z, Z, metric)
        bandwidths = _bandwidths(P[np.triu_indices(P.shape[0])], number_bandwidths)
        for kern in ks:
            for bw in bandwidths:
                K = _kernel(P, kern, bw)
                np.fill_diagonal(K, 0)
                unscaled_std = np.sqrt(np.sum(K ** 2))
                M[row] = ((np.sum(V10 * (K @ V10), 0) * (n - m + 1) * (n - 1) / (m * (m - 1))
                           + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / m
                           + np.sum(V11 * (K @ V11), 0) * (n - 1) / m)
                          / unscaled_std * np.sqrt(n * (n - 1)))
                bws[row] = bw; names.append(kern); row += 1

    all_stats = logsumexp(lambda_multiplier * M, axis=0, b=1 / N)
    stat = all_stats[-1]
    p_val = np.mean(all_stats >= stat)
    out = int(p_val <= alpha)
    if not return_details:
        return out
    # reject iff p_val <= alpha  <=>  stat > the ceil((B+1)(1-alpha))-th smallest statistic
    k = int(np.ceil((B + 1) * (1 - alpha))) - 1
    threshold = np.sort(all_stats)[k]
    return out, {"statistic": stat, "threshold": threshold, "p_value": p_val,
                 "permuted_statistics": all_stats[:-1], "kernel": names, "bandwidth": bws,
                 "mmd": M[:, -1]}
