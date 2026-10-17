"""MMDAgg two-sample test (Schrab, Kim, Albert, Laurent, Guedj, Gretton; JMLR 2023).

Re-implementation adapted from the MIT-licensed reference package
https://github.com/antoninschrab/mmdagg (Copyright (c) 2023 Antonin Schrab).
Same random-number protocol as the reference, so results agree exactly for a
given seed; the bisection step and quantile lookup are vectorized, and the
per-test p-value reporting bug of the reference (p_vals[i] vs p_vals[idx]) is fixed.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from ._common import WEIGHTS as _WEIGHTS, weights as _weights

_L1 = ("laplace",) + tuple(f"matern_{i}.5_l1" for i in range(5))
_L2 = ("gaussian", "imq") + tuple(f"matern_{i}.5_l2" for i in range(5))


def _bandwidths(distances: np.ndarray, number: int) -> np.ndarray:
    if np.min(distances) < 1e-1:
        d = np.sort(distances)
        lam_min = np.maximum(d[int(np.floor(len(d) * 0.05))], 1e-1)
    else:
        lam_min = np.min(distances)
    lam_min = lam_min / 2
    lam_max = np.maximum(np.max(distances), 3e-1) * 2
    power = (lam_max / lam_min) ** (1 / (number - 1))
    return np.array([power ** i * lam_min for i in range(number)])


def _kernel(pairwise: np.ndarray, kernel: str, bandwidth: float) -> np.ndarray:
    d = pairwise / bandwidth
    if kernel == "gaussian":
        return np.exp(-d ** 2)
    if kernel == "imq":
        return (1 + d ** 2) ** (-0.5)
    base = "matern_0.5" if kernel == "laplace" else kernel[:10]
    if base == "matern_0.5":
        return np.exp(-d)
    if base == "matern_1.5":
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    if base == "matern_2.5":
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-np.sqrt(5) * d)
    if base == "matern_3.5":
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(-np.sqrt(7) * d)
    if base == "matern_4.5":
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * np.exp(-3 * d)
    raise ValueError(f"unknown kernel {kernel!r}")


def mmdagg(X, Y, alpha=0.05, kernel="laplace_gaussian", number_bandwidths=10,
           weights_type="uniform", B1=2000, B2=2000, B3=50, seed=42, permutations_same_sample_size=False,
           return_details=False, fast=True):
    """Return 1 if MMDAgg rejects H0: P_X = P_Y at level alpha, else 0.

    fast: for equal sample sizes (wild bootstrap), evaluate r^T H r with the n x n matrix
        H = Kxx + Kyy - Kxy - Kyx instead of the reference's 2n x 2n product (about 4x fewer
        flops; mathematically identical, equal to the reference up to floating-point rounding).
        Set fast=False to reproduce the reference bit-for-bit.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2-D arrays with the same number of columns")
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("need at least 2 samples in each of X and Y")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not (isinstance(number_bandwidths, int) and number_bandwidths > 1):
        raise ValueError("number_bandwidths must be an int > 1")
    for name, v in (("B1", B1), ("B2", B2), ("B3", B3)):
        if not (isinstance(v, int) and v > 0):
            raise ValueError(f"{name} must be a positive int")
    wild = (m == n) and not permutations_same_sample_size

    ms = 500
    bw_l1 = _bandwidths(cdist(X[:ms], Y[:ms], "cityblock").ravel(), number_bandwidths)
    bw_l2 = _bandwidths(cdist(X[:ms], Y[:ms], "euclidean").ravel(), number_bandwidths)
    if kernel in _L1:
        kbl = [(kernel, bw_l1, "cityblock")]
    elif kernel in _L2:
        kbl = [(kernel, bw_l2, "euclidean")]
    elif kernel in ("laplace_gaussian", "gaussian_laplace"):
        kbl = [("laplace", bw_l1, "cityblock"), ("gaussian", bw_l2, "euclidean")]
    elif kernel == "all_matern_l1":
        kbl = [(f"matern_{i}.5_l1", bw_l1, "cityblock") for i in range(5)]
    elif kernel == "all_matern_l2":
        kbl = [(f"matern_{i}.5_l2", bw_l2, "euclidean") for i in range(5)]
    elif kernel in ("all_matern_l1_l2", "all"):
        kbl = ([(f"matern_{i}.5_l1", bw_l1, "cityblock") for i in range(5)]
               + [(f"matern_{i}.5_l2", bw_l2, "euclidean") for i in range(5)])
        if kernel == "all":
            kbl += [("gaussian", bw_l2, "euclidean"), ("imq", bw_l2, "euclidean")]
    else:
        raise ValueError(f"unsupported kernel {kernel!r}")
    if weights_type not in _WEIGHTS:
        raise ValueError(f"weights_type must be one of {_WEIGHTS}")
    w = _weights(number_bandwidths, weights_type) / len(kbl)

    rs = np.random.RandomState(seed)
    Btot = B1 + B2 + 1
    if wild:
        R = rs.choice([-1.0, 1.0], size=(Btot, n)); R[B1] = 1.0
        Rn = R.T.copy()
        R = np.concatenate((R.T, -R.T))
    else:
        idx = rs.rand(Btot, m + n).argsort(axis=1)
        def perm(v):
            V = np.take_along_axis(np.tile(v, (Btot, 1)), idx, axis=1); V[B1] = v; return V.T
        V11 = perm(np.r_[np.ones(m), -np.ones(n)])
        V10 = perm(np.r_[np.ones(m), np.zeros(n)])
        V01 = perm(np.r_[np.zeros(m), -np.ones(n)])

    Z = np.vstack((X, Y))
    N = number_bandwidths * len(kbl)
    M = np.empty((N, Btot)); bws = np.empty(N); names = []
    last = None
    for j, (kern, bandwidths, metric) in enumerate(kbl):
        if metric != last:
            P = cdist(Z, Z, metric); last = metric
        for i, bw in enumerate(bandwidths):
            K = _kernel(P, kern, bw)
            np.fill_diagonal(K, 0)
            if wild:
                np.fill_diagonal(K[:n, n:], 0); np.fill_diagonal(K[n:, :n], 0)
                if fast:
                    H = K[:n, :n] + K[n:, n:] - K[:n, n:] - K[n:, :n]
                    M[j * number_bandwidths + i] = np.sum(Rn * (H @ Rn), 0)
                else:
                    M[j * number_bandwidths + i] = np.sum(R * (K @ R), 0)
            else:
                M[j * number_bandwidths + i] = (
                    np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + np.sum(V11 * (K @ V11), 0) / (m * n))
            bws[j * number_bandwidths + i] = bw; names.append(kern)

    stat = M[:, B1]
    M1 = np.sort(M[:, :B1 + 1]); M2 = M[:, B1 + 1:]
    allw = np.tile(w, len(kbl))
    rows = np.arange(N)

    def quantiles(u):
        cols = np.ceil((B1 + 1) * (1 - u * allw)).astype(int) - 1
        return M1[rows, cols]

    u_lo, u_hi = 0.0, float(np.min(1 / w))
    for _ in range(B3):
        u = (u_lo + u_hi) / 2
        if np.mean(np.max(M2 - quantiles(u)[:, None], 0) > 0) <= alpha:
            u_lo = u
        else:
            u_hi = u
    u = u_lo
    q = quantiles(u)
    reject_each = stat > q
    out = int(reject_each.any())
    if not return_details:
        return out
    pvals = np.mean(M1 - stat[:, None] >= 0, -1)
    return out, {"kernel": names, "bandwidth": bws, "mmd": stat, "quantile": q,
                 "p_value": pvals, "threshold": u * allw, "reject": reject_each}
