"""KSDAgg goodness-of-fit test (Schrab, Guedj, Gretton; NeurIPS 2022).

Re-implementation adapted from the MIT-licensed reference package
https://github.com/antoninschrab/ksdagg (Copyright (c) 2023 Antonin Schrab).
Same random-number protocol and floating-point formulas as the reference
(``ksdagg/np.py``), so results agree exactly for a given seed; the Stein
kernel shares its bandwidth-independent terms, and the bisection step and
quantile lookup are vectorized.

Supported: kernel "imq" (the reference's only kernel), approx_type
"wild bootstrap" and "parametric", all four weights_type values, and a
user-supplied bandwidth collection. Differences from the reference:
ValueError validation instead of asserts; ``bandwidths`` may be any 1-D
array-like (the reference silently ignores non-ndarray inputs); for
"parametric", B1/B2 must have shape (number_bandwidths, B) -- the shape the
reference code actually uses, not the transposed shape in its docstring --
and number_bandwidths is taken from B1.shape[0] when bandwidths is None.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from ._common import WEIGHTS as _WEIGHTS, weights as _weights



def _stein_imq(X, S, bandwidths):
    """Yield the zero-diagonal IMQ Stein kernel matrix (beta = 1/2) per bandwidth."""
    beta = 0.5
    p = X.shape[1]
    norms = np.sum(X ** 2, -1)
    dists = -2 * X @ X.T + norms[:, None] + norms[None, :]
    diffs = np.sum(X * S, -1)[:, None] - X @ S.T
    diffs = diffs + diffs.T
    SS = S @ S.T
    for bw in bandwidths:
        g = 1 / bw ** 2
        res = 1 + g * dists
        r1 = res ** (-beta - 1)
        H = (SS * res ** (-beta) + 2 * beta * g * r1 * diffs
             + 2 * (beta * g * r1 * p
                    - 2 * beta * (beta + 1) * g ** 2 * dists * res ** (-beta - 2)))
        np.fill_diagonal(H, 0)
        yield H


def ksdagg(X, score_X, alpha=0.05, number_bandwidths=10, weights_type="uniform",
           approx_type="wild bootstrap", kernel="imq", B1=2000, B2=2000, B3=50,
           seed=42, bandwidths=None, return_details=False):
    """Return 1 if KSDAgg rejects H0: X ~ model (whose score at X is score_X), else 0."""
    X = np.asarray(X, dtype=float); S = np.asarray(score_X, dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(S)):
        raise ValueError("X and score_X must contain only finite values")
    if X.ndim != 2 or X.shape != S.shape:
        raise ValueError("X and score_X must be 2-D arrays of the same shape")
    if X.shape[1] == 0:
        raise ValueError("X and score_X must contain at least one feature")
    m = X.shape[0]
    if m < 2:
        raise ValueError("need at least 2 samples")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if kernel != "imq":
        raise ValueError(f"unsupported kernel {kernel!r}; only 'imq'")
    if weights_type not in _WEIGHTS:
        raise ValueError(f"weights_type must be one of {_WEIGHTS}")
    if not (isinstance(B3, int) and B3 > 0):
        raise ValueError("B3 must be a positive int")
    if approx_type == "wild bootstrap":
        for name, v in (("B1", B1), ("B2", B2)):
            if not (isinstance(v, int) and v > 0):
                raise ValueError(f"{name} must be a positive int")
        P1 = P2 = None
    elif approx_type == "parametric":
        P1 = np.asarray(B1, dtype=float); P2 = np.asarray(B2, dtype=float)
        if P1.ndim != 2 or P2.ndim != 2 or P1.shape[0] != P2.shape[0] or min(P1.shape + P2.shape) < 1:
            raise ValueError("parametric B1, B2 must be non-empty 2-D arrays (number_bandwidths, B)")
        B1, B2 = P1.shape[1], P2.shape[1]
        if bandwidths is None:
            number_bandwidths = P1.shape[0]
    else:
        raise ValueError("approx_type must be 'wild bootstrap' or 'parametric'")

    if bandwidths is not None:
        bandwidths = np.asarray(bandwidths, dtype=float)
        if (bandwidths.ndim != 1 or len(bandwidths) < 1
                or not np.all(np.isfinite(bandwidths)) or np.any(bandwidths <= 0)):
            raise ValueError("bandwidths must be a non-empty 1-D array of positive finite values")
        number_bandwidths = len(bandwidths)
    if not (isinstance(number_bandwidths, int) and number_bandwidths > 1) and bandwidths is None:
        raise ValueError("number_bandwidths must be an int > 1")
    if P1 is not None and P1.shape[0] != number_bandwidths:
        raise ValueError("parametric B1/B2 must have one row per bandwidth")
    if bandwidths is None:
        d = pdist(X[:500], "euclidean"); d = d[d > 0]
        lam_max = np.maximum(np.max(d), 2) if d.size else 2.0
        power = (lam_max / 1) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * 1 / X.shape[1] for i in range(number_bandwidths)])
    N = number_bandwidths
    w = _weights(N, weights_type)

    if P1 is None:
        rs = np.random.RandomState(seed)
        R = rs.choice([1.0, -1.0], size=(B1 + B2 + 1, m)); R[B1] = 1.0
        R = R.T
        M = np.empty((N, B1 + B2 + 1))
        for i, H in enumerate(_stein_imq(X, S, bandwidths)):
            M[i] = np.sum(R * (H @ R), 0) / (m * (m - 1))
        stat = M[:, B1]
        M1 = np.sort(M[:, :B1 + 1]); M2 = M[:, B1 + 1:]
    else:
        r = np.ones(m)
        stat = np.array([r @ H @ r / (m * (m - 1)) for H in _stein_imq(X, S, bandwidths)])
        M1 = np.sort(np.concatenate((P1, stat[:, None]), axis=1)); M2 = P2

    rows = np.arange(N)
    def quantiles(u):
        return M1[rows, np.ceil((B1 + 1) * (1 - u * w)).astype(int) - 1]

    u_lo, u_hi = 0, np.min(1 / w)
    for _ in range(B3):
        u = (u_hi + u_lo) / 2
        if np.sum(np.max(M2 - quantiles(u)[:, None], 0) > 0) / B2 <= alpha:
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
    return out, {"kernel": ["imq"] * N, "bandwidth": np.asarray(bandwidths, float), "ksd": stat,
                 "quantile": q, "p_value": pvals, "threshold": u * w, "reject": reject_each}
