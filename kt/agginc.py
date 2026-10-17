"""MMDAggInc, HSICAggInc and KSDAggInc: efficient aggregated kernel tests using
incomplete U-statistics (Schrab, Kim, Guedj, Gretton; NeurIPS 2022).

Re-implementation adapted from the MIT-licensed reference package
https://github.com/antoninschrab/agginc (Copyright (c) 2023 Antonin Schrab),
NumPy version ``agginc/np.py``. Same random-number protocol as the reference
(one ``RandomState(seed).choice([1., -1.], size=(N, B))`` Rademacher draw), so
results agree exactly for a given seed. Index construction, kernel evaluation
across bandwidths and the quantile bisection are vectorized; inputs are
validated with ValueError. Fixed relative to the reference: missing
``import warnings`` (NameError on non-default ``number_bandwidths`` for hsic
or non-default ``hsic_collection_parameters`` for mmd/ksd), and the custom
HSIC bandwidth check testing ``len(bandwidths[0])`` twice instead of
``bandwidths[1]``.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.spatial.distance import cdist, pdist

_TYPES = ("mmd", "hsic", "ksd")
_WEIGHTS = ("uniform", "decreasing", "increasing", "centred")


def _weights(N, kind):
    i = np.arange(1, N + 1)
    if kind == "uniform":
        return np.array([1 / N] * N)
    if kind in ("decreasing", "increasing"):
        h = sum(1 / k for k in range(1, N + 1))
        return 1 / (i * h) if kind == "decreasing" else 1 / ((N + 1 - i) * h)
    if kind == "centred":
        c = np.abs((N + 1) / 2 - i) + (1 if N % 2 == 1 else 0.5)
        return 1 / (c * sum(1 / c))
    raise ValueError(f"weights_type must be one of {_WEIGHTS}")


def _indices(N, R):
    r = np.arange(1, R + 1)
    lens = np.maximum(N - r, 0)
    ii = np.concatenate([np.arange(L) for L in lens]) if lens.sum() else np.zeros(0, int)
    jj = ii + np.repeat(r, lens)
    return ii, jj


def _h_mmd(X, Y, R, bws):
    N = min(X.shape[0], Y.shape[0])
    ii, jj = _indices(N, R)
    sq = [np.linalg.norm(A[ii] - B[jj], axis=1) ** 2 for A, B in ((X, X), (X, Y), (Y, X), (Y, Y))]
    b2 = (np.asarray(bws, dtype=float) ** 2)[:, None]
    k = [np.exp(-s[None, :] / b2) for s in sq]
    return k[0] - k[1] - k[2] + k[3], ii, jj, N


def _h_hsic(X, Y, R, bws):
    N = X.shape[0] // 2
    hx, ii, jj, _ = _h_mmd(X[:N], X[N:], R, bws[0])
    hy = _h_mmd(Y[:N], Y[N:], R, bws[1])[0]
    return (hx[None] * hy[:, None]).reshape(hx.shape[0] * hy.shape[0], -1), ii, jj, N


def _h_ksd(X, S, R, bws, beta=-0.5):
    N, d = X.shape
    ii, jj = _indices(N, R)
    D = X[ii] - X[jj]
    nrm = np.linalg.norm(D, axis=1) ** 2
    si, sj = S[ii], S[jj]
    a = np.einsum("ij,ij->i", si - sj, D, optimize=True)
    c = np.einsum("ij,ij->i", si, sj, optimize=True)
    h = np.zeros((len(bws), len(ii)))
    for r, b in enumerate(bws):  # kept as a loop: identical op order to the reference
        bn = b ** 2 + nrm
        h[r] = (c * bn ** beta - a * 2 * beta * bn ** (beta - 1)
                - 2 * d * beta * bn ** (beta - 1) - 4 * beta * (beta - 1) * nrm * bn ** (beta - 2))
    return h, ii, jj, N


_H = {"mmd": _h_mmd, "hsic": _h_hsic, "ksd": _h_ksd}


def _bootstrap(h, ii, jj, N, B, seed, batch_size, memory_percentage=0.8):
    eps = np.random.RandomState(seed).choice([1.0, -1.0], size=(N, B))
    if batch_size is None:
        bs = B
    elif batch_size == "auto":
        # Reference bug: it divides RAM by the bytes of a length-#bandwidths
        # vector, so "auto" is always one batch of size len(ii) x B (OOM for
        # large n). Size by the real per-column cost instead; batching does not
        # change any result.
        # psutil can over-report memory inside containers, so cap at ~256 MB per batch.
        try:
            import psutil
            mem = min(psutil.virtual_memory().available, 2 ** 28 / memory_percentage)
        except Exception:  # pragma: no cover
            mem = 2 ** 28
        bs = int(max(1, min(B, memory_percentage * mem / (3 * 8 * max(len(ii), 1)))))
    else:
        bs = batch_size
    out = np.zeros((h.shape[0], B))
    for s in range(0, B, bs):
        e = eps[:, s:s + bs]
        out[:, s:s + bs] = h @ (e[ii] * e[jj])
    return out / len(ii), h @ np.ones(h.shape[1]) / len(ii)


def _median(D):
    positive = D[D > 0]
    return np.median(positive) if positive.size else 1.0


def _check_common(agg_type, X, Y, R, alpha, batch_size, memory_percentage, seed):
    if agg_type not in _TYPES:
        raise ValueError('agg_type should be "mmd" or "hsic" or "ksd".')
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D arrays")
    if agg_type == "mmd" and X.shape[1] != Y.shape[1]:
        raise ValueError("mmd: X and Y must have the same number of columns")
    if agg_type == "hsic" and X.shape[0] != Y.shape[0]:
        raise ValueError("hsic: X and Y must have the same number of rows")
    if agg_type == "ksd" and X.shape != Y.shape:
        raise ValueError("ksd: Y (score of X) must have the same shape as X")
    n = min(X.shape[0], Y.shape[0]) // (2 if agg_type == "hsic" else 1)
    if n < 2:
        raise ValueError("too few samples (need >= 2 effective samples; hsic needs >= 4 rows)")
    if not (isinstance(R, (int, np.integer)) and R >= 1):
        raise ValueError("R must be a positive int")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not (batch_size is None or batch_size == "auto"
            or (isinstance(batch_size, (int, np.integer)) and batch_size > 0)):
        raise ValueError('batch_size must be None, "auto" or a positive int')
    if not 0 < memory_percentage <= 1:
        raise ValueError("memory_percentage must be in (0, 1]")
    return X, Y


def agginc(agg_type, X, Y, R=200, alpha=0.05, weights_type="uniform", number_bandwidths=10,
           hsic_collection_parameters=(2, -2, 2), batch_size="auto", memory_percentage=0.8,
           B1=500, B2=500, B3=50, seed=42, return_details=False, bandwidths=None):
    """Aggregated incomplete-U test. Returns 1 if H0 is rejected, else 0.

    agg_type "mmd": two-sample (X, Y); "hsic": independence (paired X, Y);
    "ksd": goodness-of-fit (Y is the score of the model at X). With
    return_details=True returns (output, dict of per-test arrays).
    """
    X, Y = _check_common(agg_type, X, Y, R, alpha, batch_size, memory_percentage, seed)
    for name, v in (("B1", B1), ("B2", B2), ("B3", B3)):
        if not (isinstance(v, (int, np.integer)) and v > 0):
            raise ValueError(f"{name} must be a positive int")
    if weights_type not in _WEIGHTS:
        raise ValueError(f"weights_type must be one of {_WEIGHTS}")
    if number_bandwidths != 10 and agg_type == "hsic":
        warnings.warn("number_bandwidths has no effect for hsic; use hsic_collection_parameters or bandwidths.")
    if agg_type != "hsic" and tuple(hsic_collection_parameters) != (2, -2, 2):
        warnings.warn("hsic_collection_parameters has no effect for mmd/ksd; use number_bandwidths or bandwidths.")

    if bandwidths is not None:
        if agg_type == "hsic":
            if len(bandwidths) != 2 or len(bandwidths[0]) == 0 or len(bandwidths[1]) == 0:
                raise ValueError("hsic bandwidths must be a pair of non-empty sequences")
            bandwidths = [list(bandwidths[0]), list(bandwidths[1])]
        else:
            bandwidths = np.asarray(bandwidths, dtype=float)
            if bandwidths.ndim != 1 or bandwidths.size == 0:
                raise ValueError("bandwidths must be a non-empty 1-D array")
    elif agg_type == "hsic":
        if weights_type != "uniform":
            raise ValueError("hsic requires weights_type='uniform'")
        p = tuple(hsic_collection_parameters)
        if len(p) == 3:
            power, lm, lp = p; lm, lp = (lm, lm), (lp, lp)
        elif len(p) == 5:
            power, lmx, lpx, lmy, lpy = p; lm, lp = (lmx, lmy), (lpx, lpy)
        else:
            raise ValueError("hsic_collection_parameters should have length either 3 or 5.")
        med = (_median(pdist(X[:500], "euclidean")), _median(pdist(Y[:500], "euclidean")))
        bandwidths = [[power ** i * med[j] for i in range(lm[j], lp[j] + 1)] for j in range(2)]
        if not bandwidths[0] or not bandwidths[1]:
            raise ValueError("hsic_collection_parameters give an empty bandwidth collection")
    else:
        if not (isinstance(number_bandwidths, (int, np.integer)) and number_bandwidths > 1):
            raise ValueError("number_bandwidths must be an int > 1")
        if agg_type == "mmd":
            D = cdist(X[:500], Y[:500], "euclidean").reshape(-1)
            if np.min(D) < 1e-1:
                dd = np.sort(D); lmin = np.maximum(dd[int(np.floor(len(dd) * 0.05))], 1e-1)
            else:
                lmin = np.min(D)
            lmin = lmin / 2
            lmax = np.maximum(np.max(D), 3e-1) * 2
            pw = (lmax / lmin) ** (1 / (number_bandwidths - 1))
            bandwidths = np.array([pw ** i * lmin for i in range(number_bandwidths)])
        else:
            D = pdist(X[:500], "euclidean"); D = D[D > 0]
            lmax = np.maximum(np.max(D) if D.size else 2, 2)
            pw = (lmax / 1) ** (1 / (number_bandwidths - 1))
            bandwidths = np.array([pw ** i * 1 / X.shape[1] for i in range(number_bandwidths)])

    bandwidth_values = (np.asarray(bandwidths[0] + bandwidths[1], dtype=float)
                        if agg_type == "hsic" else np.asarray(bandwidths, dtype=float))
    if not np.all(np.isfinite(bandwidth_values)) or np.any(bandwidth_values <= 0):
        raise ValueError("bandwidths must contain only positive finite values")
    h, ii, jj, N = _H[agg_type](X, Y, R, bandwidths)
    boot, stat = _bootstrap(h, ii, jj, N, B1 + B2, seed, batch_size, memory_percentage)
    nb = boot.shape[0]
    w = _weights(nb, weights_type)
    b1 = np.column_stack([boot[:, :B1], stat])
    b1s = np.sort(b1); b2 = boot[:, B1:]
    rows = np.arange(nb)

    def quantiles(u):
        return b1s[rows, np.ceil((B1 + 1) * (1 - u * w)).astype(int) - 1]

    lo, hi = 0.0, np.min(1 / w)
    for _ in range(B3):
        u = (hi + lo) / 2
        if np.sum(np.max(b2 - quantiles(u)[:, None], 0) > 0) / B2 <= alpha:
            lo = u
        else:
            hi = u
    q = quantiles(lo)
    pv = np.mean(b1 - stat[:, None] >= 0, -1)
    rej = stat > q
    out = int(rej.any())
    if not return_details:
        return out
    if agg_type == "hsic":
        bwx = np.array([bandwidths[0][i] for j in range(len(bandwidths[1])) for i in range(len(bandwidths[0]))])
        bwy = np.array([bandwidths[1][j] for j in range(len(bandwidths[1])) for i in range(len(bandwidths[0]))])
        bwinfo = {"bandwidth_x": bwx, "bandwidth_y": bwy}
    else:
        bwinfo = {"bandwidth": np.asarray(bandwidths)}
    return out, {"kernel": "imq" if agg_type == "ksd" else "gaussian", **bwinfo, "statistic": stat,
                 "quantile": q, "p_value": pv, "threshold": lo * w, "reject": rej}


def inc(agg_type, X, Y, R=200, alpha=0.05, batch_size="auto", memory_percentage=0.8, B=500,
        seed=42, return_details=False, bandwidth=None):
    """Single-bandwidth incomplete-U test (median heuristic unless bandwidth given)."""
    X, Y = _check_common(agg_type, X, Y, R, alpha, batch_size, memory_percentage, seed)
    if not (isinstance(B, (int, np.integer)) and B > 0):
        raise ValueError("B must be a positive int")
    if bandwidth is not None:
        if agg_type == "hsic":
            if len(bandwidth) != 2:
                raise ValueError("hsic bandwidth must be a pair (bandwidth_X, bandwidth_Y)")
            bws = (np.array(bandwidth[0]).reshape(1), np.array(bandwidth[1]).reshape(1))
        else:
            bws = np.array(bandwidth).reshape(1)
    elif agg_type == "mmd":
        bws = np.array([_median(cdist(X[:500], Y[:500], "euclidean").reshape(-1))])
    elif agg_type == "hsic":
        bws = (np.array([_median(pdist(X[:500], "euclidean"))]), np.array([_median(pdist(Y[:500], "euclidean"))]))
    else:
        bws = np.array([_median(pdist(X[:500], "euclidean"))])
    bandwidth_values = np.concatenate(bws) if agg_type == "hsic" else np.asarray(bws)
    if not np.all(np.isfinite(bandwidth_values)) or np.any(bandwidth_values <= 0):
        raise ValueError("bandwidth must contain only positive finite values")
    h, ii, jj, N = _H[agg_type](X, Y, R, bws)
    boot, stat = _bootstrap(h, ii, jj, N, B, seed, batch_size, memory_percentage)
    stat = stat[0]
    b1 = np.column_stack([boot, stat])
    q = np.sort(b1)[0, int(np.ceil((B + 1) * (1 - alpha))) - 1]
    pv = np.mean(b1 - stat >= 0, -1)[0]
    out = int(stat > q)
    if not return_details:
        return out
    bwinfo = ({"bandwidth_x": bws[0][0], "bandwidth_y": bws[1][0]} if agg_type == "hsic"
              else {"bandwidth": bws[0]})
    return out, {"kernel": "imq" if agg_type == "ksd" else "gaussian", **bwinfo, "statistic": stat,
                 "quantile": q, "p_value": pv, "threshold": alpha, "reject": bool(stat > q)}
