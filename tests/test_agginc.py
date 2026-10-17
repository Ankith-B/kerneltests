import warnings
import numpy as np
import pytest
from kt.agginc import agginc, inc

FAST = dict(R=20, B1=200, B2=200, B3=20)


def _gen(t, rng, n, alt):
    if t == "mmd":
        return rng.standard_normal((n, 2)), rng.standard_normal((n, 2)) + (1.0 if alt else 0.0)
    if t == "hsic":
        X = rng.standard_normal((n, 2)); Y = rng.standard_normal((n, 1))
        if alt: Y[:, 0] = X[:, 0] + 0.3 * Y[:, 0]
        return X, Y
    X = rng.standard_normal((n, 2)) + (1.0 if alt else 0.0)
    return X, -X


@pytest.mark.parametrize("t", ["mmd", "hsic", "ksd"])
def test_power(t):
    rng = np.random.default_rng(1)
    # Default R and B1=B2=500: with 25 HSIC kernels and B1=200, the per-test level
    # alpha/25 falls below 1/(B1+1), so the test can fail to reject (reference too).
    kw = dict(B3=20)
    assert sum(agginc(t, *_gen(t, rng, 400, True), seed=s, **kw) for s in range(10)) >= 9
    assert inc(t, *_gen(t, rng, 400, True), R=200, B=200) == 1


@pytest.mark.parametrize("t", ["mmd", "hsic", "ksd"])
def test_level(t):
    rng = np.random.default_rng(2); Rr = 150; alpha = 0.05
    rej = np.mean([agginc(t, *_gen(t, rng, 100, False), seed=s, alpha=alpha, **FAST) for s in range(Rr)])
    assert rej <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / Rr)


def test_determinism_and_details():
    rng = np.random.default_rng(3); X, Y = _gen("hsic", rng, 120, True)
    a = agginc("hsic", X, Y, seed=7, return_details=True, **FAST)
    b = agginc("hsic", X, Y, seed=7, return_details=True, **FAST)
    assert a[0] == b[0] and np.array_equal(a[1]["statistic"], b[1]["statistic"]) and np.array_equal(a[1]["quantile"], b[1]["quantile"])
    assert len(a[1]["bandwidth_x"]) == 25 and a[1]["reject"].any() == bool(a[0])
    for t in ("mmd", "ksd"):
        X, Y = _gen(t, rng, 80, False)
        o, d = agginc(t, X, Y, return_details=True, weights_type="centred", number_bandwidths=5, batch_size=7, **FAST)
        assert len(d["bandwidth"]) == 5 and d["kernel"] == ("imq" if t == "ksd" else "gaussian")
        assert inc(t, X, Y, seed=1, return_details=True, bandwidth=1.0)[1]["bandwidth"] == 1.0
    o1 = agginc("mmd", X, Y, batch_size=None, **FAST); o2 = agginc("mmd", X, Y, batch_size=13, **FAST)
    assert o1 == o2
    assert inc("hsic", *_gen("hsic", rng, 60, False), bandwidth=(1.0, 2.0), return_details=True)[1]["bandwidth_y"] == 2.0
    o, d = agginc("hsic", *_gen("hsic", rng, 60, False), bandwidths=[[1.0], [1.0, 2.0]], hsic_collection_parameters=(2, -1, 1), return_details=True, **FAST)
    assert len(d["statistic"]) == 2
    o, d = agginc("hsic", *_gen("hsic", rng, 60, False), hsic_collection_parameters=(2, -1, 1, 0, 0), return_details=True, **FAST)
    assert len(d["statistic"]) == 3
    o, d = agginc("mmd", X, Y, bandwidths=[0.5, 1.0], return_details=True, **FAST)
    assert len(d["statistic"]) == 2


def test_warnings():
    rng = np.random.default_rng(4)
    with pytest.warns(UserWarning):
        agginc("hsic", *_gen("hsic", rng, 40, False), number_bandwidths=4, **FAST)
    with pytest.warns(UserWarning):
        agginc("mmd", *_gen("mmd", rng, 40, False), hsic_collection_parameters=(2, -1, 1), **FAST)


X2 = np.zeros((10, 2)) + np.arange(10)[:, None]


@pytest.mark.parametrize("args,kw", [
    (("foo", X2, X2), {}), (("mmd", X2[:, 0], X2), {}), (("mmd", X2, np.ones((10, 3))), {}),
    (("hsic", X2, X2[:8]), {}), (("ksd", X2, X2[:, :1]), {}), (("mmd", X2[:1], X2[:1]), {}),
    (("mmd", X2, X2), {"R": 0}), (("mmd", X2, X2), {"alpha": 1.5}), (("mmd", X2, X2), {"batch_size": -1}),
    (("mmd", X2, X2), {"memory_percentage": 0}), (("mmd", X2, X2), {"B1": 0}),
    (("mmd", X2, X2), {"weights_type": "x"}), (("mmd", X2, X2), {"number_bandwidths": 1}),
    (("hsic", X2, X2), {"weights_type": "decreasing"}), (("hsic", X2, X2), {"hsic_collection_parameters": (2, 1)}),
    (("hsic", X2, X2), {"hsic_collection_parameters": (2, 2, -2)}), (("hsic", X2, X2), {"bandwidths": [[1.0], []]}),
    (("mmd", X2, X2), {"bandwidths": np.ones((2, 2))}),
])
def test_validation_agginc(args, kw):
    with pytest.raises(ValueError):
        agginc(*args, **kw)


@pytest.mark.parametrize("kw", [{"B": 0}, {"bandwidth": (1.0,)}])
def test_validation_inc(kw):
    with pytest.raises(ValueError):
        inc("hsic" if "bandwidth" in kw else "mmd", X2, X2, **kw)
