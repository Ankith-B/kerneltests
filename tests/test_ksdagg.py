import numpy as np
import pytest
from kt.ksdagg import ksdagg


def test_detects_misfit():
    X = np.random.default_rng(0).standard_normal((200, 2)) + 0.7
    out, d = ksdagg(X, -X, return_details=True)
    assert out == 1 and d["reject"].any()
    assert set(d) == {"kernel", "bandwidth", "ksd", "quantile", "p_value", "threshold", "reject"}


def test_level_under_null():
    R, alpha = 150, 0.05
    rng = np.random.default_rng(123)
    rej = 0
    for r in range(R):
        X = rng.standard_normal((60, 2))
        rej += ksdagg(X, -X, alpha=alpha, B1=300, B2=300, B3=30, seed=r)
    se = np.sqrt(alpha * (1 - alpha) / R)
    assert rej / R <= alpha + 3 * se


def test_determinism_and_options():
    X = np.random.default_rng(1).standard_normal((50, 1)) * 1.3
    a = ksdagg(X, -X, seed=7, B1=200, B2=200, return_details=True)
    b = ksdagg(X, -X, seed=7, B1=200, B2=200, return_details=True)
    assert a[0] == b[0] and all(np.array_equal(a[1][k], b[1][k]) for k in ("ksd", "quantile"))
    for wt in ("decreasing", "increasing", "centred"):
        ksdagg(X, -X, weights_type=wt, number_bandwidths=5, B1=100, B2=100)
    ksdagg(X, -X, number_bandwidths=4, weights_type="centred", B1=100, B2=100)
    rng = np.random.default_rng(2)
    out = ksdagg(X, -X, approx_type="parametric", bandwidths=[0.5, 1.0, 2.0],
                 B1=rng.normal(size=(3, 100)), B2=rng.normal(size=(3, 80)))
    assert out in (0, 1)
    ksdagg(X, -X, approx_type="parametric", B1=rng.normal(size=(3, 50)), B2=rng.normal(size=(3, 50)))
    Xc = np.zeros((5, 1)); ksdagg(Xc, Xc, B1=20, B2=20)  # all-identical points


@pytest.mark.parametrize("kw", [
    dict(alpha=0), dict(alpha=1.5), dict(kernel="gaussian"), dict(weights_type="foo"),
    dict(B1=0), dict(B2=1.5), dict(B3=0), dict(number_bandwidths=1), dict(approx_type="x"),
    dict(bandwidths=[[1.0]]), dict(bandwidths=[-1.0, 1.0]),
    dict(approx_type="parametric", B1=np.ones(5), B2=np.ones(5)),
    dict(approx_type="parametric", bandwidths=[1.0, 2.0], B1=np.ones((3, 5)), B2=np.ones((3, 5))),
])
def test_validation(kw):
    X = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError):
        ksdagg(X, -X, **kw)


def test_shape_validation():
    X = np.zeros((10, 2))
    for a, b in ((X, X[:, :1]), (X[:, 0], X[:, 0]), (X[:1], X[:1])):
        with pytest.raises(ValueError):
            ksdagg(a, b)
