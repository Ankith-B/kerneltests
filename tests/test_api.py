import numpy as np, pytest
import kt
from kt.mmdagg import mmdagg
from kt.ksdagg import ksdagg


def test_two_sample_matches_direct_call():
    rng = np.random.default_rng(0); X, Y = rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + .6
    res = kt.two_sample(X, Y, B1=200, B2=200, seed=3)
    assert res.reject == bool(mmdagg(X, Y, B1=200, B2=200, seed=3))
    assert res.method == "mmdagg" and res.alpha == 0.05 and "mmd" in res.details


def test_goodness_of_fit_matches_direct_call():
    rng = np.random.default_rng(1); X = rng.standard_normal((80, 2)) + .7
    res = kt.goodness_of_fit(X, -X, B1=200, B2=200, seed=3)
    assert res.reject == bool(ksdagg(X, -X, B1=200, B2=200, seed=3)) and res.reject


def test_result_is_truthy():
    rng = np.random.default_rng(2)
    assert kt.two_sample(rng.uniform(size=(80, 3)), rng.uniform(size=(80, 3)) + 1, B1=200, B2=200)


@pytest.mark.parametrize("call", [lambda: kt.two_sample(np.zeros((5, 1)), np.ones((5, 1)), method="nope"),
                                  lambda: kt.goodness_of_fit(np.zeros((5, 1)), np.zeros((5, 1)), method="nope")])
def test_unknown_method(call):
    with pytest.raises(ValueError):
        call()


def test_independence_detects_dependence():
    rng = np.random.default_rng(4); X = rng.standard_normal((400, 1))
    res = kt.independence(X, X + 0.3 * rng.standard_normal((400, 1)))
    assert res.method == "hsicagginc" and res.reject


def test_inc_methods_route_correctly():
    rng = np.random.default_rng(5); X = rng.standard_normal((300, 2))
    assert kt.two_sample(X, rng.standard_normal((300, 2)) + 1, method="mmdagginc").reject
    assert kt.goodness_of_fit(X + 1, -(X + 1), method="ksdagginc").reject


def test_unknown_independence_method():
    with pytest.raises(ValueError):
        kt.independence(np.zeros((5, 1)), np.zeros((5, 1)), method="nope")
