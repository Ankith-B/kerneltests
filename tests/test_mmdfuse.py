import numpy as np, pytest
import kt
from kt.mmdfuse import mmdfuse, KERNELS


def test_power_mean_shift():
    rng = np.random.default_rng(0)
    rej = [mmdfuse(rng.standard_normal((60, 2)), rng.standard_normal((50, 2)) + 0.8,
                   number_permutations=200, seed=s) for s in range(10)]
    assert np.mean(rej) >= 0.9


def test_power_scale_unequal_sizes():
    rng = np.random.default_rng(1)
    out, det = mmdfuse(rng.standard_normal((150, 1)), 2.5 * rng.standard_normal((80, 1)),
                       number_permutations=300, return_details=True)
    assert out == 1 and det["statistic"] > det["threshold"] and det["p_value"] <= 0.05


def test_level_under_null():
    rng = np.random.default_rng(2); R = 200; alpha = 0.05
    rej = [mmdfuse(rng.standard_normal((30, 2)), rng.standard_normal((25, 2)),
                   number_permutations=100, number_bandwidths=5, seed=r) for r in range(R)]
    assert np.mean(rej) <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / R)


def test_determinism_and_seed():
    rng = np.random.default_rng(3); X, Y = rng.standard_normal((40, 3)), rng.standard_normal((40, 3)) + .3
    a = mmdfuse(X, Y, number_permutations=200, seed=5, return_details=True)[1]
    b = mmdfuse(X, Y, number_permutations=200, seed=5, return_details=True)[1]
    c = mmdfuse(X, Y, number_permutations=200, seed=6, return_details=True)[1]
    assert np.array_equal(a["permuted_statistics"], b["permuted_statistics"]) and a["statistic"] == b["statistic"]
    assert not np.array_equal(a["permuted_statistics"], c["permuted_statistics"])
    assert a["statistic"] == c["statistic"]  # observed statistic does not depend on the seed


def test_swap_symmetry_of_statistic():
    rng = np.random.default_rng(4); X, Y = rng.standard_normal((30, 2)), rng.standard_normal((20, 2))
    s1 = mmdfuse(X, Y, number_permutations=50, return_details=True)[1]["statistic"]
    s2 = mmdfuse(Y, X, number_permutations=50, return_details=True)[1]["statistic"]
    assert s1 == s2


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_runs(kernel):
    rng = np.random.default_rng(5)
    out, det = mmdfuse(rng.standard_normal((25, 2)), rng.standard_normal((25, 2)) + 1.5, kernels=kernel,
                       number_permutations=100, return_details=True)
    assert out == 1 and det["kernel"] == [kernel] * 10 and np.all(np.isfinite(det["mmd"]))


def test_permutation_hook():
    rng = np.random.default_rng(6); X, Y = rng.standard_normal((10, 1)), rng.standard_normal((8, 1))
    idx = np.random.RandomState(42).rand(20, 18).argsort(axis=1)
    assert mmdfuse(X, Y, number_permutations=20, _permutations=idx) == mmdfuse(X, Y, number_permutations=20)
    with pytest.raises(ValueError):
        mmdfuse(X, Y, number_permutations=20, _permutations=idx[:5])


@pytest.mark.parametrize("kw", [dict(alpha=0), dict(alpha=1.0), dict(lambda_multiplier=0),
                                dict(number_bandwidths=1), dict(number_bandwidths=2.0),
                                dict(number_permutations=0), dict(kernels="nope"), dict(kernels=())])
def test_validation(kw):
    X = np.zeros((5, 1)); Y = np.ones((5, 1))
    with pytest.raises(ValueError):
        mmdfuse(X, Y, **kw)


@pytest.mark.parametrize("X,Y", [(np.zeros(5), np.zeros((5, 1))), (np.zeros((5, 2)), np.zeros((5, 1))),
                                 (np.zeros((5, 1)), np.zeros((1, 1)))])
def test_shape_validation(X, Y):
    with pytest.raises(ValueError):
        mmdfuse(X, Y)


def test_api_routing():
    rng = np.random.default_rng(7); X, Y = rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + .7
    res = kt.two_sample(X, Y, method="mmdfuse", number_permutations=300, seed=3)
    assert res.method == "mmdfuse" and res.alpha == 0.05
    assert res.reject == bool(mmdfuse(X, Y, number_permutations=300, seed=3)) and "statistic" in res.details
