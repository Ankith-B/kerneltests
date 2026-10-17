import numpy as np, pytest
from kt.mmdagg import mmdagg

def test_detects_large_shift():
    rng = np.random.default_rng(0)
    assert mmdagg(rng.uniform(size=(200, 5)), rng.uniform(size=(200, 5)) + 1, B1=300, B2=300) == 1

@pytest.mark.parametrize("n", [50, 40])  # wild bootstrap and permutation paths
def test_level_under_null(n):
    rng = np.random.default_rng(1); R = 150
    rej = sum(mmdagg(rng.standard_normal((50, 2)), rng.standard_normal((n, 2)), B1=200, B2=200, B3=25, seed=s) for s in range(R))
    assert rej / R <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / R)

def test_deterministic_given_seed():
    rng = np.random.default_rng(2); X, Y = rng.standard_normal((40, 3)), rng.standard_normal((40, 3)) + .4
    assert mmdagg(X, Y, B1=200, B2=200, seed=7, return_details=True)[1]["quantile"].tolist() == \
           mmdagg(X, Y, B1=200, B2=200, seed=7, return_details=True)[1]["quantile"].tolist()

def test_pvalues_reported_per_kernel():
    rng = np.random.default_rng(3); X, Y = rng.standard_normal((60, 3)), rng.standard_normal((60, 3)) + .5
    p = mmdagg(X, Y, B1=300, B2=300, return_details=True)[1]["p_value"]
    assert not np.array_equal(p[:10], p[10:])   # regression test for the reference's p_vals[i] bug

@pytest.mark.parametrize("bad", [dict(alpha=0), dict(alpha=1), dict(B1=0), dict(number_bandwidths=1), dict(kernel="nope")])
def test_input_validation(bad):
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        mmdagg(rng.standard_normal((10, 2)), rng.standard_normal((10, 2)), **bad)

def test_shape_validation():
    with pytest.raises(ValueError):
        mmdagg(np.zeros((10, 2)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        mmdagg(np.zeros((1, 2)), np.zeros((10, 2)))

from kt._common import weights as _weights
ALL_KERNELS = ["gaussian", "laplace", "imq"] + [f"matern_{i}.5_l{j}" for i in range(5) for j in (1, 2)] + \
    ["laplace_gaussian", "gaussian_laplace", "all_matern_l1", "all_matern_l2", "all_matern_l1_l2", "all"]

@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_every_kernel_detects_shift(kernel):
    rng = np.random.default_rng(6)
    assert mmdagg(rng.standard_normal((80, 2)), rng.standard_normal((80, 2)) + 1.5, kernel=kernel, B1=200, B2=200, B3=20) == 1

@pytest.mark.parametrize("kind", ["uniform", "decreasing", "increasing", "centred"])
@pytest.mark.parametrize("N", [2, 7, 10])
def test_weights_are_probability_vectors(kind, N):
    w = _weights(N, kind)
    assert w.shape == (N,) and np.all(w > 0) and np.isclose(w.sum(), 1.0)

def test_bad_weights_type():
    with pytest.raises(ValueError):
        mmdagg(np.zeros((10, 2)), np.ones((10, 2)), weights_type="nope")

@pytest.mark.parametrize("seed", range(5))
def test_fast_path_matches_exact_path(seed):
    rng = np.random.default_rng(seed); X, Y = rng.standard_normal((40, 3)), rng.standard_normal((40, 3)) + .4
    a = mmdagg(X, Y, B1=200, B2=200, seed=seed, return_details=True, fast=True)
    b = mmdagg(X, Y, B1=200, B2=200, seed=seed, return_details=True, fast=False)
    assert a[0] == b[0] and np.allclose(a[1]["mmd"], b[1]["mmd"], rtol=1e-9) and np.allclose(a[1]["quantile"], b[1]["quantile"], rtol=1e-9)
