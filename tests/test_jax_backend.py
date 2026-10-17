import builtins, importlib, sys
import numpy as np
import pytest

from kt.mmdagg import mmdagg

pytest.importorskip("jax")


@pytest.mark.parametrize("kernel,m,n,shift", [
    ("laplace_gaussian", 40, 40, 0.0), ("laplace_gaussian", 40, 40, 0.8),
    ("all", 30, 24, 0.5), ("imq", 30, 30, 0.3), ("matern_2.5_l1", 25, 35, 0.0)])
def test_parity_with_numpy(kernel, m, n, shift):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((m, 3)); Y = rng.standard_normal((n, 3)) + shift
    kw = dict(kernel=kernel, B1=150, B2=150, B3=20, seed=3, return_details=True)
    no, nd = mmdagg(X, Y, backend="numpy", **kw)
    jo, jd = mmdagg(X, Y, backend="jax", **kw)
    assert no == jo
    assert (nd["reject"] == jd["reject"]).all()
    np.testing.assert_allclose(jd["mmd"], nd["mmd"], rtol=1e-9)
    np.testing.assert_allclose(jd["quantile"], nd["quantile"], rtol=1e-9)
    assert list(nd["kernel"]) == list(jd["kernel"])
    np.testing.assert_array_equal(nd["bandwidth"], jd["bandwidth"])


def test_two_sample_entry_point_and_batching(monkeypatch):
    import kt
    import kt.backends.jax_mmdagg as jm
    monkeypatch.setattr(jm, "_MEM_BUDGET", 1)  # force one bandwidth per vmap batch
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 2)); Y = rng.standard_normal((30, 2)) + 1.0
    r = kt.two_sample(X, Y, B1=100, B2=100, B3=10, backend="jax")
    assert r.reject
    assert r.details["mmd"].shape == (20,)


def test_bad_backend():
    with pytest.raises(ValueError, match="backend"):
        mmdagg(np.zeros((5, 2)), np.ones((5, 2)), backend="torch")


def test_import_error_when_jax_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("No module named 'jax'")
        return real_import(name, *a, **k)
    for mod in [k for k in sys.modules if k == "jax" or k.startswith("jax.") or k == "kt.backends.jax_mmdagg"]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"kerneltests\[jax\]"):
        mmdagg(np.zeros((5, 2)), np.ones((5, 2)), backend="jax")
    # numpy backend unaffected
    assert mmdagg(np.zeros((5, 2)), np.ones((5, 2)) * 5, B1=20, B2=20, B3=5) in (0, 1)
