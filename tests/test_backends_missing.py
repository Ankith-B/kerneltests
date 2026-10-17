"""The JAX backend must fail with a helpful ImportError when JAX is absent."""
import sys

import numpy as np
import pytest

import kt
from kt.backends import require_jax


def test_require_jax_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)  # makes `import jax` raise ImportError
    with pytest.raises(ImportError, match="pip install kerneltests\\[jax\\]"):
        require_jax()


def test_mmdagg_jax_backend_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)
    for m in [m for m in sys.modules if m.startswith("kt.backends.jax")]:
        monkeypatch.delitem(sys.modules, m)
    X = np.zeros((10, 1)); Y = np.ones((10, 1))
    with pytest.raises(ImportError):
        kt.two_sample(X, Y, backend="jax")
