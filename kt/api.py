"""Unified entry points: one function per testing problem, one result type."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np


@dataclass(frozen=True)
class TestResult:
    """Outcome of a kernel hypothesis test.

    reject  -- True if the null hypothesis is rejected at level ``alpha``.
    method  -- name of the test that was run.
    alpha   -- significance level used.
    details -- per-kernel/bandwidth diagnostics (statistics, quantiles, p-values, ...).
    """
    reject: bool
    method: str
    alpha: float
    details: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __bool__(self) -> bool:
        return self.reject


def _run(fn: Callable, method: str, alpha: float, *args, **kwargs) -> TestResult:
    out, details = fn(*args, alpha=alpha, return_details=True, **kwargs)
    return TestResult(reject=bool(out), method=method, alpha=alpha, details=details)


def two_sample(X, Y, method: str = "mmdagg", alpha: float = 0.05, **kwargs) -> TestResult:
    """Test H0: X and Y are drawn from the same distribution.

    method: "mmdagg" (Schrab et al., JMLR 2023; quadratic time) or
            "mmdagginc" (Schrab et al., NeurIPS 2022; incomplete U-statistics, faster for large n).
    """
    if method == "mmdagg":
        from .mmdagg import mmdagg
        return _run(mmdagg, method, alpha, X, Y, **kwargs)
    if method == "mmdagginc":
        from .agginc import agginc
        return _run(lambda *a, **k: agginc("mmd", *a, **k), method, alpha, X, Y, **kwargs)
    raise ValueError(f"unknown two-sample method {method!r}; available: 'mmdagg', 'mmdagginc'")


def independence(X, Y, method: str = "hsicagginc", alpha: float = 0.05, **kwargs) -> TestResult:
    """Test H0: paired samples X[i], Y[i] come from independent variables.

    method: "hsicagginc" (Schrab et al., NeurIPS 2022).
    """
    if method == "hsicagginc":
        from .agginc import agginc
        return _run(lambda *a, **k: agginc("hsic", *a, **k), method, alpha, X, Y, **kwargs)
    raise ValueError(f"unknown independence method {method!r}; available: 'hsicagginc'")


def goodness_of_fit(X, score_X, method: str = "ksdagg", alpha: float = 0.05, **kwargs) -> TestResult:
    """Test H0: X is drawn from the model whose score function values at X are ``score_X``.

    method: "ksdagg" (Schrab, Guedj, Gretton, NeurIPS 2022) or "ksdagginc" (incomplete U-statistics).
    """
    if method == "ksdagg":
        from .ksdagg import ksdagg
        X = np.asarray(X, dtype=float)
        return _run(ksdagg, method, alpha, X, score_X, **kwargs)
    if method == "ksdagginc":
        from .agginc import agginc
        return _run(lambda *a, **k: agginc("ksd", *a, **k), method, alpha, X, score_X, **kwargs)
    raise ValueError(f"unknown goodness-of-fit method {method!r}; available: 'ksdagg', 'ksdagginc'")
