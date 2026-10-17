import numpy as np
import pytest
from kt.agginc import agginc, inc


@pytest.mark.parametrize("agg_type", ["mmd", "hsic", "ksd"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_aggregated_tests_reject_invalid_bandwidths(agg_type, bad):
    X = np.arange(40, dtype=float).reshape(20, 2)
    Y = X.copy() if agg_type == "ksd" else X[:, :1].copy() if agg_type == "hsic" else X + 1
    bandwidths = ([bad], [1.0]) if agg_type == "hsic" else [bad, 1.0]
    with pytest.raises(ValueError, match="positive finite"):
        agginc(agg_type, X, Y, bandwidths=bandwidths)


@pytest.mark.parametrize("agg_type", ["mmd", "hsic", "ksd"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_single_tests_reject_invalid_bandwidths(agg_type, bad):
    X = np.arange(40, dtype=float).reshape(20, 2)
    Y = X.copy() if agg_type == "ksd" else X[:, :1].copy() if agg_type == "hsic" else X + 1
    bandwidth = (bad, 1.0) if agg_type == "hsic" else bad
    with pytest.raises(ValueError, match="positive finite"):
        inc(agg_type, X, Y, bandwidth=bandwidth)


@pytest.mark.parametrize("agg_type", ["mmd", "hsic", "ksd"])
def test_automatic_bandwidths_handle_constant_data(agg_type):
    X = np.zeros((40, 2))
    Y = X.copy() if agg_type == "ksd" else np.zeros((40, 1)) if agg_type == "hsic" else X.copy()
    tests = ((agginc, {"R": 20, "B1": 100, "B2": 100, "B3": 20}),
             (inc, {"R": 20, "B": 100}))
    for test, options in tests:
        _, details = test(agg_type, X, Y, return_details=True, **options)
        for key in ("bandwidth", "bandwidth_x", "bandwidth_y", "statistic", "quantile", "p_value", "threshold"):
            if key in details:
                assert np.all(np.isfinite(details[key]))
