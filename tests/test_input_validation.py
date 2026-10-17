import numpy as np
import pytest
from kt.agginc import agginc
from kt.ksdagg import ksdagg
from kt.mmdagg import mmdagg
from kt.mmdfuse import mmdfuse


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_two_sample_methods_reject_nonfinite_samples(bad):
    X = np.zeros((8, 1))
    Y = np.ones((8, 1))
    X[0, 0] = bad
    calls = (
        lambda: mmdagg(X, Y, B1=10, B2=10),
        lambda: mmdfuse(X, Y, number_permutations=10),
        lambda: agginc("mmd", X, Y, R=5, B1=10, B2=10),
    )
    for call in calls:
        with pytest.raises(ValueError, match="finite"):
            call()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_ksdagg_rejects_nonfinite_samples_and_scores(bad):
    X = np.zeros((8, 1))
    score = np.zeros_like(X)
    X[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ksdagg(X, score, B1=10, B2=10)
    X[0, 0] = 0
    score[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ksdagg(X, score, B1=10, B2=10)
    with pytest.raises(ValueError, match="finite"):
        agginc("ksd", X, score, R=5, B1=10, B2=10)


def test_methods_reject_zero_feature_arrays():
    X = np.empty((8, 0))
    calls = (
        lambda: mmdagg(X, X),
        lambda: mmdfuse(X, X),
        lambda: agginc("mmd", X, X),
        lambda: agginc("ksd", X, X),
        lambda: ksdagg(X, X),
    )
    for call in calls:
        with pytest.raises(ValueError, match="at least one feature"):
            call()
