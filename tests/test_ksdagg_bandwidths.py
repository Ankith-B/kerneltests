import numpy as np
import pytest
from kt.ksdagg import ksdagg


@pytest.mark.parametrize("bandwidth", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_ksdagg_rejects_invalid_bandwidths(bandwidth):
    X = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive finite"):
        ksdagg(X, -X, bandwidths=[bandwidth])
