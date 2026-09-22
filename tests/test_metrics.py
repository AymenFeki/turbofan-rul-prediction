import numpy as np
import pytest

from rul.metrics import mae, phm08_score, rmse, summarize


def test_known_values():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert mae([0, 0], [3, -4]) == pytest.approx(3.5)
    assert rmse([1, 2], [1, 2]) == 0 and phm08_score([1, 2], [1, 2]) == 0


def test_phm08_penalises_late_more_than_early():
    assert phm08_score([50], [60]) > phm08_score([50], [40])
    assert phm08_score([50], [60]) == pytest.approx(np.exp(1) - 1)


def test_summarize_keys():
    assert set(summarize([1, 2], [1, 3])) == {"rmse", "mae", "phm08"}
