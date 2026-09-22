import numpy as np
import pytest
from conftest import NOISE, WEAR

from rul.config import UNIT
from rul.datasets import load_dataset
from rul.features import Windows, last_rows, rolling_features


@pytest.fixture
def train(spec):
    return load_dataset(spec).train


def test_rolling_features_do_not_cross_units(train):
    out = rolling_features(train, [WEAR], window=5)
    first = train.groupby(UNIT).head(1).index
    np.testing.assert_allclose(out.loc[first, f"{WEAR}_mean"], train.loc[first, WEAR])
    assert (out.loc[first, f"{WEAR}_std"] == 0).all()
    assert not out.isna().any().any()


def test_rolling_mean_matches_manual(train):
    out = rolling_features(train, [WEAR], window=3)
    g = train[train[UNIT] == train[UNIT].iloc[0]]
    assert out.loc[g.index[5], f"{WEAR}_mean"] == pytest.approx(g[WEAR].iloc[3:6].mean())


@pytest.mark.parametrize("seq_len", [1, 7, 200])
def test_one_window_per_row_ending_at_that_row(train, seq_len):
    feats = [WEAR, NOISE]
    w = Windows(train, feats, seq_len)
    X = w[np.arange(len(w))]
    assert X.shape == (len(train), seq_len, 2)
    np.testing.assert_allclose(X[:, -1, :], train[feats].to_numpy(np.float32))


def test_windows_are_padded_within_their_unit(train):
    w = Windows(train, [WEAR], 10)
    starts = np.flatnonzero(train[UNIT] != train[UNIT].shift())
    for s in starts:
        np.testing.assert_allclose(w[s][0, :, 0], train[WEAR].iloc[s])
        window = w[s + 3][0, :, 0]
        np.testing.assert_allclose(window[:7], train[WEAR].iloc[s])
        np.testing.assert_allclose(window[6:], train[WEAR].iloc[s:s + 4])


def test_last_rows_one_per_unit(train):
    mask = last_rows(train)
    assert mask.sum() == train[UNIT].nunique()
    assert (train.loc[mask, "rul"] == 0).all()
