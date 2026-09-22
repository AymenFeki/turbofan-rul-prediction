import numpy as np
import pandas as pd
import pytest
from conftest import FLAT, WEAR, make_frame

from rul.config import RUL, UNIT
from rul.datasets import COLUMNS, SENSORS, SETTINGS, add_rul, load_dataset
from rul.preprocessing import Preprocessor, split_units


def test_files_are_read_with_cmapss_columns(spec):
    ds = load_dataset(spec)
    assert list(ds.train.columns) == COLUMNS + [RUL]
    assert list(ds.test.columns) == COLUMNS


def test_rul_counts_down_to_zero(spec):
    ds = load_dataset(spec)
    for _, g in ds.train.groupby(UNIT):
        assert g[RUL].iloc[-1] == 0
        assert (np.diff(g[RUL]) == -1).all()


def test_rul_clip():
    assert add_rul(make_frame(), clip=20)[RUL].max() == 20


def test_one_true_rul_per_test_unit(spec):
    ds = load_dataset(spec)
    assert len(ds.test_rul) == ds.test[UNIT].nunique()
    assert (ds.test_rul > 0).all()


def test_mismatched_rul_file_is_rejected(spec):
    pd.DataFrame({"rul": [1, 2]}).to_csv(spec.path("RUL"), header=False, index=False)
    with pytest.raises(ValueError):
        load_dataset(spec)


def test_split_units_is_disjoint_and_complete(spec):
    ds = load_dataset(spec)
    train, val = split_units(ds.train, frac=0.25)
    assert set(train[UNIT]).isdisjoint(val[UNIT])
    assert len(train) + len(val) == len(ds.train)
    assert val[UNIT].nunique() == round(0.25 * ds.train[UNIT].nunique())


def test_constant_feature_is_dropped(spec):
    ds = load_dataset(spec)
    pre = Preprocessor(SENSORS).fit(ds.train)
    assert FLAT not in pre.features_
    assert WEAR in pre.features_


def test_standardised_on_train_statistics_only(spec):
    ds = load_dataset(spec)
    pre = Preprocessor(SENSORS).fit(ds.train)
    out = pre.transform(ds.train)
    assert out[WEAR].mean() == pytest.approx(0, abs=1e-9)
    assert out[WEAR].std(ddof=0) == pytest.approx(1, abs=1e-9)

    shifted = ds.train.assign(**{WEAR: ds.train[WEAR] + 1000})
    assert pre.transform(shifted)[WEAR].mean() > 10


def test_condition_offsets_are_removed(multi_spec):
    ds = load_dataset(multi_spec)
    pre = Preprocessor(SENSORS, SETTINGS, n_conditions=3).fit(ds.train)
    out = pre.transform(ds.train)
    assert out["condition"].nunique() == 3
    means = out.groupby("condition")[WEAR].mean()
    np.testing.assert_allclose(means, 0, atol=1e-9)


def test_inputs_are_not_mutated(spec):
    ds = load_dataset(spec)
    before = ds.train.copy()
    Preprocessor(SENSORS).fit_transform(ds.train)
    pd.testing.assert_frame_equal(ds.train, before)
