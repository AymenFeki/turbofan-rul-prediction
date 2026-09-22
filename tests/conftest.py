"""Small synthetic sub-datasets written in the C-MAPSS file format."""

import numpy as np
import pandas as pd
import pytest

from rul.config import TIME, UNIT
from rul.datasets import COLUMNS, DatasetSpec

# Every other sensor is constant, so preprocessing should drop it.
WEAR, NOISE, FLAT = "sensor_1", "sensor_2", "sensor_3"


def make_frame(n_units=12, n_conditions=1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(40, 90))
        for t in range(1, life + 1):
            regime = int(rng.integers(n_conditions))
            row = dict.fromkeys(COLUMNS, 3.0)
            row.update({
                UNIT: u,
                TIME: t,
                "setting_1": 10.0 * regime,
                WEAR: -0.2 * t + 50.0 * regime + rng.normal(0, 0.5),
                NOISE: rng.normal(0, 1),
            })
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def write_dataset(tmp_path, n_units=12, n_conditions=1):
    """Train on one set of engines; test on another, cut at a random cycle."""
    train = make_frame(n_units, n_conditions, seed=0)
    full = make_frame(6, n_conditions, seed=1)
    rng = np.random.default_rng(2)
    parts, rul = [], []
    for _, g in full.groupby(UNIT):
        cut = int(rng.integers(10, len(g)))
        parts.append(g.iloc[:cut])
        rul.append(len(g) - cut)

    spec = DatasetSpec("synthetic", n_conditions, data_dir=tmp_path)
    for kind, df in [("train", train), ("test", pd.concat(parts)), ("RUL", pd.DataFrame(rul))]:
        df.to_csv(spec.path(kind), sep=" ", header=False, index=False)
    return spec


@pytest.fixture
def spec(tmp_path):
    return write_dataset(tmp_path)


@pytest.fixture
def multi_spec(tmp_path):
    return write_dataset(tmp_path, n_units=16, n_conditions=3)


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("rul.config.OUTPUT_ROOT", tmp_path / "outputs")
