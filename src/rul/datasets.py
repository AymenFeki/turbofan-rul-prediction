"""
Loading a C-MAPSS sub-dataset: a run-to-failure training file, a truncated
test file and the true RUL of each test engine.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rul.config import DATA_DIR, RUL, RUL_CLIP, TIME, UNIT

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = [UNIT, TIME, *SETTINGS, *SENSORS]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_conditions: int = 1
    data_dir: Path = DATA_DIR

    def path(self, kind):
        """kind is "train", "test" or "RUL", as in the NASA file names."""
        return Path(self.data_dir) / f"{kind}_{self.name}.txt"


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray


def read_raw(path):
    """One header-less, space-separated C-MAPSS file."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df, clip=None):
    """RUL = the engine's last cycle minus the current cycle."""
    df = df.copy()
    df[RUL] = df.groupby(UNIT)[TIME].transform("max") - df[TIME]
    if clip is not None:
        df[RUL] = df[RUL].clip(upper=clip)
    return df


def load_dataset(spec):
    train = read_raw(spec.path("train")).sort_values([UNIT, TIME]).reset_index(drop=True)
    test = read_raw(spec.path("test")).sort_values([UNIT, TIME]).reset_index(drop=True)
    test_rul = pd.read_csv(spec.path("RUL"), sep=r"\s+", header=None).iloc[:, 0].to_numpy(float)
    if len(test_rul) != test[UNIT].nunique():
        raise ValueError(
            f"{spec.name}: {len(test_rul)} true RUL values for {test[UNIT].nunique()} test engines"
        )
    return Dataset(add_rul(train, RUL_CLIP), test, np.minimum(test_rul, RUL_CLIP))
