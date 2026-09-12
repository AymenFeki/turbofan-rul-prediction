"""
Model inputs. Frames are assumed sorted by (unit, time); every model makes
one prediction per row, so all models are evaluated on the same points.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rul.config import UNIT


def rolling_features(df, features, window):
    """Raw values plus per-unit rolling mean and std, for the tabular model."""
    rolled = df.groupby(UNIT)[features].rolling(window, min_periods=1)
    mean = rolled.mean().reset_index(level=0, drop=True).add_suffix("_mean")
    std = rolled.std().reset_index(level=0, drop=True).fillna(0.0).add_suffix("_std")
    return df[features].join(mean).join(std)


class Windows:
    """
    The `seq_len` rows ending at each row of `df`, shape (seq_len, n_features).
    Early rows are front-padded by repeating the unit's first row. Backed by a
    strided view, so batches are only copied when indexed.
    """

    def __init__(self, df, features, seq_len):
        blocks, starts, offset = [], [], 0
        for _, g in df.groupby(UNIT, sort=False):
            values = g[features].to_numpy(np.float32)
            blocks.append(np.concatenate([np.repeat(values[:1], seq_len - 1, axis=0), values]))
            starts.append(offset + np.arange(len(values)))
            offset += len(values) + seq_len - 1
        self._view = sliding_window_view(np.concatenate(blocks), seq_len, axis=0)
        self._starts = np.concatenate(starts)
        self.seq_len = seq_len

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, idx):
        idx = np.atleast_1d(idx)
        return np.ascontiguousarray(self._view[self._starts[idx]].transpose(0, 2, 1))


def last_rows(df):
    """Boolean mask of the final row of every unit."""
    return (df[UNIT] != df[UNIT].shift(-1)).to_numpy()
