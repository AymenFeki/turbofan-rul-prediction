import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from rul.config import SEED, UNIT, VAL_FRAC


def split_units(df, frac=VAL_FRAC, seed=SEED):
    """Split a frame into (train, val) with every unit entirely on one side."""
    units = np.sort(df[UNIT].unique())
    n_val = min(max(1, int(round(frac * len(units)))), len(units) - 1)
    val_units = np.random.default_rng(seed).choice(units, n_val, replace=False)
    is_val = df[UNIT].isin(val_units)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class Preprocessor:
    """
    Assigns an operating condition to each row (KMeans on the setting columns,
    or a single condition), drops features that are constant within any
    condition, and standardises each feature within its condition.
    Fit on training units only.
    """

    def __init__(self, features, settings=(), n_conditions=1, threshold=1e-6, seed=SEED):
        self.features = list(features)
        self.settings = list(settings)
        self.n_conditions = n_conditions
        self.threshold = threshold
        self.seed = seed

    def conditions(self, df):
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(df[self.settings].to_numpy(float))

    def fit(self, df):
        self.kmeans_ = None
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans_.fit(df[self.settings].to_numpy(float))

        grouped = df[self.features].astype(float).groupby(self.conditions(df))
        std = grouped.std(ddof=0)
        self.features_ = [f for f in self.features if (std[f] > self.threshold).all()]
        if not self.features_:
            raise ValueError("no feature varies within every operating condition")
        self.mean_ = grouped.mean()[self.features_]
        self.std_ = std[self.features_]
        return self

    def transform(self, df):
        cond = self.conditions(df)
        df = df.copy()
        values = df[self.features_].to_numpy(float)
        mean = self.mean_.loc[cond].to_numpy()
        std = self.std_.loc[cond].to_numpy()
        df[self.features_] = pd.DataFrame((values - mean) / std, index=df.index)
        df["condition"] = cond
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
