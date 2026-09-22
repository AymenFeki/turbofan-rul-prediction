import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from rul.config import RUL, SEED, TIME, TUNING, UNIT, output_dir
from rul.datasets import SENSORS, SETTINGS, DatasetSpec, load_dataset
from rul.estimators import make_estimator
from rul.features import last_rows
from rul.metrics import rmse, summarize
from rul.preprocessing import Preprocessor, split_units


@dataclass
class Split:
    spec: DatasetSpec
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray
    preprocessor: Preprocessor

    @property
    def features(self):
        return self.preprocessor.features_


def prepare(spec):
    """Load a dataset, split train/val by unit and fit preprocessing on train."""
    ds = load_dataset(spec)
    train, val = split_units(ds.train)
    pre = Preprocessor(SENSORS, SETTINGS, spec.n_conditions).fit(train)
    return Split(spec, pre.transform(train), pre.transform(val), pre.transform(ds.test),
                 ds.test_rul, pre)


@dataclass
class RULModel:
    """Everything needed to go from a raw C-MAPSS frame to RUL predictions."""

    preprocessor: Preprocessor
    estimator: object

    def predict(self, df):
        """Prediction at the last observed cycle of every engine."""
        df = df.sort_values([UNIT, TIME]).reset_index(drop=True)
        last = last_rows(df)
        pred = self.estimator.predict(self.preprocessor.transform(df))[last]
        return pd.DataFrame({
            UNIT: df.loc[last, UNIT].to_numpy(),
            TIME: df.loc[last, TIME].to_numpy(),
            "rul_pred": pred,
        })

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_and_evaluate(split, model, params, max_epochs=TUNING["max_epochs"], verbose=False):
    """Fit on the train units, score on val and test, save everything."""
    name = split.spec.name
    training = {} if model == "gb" else {"max_epochs": max_epochs, "verbose": verbose}
    est = make_estimator(model, params, **training)
    est.fit(split.train, split.val, split.features)

    pred = est.predict(split.test)[last_rows(split.test)]
    metrics = {"model": model, "val_rmse": rmse(split.val[RUL], est.predict(split.val)),
               **summarize(split.test_rul, pred), "params": params}
    if hasattr(est, "net"):
        from rul.networks import count_parameters

        metrics["n_params"] = count_parameters(est.net)
        metrics["epochs"] = len(est.history["val"])

    RULModel(split.preprocessor, est).save(output_dir(name, "models") / f"{model}.joblib")
    results = output_dir(name, "results")
    pd.DataFrame({UNIT: split.test.loc[last_rows(split.test), UNIT].to_numpy(),
                  "rul_true": split.test_rul, "rul_pred": pred}
                 ).to_csv(results / f"{model}_predictions.csv", index=False)
    with open(results / f"{model}_metrics.json", "w") as f:
        json.dump(metrics | {"history": getattr(est, "history", None)}, f, indent=2)
    return metrics


def seed_runs(split, model, params, n_seeds=5, max_epochs=TUNING["max_epochs"]):
    """
    Retrain one network with the same parameters and split but different
    random seeds, and score each run on test. Shows how much of a difference
    between models is just run-to-run noise. Nothing is saved.
    """
    rows = []
    for seed in range(SEED, SEED + n_seeds):
        est = make_estimator(model, params, max_epochs=max_epochs, seed=seed)
        est.fit(split.train, split.val, split.features)
        pred = est.predict(split.test)[last_rows(split.test)]
        rows.append({"model": model, "seed": seed, **summarize(split.test_rul, pred)})
    return rows
