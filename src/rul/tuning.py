"""
Hyperparameter search with Optuna: the same sampler, number of trials,
train/val units and objective (validation RMSE) for every model.
"""

import json

import numpy as np

from rul.config import RUL, SEED, TUNING, output_dir
from rul.estimators import make_estimator
from rul.metrics import rmse

DEFAULTS = {
    "gb": {"window": 10, "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
           "min_samples_leaf": 20, "l2_regularization": 1e-3},
    "lstm": {"seq_len": 30, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-5,
             "dropout": 0.2, "hidden": 64, "layers": 2},
    "cnn": {"seq_len": 30, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-5,
            "dropout": 0.2, "channels": 64, "layers": 2, "kernel": 5},
    "transformer": {"seq_len": 30, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-5,
                    "dropout": 0.1, "d_model": 64, "heads": 4, "layers": 2, "ff_mult": 2},
}


def _sequence_common(trial):
    return {
        "seq_len": trial.suggest_int("seq_len", 10, 80, step=10),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def suggest(trial, model):
    if model == "gb":
        return {
            "window": trial.suggest_int("window", 2, 30),
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
        }
    params = _sequence_common(trial)
    if model == "lstm":
        params |= {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128, 256]),
            "layers": trial.suggest_int("layers", 1, 3),
        }
    elif model == "cnn":
        params |= {
            "channels": trial.suggest_categorical("channels", [16, 32, 64, 128]),
            "layers": trial.suggest_int("layers", 1, 4),
            "kernel": trial.suggest_categorical("kernel", [3, 5, 7]),
        }
    elif model == "transformer":
        params |= {
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "layers": trial.suggest_int("layers", 1, 4),
            "ff_mult": trial.suggest_categorical("ff_mult", [2, 4]),
        }
    else:
        raise ValueError(f"unknown model '{model}'")
    return params


def tune(split, model, dataset, n_trials=TUNING["n_trials"], storage=None,
         max_epochs=TUNING["max_epochs"]):
    """
    Run trials until the study has `n_trials` completed ones (resumes an
    existing study in `storage`), save the best params and return them.
    """
    import optuna

    features = split.features
    y_val = split.val[RUL].to_numpy()

    def objective(trial):
        params = suggest(trial, model)
        est = make_estimator(model, params, max_epochs=max_epochs)
        est.fit(split.train, split.val, features)
        pred = est.predict(split.val)
        if not np.all(np.isfinite(pred)):
            raise optuna.TrialPruned("non-finite predictions")
        return rmse(y_val, pred)

    study = optuna.create_study(
        study_name=f"{dataset}-{model}",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    while done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    best = {"params": study.best_params, "val_rmse": study.best_value,
            "n_trials": done, "trial": study.best_trial.number}
    with open(output_dir(dataset, "params") / f"{model}.json", "w") as f:
        json.dump(best, f, indent=2)
    return study.best_params


def load_params(dataset, model):
    path = output_dir(dataset, "params") / f"{model}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["params"], True
    return DEFAULTS[model], False
