import numpy as np


def _arrays(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def rmse(y_true, y_pred):
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def phm08_score(y_true, y_pred, early=13.0, late=10.0):
    """
    Asymmetric exponential score (Saxena et al., 2008). Late predictions
    (overestimated RUL) cost more than early ones. A sum, so it only compares
    models on the same test set. Lower is better.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / early) - 1, np.exp(d / late) - 1)))


def summarize(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
    }
