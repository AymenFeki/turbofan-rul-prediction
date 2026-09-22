"""
One interface for every model: fit(train, val, features) and predict(df),
which returns one RUL prediction per row of df, floored at 0 because
remaining life cannot be negative. Tuning, training and prediction all go
through predict, so they all see the same numbers.
"""

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from rul.config import RUL, SEED, TUNING
from rul.features import Windows, rolling_features


def get_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GradientBoosting:
    def __init__(self, window=5, **params):
        self.window = window
        self.params = params

    def _design(self, df):
        return rolling_features(df, self.features, self.window)

    def fit(self, train, val, features):
        self.features = list(features)
        X = self._design(train)
        self.model = HistGradientBoostingRegressor(
            early_stopping=False, random_state=SEED, **self.params
        ).fit(X, train[RUL])
        return self

    def predict(self, df):
        return np.maximum(self.model.predict(self._design(df)), 0.0)


class SequenceModel:
    """LSTM / CNN / Transformer with Adam(W), MSE loss and early stopping on val."""

    def __init__(self, arch, seq_len=30, lr=1e-3, batch_size=64, weight_decay=0.0,
                 max_epochs=TUNING["max_epochs"], patience=TUNING["patience"],
                 seed=SEED, verbose=False, **net_params):
        self.arch = arch
        self.seq_len = seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.net_params = net_params

    def build(self, n_features):
        from rul.networks import build_network

        return build_network(self.arch, n_features, **self.net_params)

    def _batches(self, windows, y, shuffle, rng=None):
        import torch

        order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            X = torch.from_numpy(windows[idx]).to(self.device)
            yield X, (None if y is None else torch.from_numpy(y[idx]).to(self.device))

    def _run_epoch(self, windows, y, optimizer=None, rng=None):
        import torch

        training = optimizer is not None
        self.net.train(training)
        total = 0.0
        with torch.set_grad_enabled(training):
            for X, target in self._batches(windows, y, shuffle=training, rng=rng):
                loss = torch.nn.functional.mse_loss(self.net(X), target)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(X)
        return total / len(windows)

    def fit(self, train, val, features):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.features = list(features)
        self.device = get_device()
        self.net = self.build(len(features)).to(self.device)

        w_train, y_train = Windows(train, features, self.seq_len), train[RUL].to_numpy(np.float32)
        w_val, y_val = Windows(val, features, self.seq_len), val[RUL].to_numpy(np.float32)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best, best_state, stale = np.inf, None, 0
        self.history = {"train": [], "val": []}
        for epoch in range(self.max_epochs):
            train_loss = self._run_epoch(w_train, y_train, optimizer, rng)
            val_loss = self._run_epoch(w_val, y_val)
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if self.verbose:
                print(f"    epoch {epoch + 1:3d}  train MSE {train_loss:8.2f}  "
                      f"val MSE {val_loss:8.2f}")
            if val_loss < best - 1e-4:
                best, best_state, stale = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to("cpu").eval()
        self.device = "cpu"
        return self

    def predict(self, df):
        import torch

        self.device = get_device()
        self.net.to(self.device).eval()
        windows = Windows(df, self.features, self.seq_len)
        with torch.no_grad():
            out = [self.net(X).cpu().numpy() for X, _ in self._batches(windows, None, False)]
        self.net.to("cpu")
        self.device = "cpu"
        return np.maximum(np.concatenate(out), 0.0)


def make_estimator(name, params=None, **training):
    params = dict(params or {})
    if name == "gb":
        return GradientBoosting(**params)
    return SequenceModel(name, **params, **training)
