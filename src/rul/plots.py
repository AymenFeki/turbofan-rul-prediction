import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rul.config import RUL, TIME, UNIT  # noqa: E402
from rul.metrics import rmse  # noqa: E402

LABELS = {"gb": "Gradient Boosting", "lstm": "LSTM", "cnn": "1D-CNN",
          "transformer": "Transformer"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def lifetimes(train, name, path):
    life = train.groupby(UNIT)[TIME].max()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(life, bins=25, edgecolor="white")
    ax.set(xlabel="lifetime (cycles to failure)", ylabel="engines", title=f"{name}: lifetimes")
    _save(fig, path)


def feature_trends(train, features, name, path, n_features=4, n_units=3):
    """The sensors most correlated with RUL, over a few engines' full lives."""
    corr = train[features].corrwith(train[RUL]).abs().sort_values(ascending=False)
    top = list(corr.index[:n_features])
    units = train[UNIT].unique()[:n_units]
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 2.6 * len(top)), sharex=True,
                             squeeze=False)
    for ax, feat in zip(axes[:, 0], top):
        for u in units:
            g = train[train[UNIT] == u]
            ax.plot(g[TIME], g[feat], linewidth=0.9, alpha=0.8, label=f"engine {u}")
        ax.set_ylabel(f"{feat}\n|r|={corr[feat]:.2f}")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(f"{name}: sensors most correlated with RUL")
    axes[-1, 0].set_xlabel("cycle")
    _save(fig, path)


def predicted_vs_true(preds, name, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4.2), squeeze=False)
    for ax, (key, p) in zip(axes[0], preds.items()):
        lim = [0, 1.05 * max(p["rul_true"].max(), p["rul_pred"].max())]
        ax.scatter(p["rul_true"], p["rul_pred"], s=20, alpha=0.6)
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set(xlim=lim, ylim=lim, xlabel="true RUL", ylabel="predicted RUL",
               title=f"{LABELS.get(key, key)}\nRMSE {rmse(p['rul_true'], p['rul_pred']):.1f}")
    fig.suptitle(name)
    _save(fig, path)


def error_distribution(preds, name, path):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for key, p in preds.items():
        ax.hist(p["rul_pred"] - p["rul_true"], bins=25, alpha=0.5, label=LABELS.get(key, key))
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set(xlabel="error (positive = RUL overestimated)", ylabel="test engines",
           title=f"{name}: prediction errors")
    ax.legend()
    _save(fig, path)


def training_curves(histories, name, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(4.2 * len(histories), 3.8),
                             squeeze=False)
    for ax, (key, h) in zip(axes[0], histories.items()):
        ax.plot(h["train"], label="train")
        ax.plot(h["val"], label="val")
        best = int(np.argmin(h["val"]))
        ax.axvline(best, color="k", linestyle=":", label=f"best epoch {best + 1}")
        ax.set(title=LABELS.get(key, key), xlabel="epoch", ylabel="MSE")
        ax.legend(fontsize=8)
    fig.suptitle(name)
    _save(fig, path)


def shap_summary(values, X, path, bar=False, top_k=15):
    import shap

    plt.figure()
    shap.summary_plot(values, X, max_display=top_k, show=False,
                      plot_type="bar" if bar else None)
    _save(plt.gcf(), path)


def attention(attn, preds, truths, name, path):
    fig, axes = plt.subplots(1, len(attn), figsize=(4 * len(attn), 3.6), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.bar(range(attn.shape[-1]), attn[i].mean(axis=0))
        ax.set(title=f"test engine {i + 1}: pred {preds[i]:.0f}, true {truths[i]:.0f}",
               xlabel="position in window (0 = oldest)")
    axes[0, 0].set_ylabel("mean attention")
    fig.suptitle(name)
    _save(fig, path)
