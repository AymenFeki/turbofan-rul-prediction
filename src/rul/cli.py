"""
    rul explore --dataset FD001
    rul tune    --dataset FD001 FD002 --model all --n-trials 30
    rul train   --dataset all --model all
    rul compare --dataset all
    rul explain --dataset FD001
    rul seeds   --dataset all --n-seeds 5                 retrain networks with several seeds
    rul run     --dataset all                     explore -> tune -> train -> compare -> explain
    rul predict --model outputs/FD001/models/gb.joblib --input CMAPSSData/test_FD001.txt
"""

import argparse
import json

import numpy as np
import pandas as pd

from rul import plots
from rul.config import DATASETS, MODELS, OUTPUT_ROOT, TUNING, UNIT, output_dir
from rul.datasets import SENSORS, SETTINGS, DatasetSpec, load_dataset, read_raw
from rul.features import last_rows, rolling_features
from rul.pipeline import RULModel, prepare, seed_runs, train_and_evaluate
from rul.preprocessing import Preprocessor
from rul.tuning import load_params, tune


def _datasets(names):
    if names == ["all"]:
        names = list(DATASETS)
    missing = set(names) - set(DATASETS)
    if missing:
        raise SystemExit(f"unknown dataset(s) {sorted(missing)}. Available: {list(DATASETS)}")
    return [DatasetSpec(n, DATASETS[n]) for n in names]


def _models(names):
    return MODELS if names == ["all"] else names


def cmd_explore(args):
    for spec in _datasets(args.dataset):
        ds = load_dataset(spec)
        pre = Preprocessor(SENSORS, SETTINGS, spec.n_conditions).fit(ds.train)
        life = ds.train.groupby(UNIT).size()
        print(f"=== {spec.name} ===")
        print(f"train: {len(ds.train):,} rows, {len(life)} engines, "
              f"lifetime {life.min()}-{life.max()} (median {life.median():.0f})")
        print(f"test:  {len(ds.test):,} rows, {ds.test[UNIT].nunique()} engines")
        print(f"conditions: {spec.n_conditions}, sensors kept {len(pre.features_)}"
              f"/{len(SENSORS)}, dropped {[s for s in SENSORS if s not in pre.features_]}")
        figs = output_dir(spec.name, "figures")
        plots.lifetimes(ds.train, spec.name, figs / "lifetimes.png")
        plots.feature_trends(pre.transform(ds.train), pre.features_, spec.name,
                             figs / "feature_trends.png")


def cmd_tune(args):
    storage = f"sqlite:///{OUTPUT_ROOT / 'optuna.db'}"
    OUTPUT_ROOT.mkdir(exist_ok=True)
    for spec in _datasets(args.dataset):
        split = prepare(spec)
        for model in _models(args.model):
            print(f"\n=== tuning {spec.name} / {model} ({args.n_trials} trials) ===")
            best = tune(split, model, spec.name, n_trials=args.n_trials, storage=storage,
                        max_epochs=args.max_epochs)
            print(f"best: {best}")


def cmd_train(args):
    for spec in _datasets(args.dataset):
        split = prepare(spec)
        print(f"\n=== {spec.name}: {len(split.features)} features, "
              f"{split.train[UNIT].nunique()} train / {split.val[UNIT].nunique()} val / "
              f"{len(split.test_rul)} test engines ===")
        for model in _models(args.model):
            params, tuned = load_params(spec.name, model)
            if not tuned:
                print(f"  {model}: no tuned params found, using defaults")
            m = train_and_evaluate(split, model, params, max_epochs=args.max_epochs,
                                   verbose=args.verbose)
            print(f"  {model:<12} test RMSE {m['rmse']:6.2f}  MAE {m['mae']:6.2f}  "
                  f"PHM08 {m['phm08']:9.1f}")


def _load_results(name):
    results = output_dir(name, "results")
    preds, metrics = {}, {}
    for model in MODELS:
        if (results / f"{model}_predictions.csv").exists():
            preds[model] = pd.read_csv(results / f"{model}_predictions.csv")
            with open(results / f"{model}_metrics.json") as f:
                metrics[model] = json.load(f)
    return preds, metrics


def cmd_compare(args):
    for spec in _datasets(args.dataset):
        preds, metrics = _load_results(spec.name)
        if not preds:
            print(f"{spec.name}: no results, run `rul train` first")
            continue
        rows = sorted(metrics.values(), key=lambda m: m["rmse"])
        lines = ["| Model | Params | Val RMSE | Test RMSE | MAE | PHM08 |",
                 "|---|---|---|---|---|---|"]
        for m in rows:
            lines.append(
                f"| {plots.LABELS[m['model']]} | {m.get('n_params', '-')} | {m['val_rmse']:.2f} | "
                f"{m['rmse']:.2f} | {m['mae']:.2f} | {m['phm08']:.1f} |")
        table = "\n".join(lines)
        print(f"\n=== {spec.name} ===\n{table}")
        (output_dir(spec.name, "results") / "summary.md").write_text(table + "\n")

        figs = output_dir(spec.name, "figures")
        plots.predicted_vs_true(preds, spec.name, figs / "predicted_vs_true.png")
        plots.error_distribution(preds, spec.name, figs / "error_distribution.png")
        histories = {k: m["history"] for k, m in metrics.items() if m.get("history")}
        if histories:
            plots.training_curves(histories, spec.name, figs / "training_curves.png")


def cmd_seeds(args):
    for spec in _datasets(args.dataset):
        split = prepare(spec)
        rows = []
        for model in _models(args.model):
            if model == "gb":
                continue
            params, _ = load_params(spec.name, model)
            print(f"\n=== {spec.name} / {model}: {args.n_seeds} seeds ===")
            for r in seed_runs(split, model, params, args.n_seeds, args.max_epochs):
                print(f"  seed {r['seed']}  test RMSE {r['rmse']:6.2f}")
                rows.append(r)
        if not rows:
            continue
        results = output_dir(spec.name, "results")
        with open(results / "seeds.json", "w") as f:
            json.dump(rows, f, indent=2)
        df = pd.DataFrame(rows).groupby("model", sort=False)
        lines = ["| Model | Test RMSE (mean ± std) | Min - max | MAE | PHM08 |",
                 "|---|---|---|---|---|"]
        for model, g in df:
            lines.append(f"| {plots.LABELS[model]} | {g.rmse.mean():.2f} ± {g.rmse.std():.2f} | "
                         f"{g.rmse.min():.2f} - {g.rmse.max():.2f} | {g.mae.mean():.2f} | "
                         f"{g.phm08.mean():.1f} |")
        table = "\n".join(lines)
        print(f"\n=== {spec.name}: {len(g)} seeds per model ===\n{table}")
        (results / "seeds.md").write_text(table + "\n")


def cmd_explain(args):
    for spec in _datasets(args.dataset):
        models = output_dir(spec.name, "models")
        figs = output_dir(spec.name, "figures")
        split = None

        if (models / "gb.joblib").exists():
            import shap

            gb = RULModel.load(models / "gb.joblib").estimator
            split = split or prepare(spec)
            X = rolling_features(split.train, gb.features, gb.window)
            X = X.sample(min(args.n_samples, len(X)), random_state=0)
            values = shap.TreeExplainer(gb.model).shap_values(X)
            plots.shap_summary(values, X, figs / "shap_summary.png")
            plots.shap_summary(values, X, figs / "shap_importance.png", bar=True)

        if (models / "transformer.joblib").exists():
            import torch

            from rul.features import Windows

            tr = RULModel.load(models / "transformer.joblib").estimator
            split = split or prepare(spec)
            idx = np.flatnonzero(last_rows(split.test))[: args.n_units]
            X = torch.from_numpy(Windows(split.test, tr.features, tr.seq_len)[idx])
            with torch.no_grad():
                pred, attn = tr.net.eval()(X, return_attention=True)
            plots.attention(attn.numpy(), pred.numpy(), split.test_rul[: args.n_units],
                            spec.name, figs / "attention.png")


def cmd_predict(args):
    model = RULModel.load(args.model)
    out = model.predict(read_raw(args.input)).round(1)
    print(out.to_string(index=False, max_rows=20))
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


def cmd_run(args):
    cmd_explore(args)
    cmd_tune(args)
    cmd_train(args)
    cmd_compare(args)
    cmd_explain(args)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rul", description="Remaining useful life models")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, datasets=True, models=False):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if datasets:
            p.add_argument("--dataset", nargs="+", default=["all"], help=f"{list(DATASETS)} or all")
        if models:
            p.add_argument("--model", nargs="+", default=["all"], choices=MODELS + ["all"])
        return p

    def n_trials(p):
        p.add_argument("--n-trials", type=int, default=TUNING["n_trials"])

    def max_epochs(p):
        p.add_argument("--max-epochs", type=int, default=TUNING["max_epochs"])

    def verbose(p):
        p.add_argument("--verbose", action="store_true", help="print every epoch")

    def explain_options(p):
        p.add_argument("--n-samples", type=int, default=500, help="rows sampled for SHAP")
        p.add_argument("--n-units", type=int, default=4, help="test engines in the attention plot")

    add("explore", cmd_explore)
    p = add("tune", cmd_tune, models=True)
    n_trials(p)
    max_epochs(p)
    p = add("train", cmd_train, models=True)
    max_epochs(p)
    verbose(p)
    add("compare", cmd_compare)
    explain_options(add("explain", cmd_explain))
    p = add("seeds", cmd_seeds, models=True)
    p.add_argument("--n-seeds", type=int, default=5)
    max_epochs(p)
    p = add("run", cmd_run, models=True)
    n_trials(p)
    max_epochs(p)
    verbose(p)
    explain_options(p)
    p = add("predict", cmd_predict, datasets=False)
    p.add_argument("--model", required=True, help="a saved outputs/<dataset>/models/*.joblib")
    p.add_argument("--input", required=True, help="a C-MAPSS train or test file")
    p.add_argument("-o", "--output")

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
