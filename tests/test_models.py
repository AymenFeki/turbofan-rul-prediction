import numpy as np
import pytest

from rul.config import MODELS, UNIT
from rul.datasets import read_raw
from rul.pipeline import RULModel, prepare, seed_runs, train_and_evaluate
from rul.tuning import DEFAULTS

TINY = {
    "gb": {"window": 3, "max_iter": 20},
    "lstm": {"seq_len": 8, "hidden": 8, "layers": 1},
    "cnn": {"seq_len": 8, "channels": 8, "layers": 1, "kernel": 3},
    "transformer": {"seq_len": 8, "d_model": 8, "heads": 2, "layers": 1},
}


def _needs(model):
    if model != "gb":
        pytest.importorskip("torch")


@pytest.mark.parametrize("model", MODELS)
def test_defaults_are_complete(model):
    assert set(TINY[model]) <= set(DEFAULTS[model])


@pytest.mark.parametrize("model", MODELS)
def test_train_evaluate_save_load(spec, model, tmp_path):
    _needs(model)
    split = prepare(spec)
    metrics = train_and_evaluate(split, model, DEFAULTS[model] | TINY[model], max_epochs=2)
    assert np.isfinite([metrics["rmse"], metrics["val_rmse"]]).all()

    path = tmp_path / "outputs" / spec.name / "models" / f"{model}.joblib"
    out = RULModel.load(path).predict(read_raw(spec.path("test")))
    assert len(out) == split.test[UNIT].nunique()
    assert (out["rul_pred"] >= 0).all()


@pytest.mark.parametrize("model", MODELS)
def test_one_prediction_per_row(multi_spec, model):
    _needs(model)
    from rul.estimators import make_estimator

    split = prepare(multi_spec)
    training = {} if model == "gb" else {"max_epochs": 1}
    est = make_estimator(model, DEFAULTS[model] | TINY[model], **training)
    est.fit(split.train, split.val, split.features)
    pred = est.predict(split.val)
    assert pred.shape == (len(split.val),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", ["lstm", "cnn", "transformer"])
def test_network_shapes(name):
    torch = pytest.importorskip("torch")
    from rul.networks import build_network

    net = build_network(name, 5, **{k: v for k, v in TINY[name].items() if k != "seq_len"})
    for seq_len in (1, 8, 33):
        assert net(torch.randn(4, seq_len, 5)).shape == (4,)


def test_transformer_attention_is_a_distribution():
    torch = pytest.importorskip("torch")
    from rul.networks import build_network

    net = build_network("transformer", 5, d_model=8, heads=2, layers=1).eval()
    _, attn = net(torch.randn(3, 10, 5), return_attention=True)
    assert attn.shape == (3, 10, 10)
    torch.testing.assert_close(attn.sum(-1), torch.ones(3, 10))


def test_transformer_rejects_bad_heads():
    pytest.importorskip("torch")
    from rul.networks import build_network

    with pytest.raises(ValueError):
        build_network("transformer", 5, d_model=10, heads=4)


def test_seed_runs_give_one_row_per_seed(spec):
    pytest.importorskip("torch")
    split = prepare(spec)
    rows = seed_runs(split, "lstm", DEFAULTS["lstm"] | TINY["lstm"], n_seeds=3, max_epochs=2)
    assert [r["seed"] for r in rows] == [42, 43, 44]
    assert all(np.isfinite(r["rmse"]) for r in rows)
