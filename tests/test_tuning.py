import json

import pytest

from rul.config import MODELS
from rul.pipeline import prepare
from rul.tuning import DEFAULTS, load_params, suggest, tune


@pytest.mark.parametrize("model", MODELS)
def test_search_space_matches_defaults(model):
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial(DEFAULTS[model])
    assert suggest(trial, model) == DEFAULTS[model]


def test_tune_completes_requested_trials_and_saves(spec, tmp_path):
    pytest.importorskip("optuna")
    split = prepare(spec)
    best = tune(split, "gb", spec.name, n_trials=3)
    saved = json.loads((tmp_path / "outputs" / spec.name / "params" / "gb.json").read_text())
    assert saved["n_trials"] == 3
    assert saved["params"] == best
    assert load_params(spec.name, "gb") == (best, True)


def test_defaults_used_when_untuned(spec):
    assert load_params(spec.name, "lstm") == (DEFAULTS["lstm"], False)
