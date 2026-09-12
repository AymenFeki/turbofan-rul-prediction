from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Sub-dataset -> number of operating conditions.
DATASETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

UNIT = "unit"
TIME = "cycle"
RUL = "rul"
RUL_CLIP = 125

SEED = 42
VAL_FRAC = 0.2

MODELS = ["gb", "lstm", "cnn", "transformer"]

# Shared by every model's search.
TUNING = {
    "n_trials": 30,
    "max_epochs": 60,
    "patience": 8,
}


def output_dir(dataset, kind):
    path = OUTPUT_ROOT / dataset / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
