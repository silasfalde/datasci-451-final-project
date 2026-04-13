import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
CONFIG = {
    "seed": 451,
    "folds": 3,
    "max_bayes_folds": 1,
    "outcomes": ["h1n1_vaccine", "seasonal_vaccine"],
    "id_col": "respondent_id",
    "group_col": "hhs_geo_region",
    "paths": {
        "raw_features": PROJECT_ROOT / "data" / "raw" / "training_set_features.csv",
        "raw_labels": PROJECT_ROOT / "data" / "raw" / "training_set_labels.csv",
        "processed_data": PROJECT_ROOT / "data" / "processed" / "training_joined.pkl",
        "predictions": PROJECT_ROOT / "outputs" / "predictions" / "cv_predictions.csv",
        "metrics": PROJECT_ROOT / "outputs" / "metrics" / "model_metrics.csv",
        "roc_plot": PROJECT_ROOT / "outputs" / "figures" / "roc_curves.png",
        "calibration_plot": PROJECT_ROOT / "outputs" / "figures" / "calibration_curves.png",
        "figures": PROJECT_ROOT / "outputs" / "figures",
        "model_dir": PROJECT_ROOT / "outputs" / "models",
    },
    "bayes": {
        "draws": 700,
        "tune": 350,
        "chains": 2,
        "prior_sd": 1.0,
    },
}


def ensure_output_dirs():
    """Create all necessary output directories."""
    for key, path in CONFIG["paths"].items():
        if key == "model_dir":
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
