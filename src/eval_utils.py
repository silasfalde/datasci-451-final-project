import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, brier_score_loss, roc_auc_score
from config import CONFIG


def safe_auc(truth, estimate):
    """Compute AUC safely, returning NaN if class not present."""
    if len(np.unique(truth)) < 2:
        return np.nan

    return roc_auc_score(truth, estimate)


def brier_score(truth, estimate):
    """Compute Brier score."""
    return brier_score_loss(truth, estimate)


def calibration_table(df, bins=10):
    """Create calibration table from predictions."""
    df = df.copy()
    df["bin"] = pd.cut(
        df["prediction"], bins=np.linspace(0, 1, bins + 1), include_lowest=True
    )

    cal = (
        df.groupby(["model", "outcome", "bin"])
        .agg(
            mean_pred=("prediction", "mean"),
            mean_truth=("truth", "mean"),
            n=("prediction", "size"),
        )
        .reset_index()
    )

    return cal


def summarize_metrics(pred_df):
    """Compute per-outcome and macro metrics."""
    per_outcome = (
        pred_df.groupby(["model", "outcome"])
        .apply(
            lambda x: pd.Series(
                {
                    "auc": safe_auc(x["truth"], x["prediction"]),
                    "brier": brier_score(x["truth"], x["prediction"]),
                    "n": len(x),
                }
            )
        )
        .reset_index()
    )

    macro = (
        per_outcome.groupby("model")
        .agg(
            macro_auc=("auc", "mean"),
            macro_brier=("brier", "mean"),
        )
        .reset_index()
    )

    result = per_outcome.merge(macro, on="model")
    return result


def plot_roc_curves(pred_df, out_path):
    """Plot ROC curves by outcome and model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outcomes = pred_df["outcome"].unique()

    for idx, outcome in enumerate(sorted(outcomes)):
        ax = axes[idx]
        subset = pred_df[pred_df["outcome"] == outcome]

        for model in sorted(subset["model"].unique()):
            model_data = subset[subset["model"] == model]
            fpr, tpr, _ = roc_curve(model_data["truth"], model_data["prediction"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model} (AUC={roc_auc:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {outcome}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_calibration(pred_df, out_path):
    """Plot calibration curves by outcome and model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cal = calibration_table(pred_df, bins=10)
    outcomes = sorted(cal["outcome"].unique())

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        subset = cal[cal["outcome"] == outcome]

        for model in sorted(subset["model"].unique()):
            model_data = subset[subset["model"] == model].sort_values("mean_pred")
            ax.plot(
                model_data["mean_pred"],
                model_data["mean_truth"],
                marker="o",
                label=model,
                linewidth=2,
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Positive Rate")
        ax.set_title(f"Calibration: {outcome}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def write_outputs(pred_df, metric_df, cfg=None):
    """Write prediction and metric CSVs."""
    if cfg is None:
        cfg = CONFIG

    pred_df.to_csv(cfg["paths"]["predictions"], index=False)
    metric_df.to_csv(cfg["paths"]["metrics"], index=False)
