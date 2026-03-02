"""Evaluate a pipeline on a given window's eval set and persist metrics."""

import json

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import EVALUATIONS_DIR


def evaluate_and_save(
    pipe: Pipeline,
    X_val,
    y_val,
    model_date: str,
    eval_window_date: str,
) -> dict:
    """Score the pipeline and write a JSON metrics file.

    File naming: eval_model_<trained_on>_on_<evaluated_on>.json
    """
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    metrics = {
        "model_date": model_date,
        "eval_window_date": eval_window_date,
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_val, y_proba), 4),
    }

    path = EVALUATIONS_DIR / f"eval_model_{model_date}_on_{eval_window_date}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
