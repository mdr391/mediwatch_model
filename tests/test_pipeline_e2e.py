"""End-to-end pipeline test on synthetic data.

Runs a minimal 2-window champion/challenger cycle:
  Window 1: cold start — train and install champion.
  Window 2: challenge — train challenger, compare, decide.

Uses tmp_path for all artifacts and a temporary MLflow tracking URI
so nothing leaks into the real MLflow store.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_COLS,
    HIGH_CARDINALITY_COLS,
    NUMERIC_COLS,
    RANDOM_SEED,
)


def _make_synthetic_window(n: int, readmit_rate: float, seed: int) -> pd.DataFrame:
    """Generate a synthetic DataFrame resembling one window of the dataset."""
    rng = np.random.default_rng(seed)
    data = {}

    # ID columns
    data["encounter_id"] = list(range(n))
    data["patient_nbr"] = list(range(1000, 1000 + n))

    # Numeric features
    for col in NUMERIC_COLS:
        data[col] = rng.integers(0, 20, size=n).astype(float)

    # Categorical features (non-ICD9)
    cat_values = {
        "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
        "gender": ["Female", "Male"],
        "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        "admission_type_id": ["1", "2", "3"],
        "discharge_disposition_id": ["1", "2", "3"],
        "admission_source_id": ["1", "7", "4"],
        "payer_code": ["MC", "MD", "HM", "SP"],
        "medical_specialty": ["InternalMedicine", "Cardiology", "Surgery", "Emergency"],
        "max_glu_serum": ["None", ">200", ">300", "Norm"],
        "A1Cresult": ["None", ">7", ">8", "Norm"],
        "change": ["Ch", "No"],
        "diabetesMed": ["Yes", "No"],
    }
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    for col in med_cols:
        cat_values[col] = ["No", "Steady", "Up", "Down"]

    for col in CATEGORICAL_COLS:
        choices = cat_values.get(col, ["A", "B", "C"])
        data[col] = rng.choice(choices, size=n).tolist()

    # ICD-9 diagnosis codes
    icd9_samples = ["250.0", "410", "460", "520", "V01", "E800", "100", "785"]
    for col in HIGH_CARDINALITY_COLS:
        data[col] = rng.choice(icd9_samples, size=n).tolist()

    # Target
    n_positive = int(n * readmit_rate)
    readmitted = ["<30"] * n_positive + [">30"] * (n - n_positive)
    rng.shuffle(readmitted)
    data["readmitted"] = readmitted

    return pd.DataFrame(data)


@pytest.fixture()
def pipeline_env(tmp_path):
    """Set up isolated directories and MLflow tracking for the e2e test."""
    windows_dir = tmp_path / "windows"
    artifacts_dir = tmp_path / "artifacts"
    pipelines_dir = artifacts_dir / "pipelines"
    evaluations_dir = artifacts_dir / "evaluations"
    reports_dir = artifacts_dir / "reports"

    for d in [windows_dir, pipelines_dir, evaluations_dir, reports_dir]:
        d.mkdir(parents=True)

    # Generate two synthetic windows
    dates = ["2004-12-31", "2005-12-31"]
    for i, date in enumerate(dates):
        df = _make_synthetic_window(n=200, readmit_rate=0.12 + i * 0.05, seed=RANDOM_SEED + i)
        train = df.iloc[:160]
        eval_ = df.iloc[160:]
        train.to_parquet(windows_dir / f"{date}-train.parquet", index=False)
        eval_.to_parquet(windows_dir / f"{date}-eval.parquet", index=False)

    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    return {
        "tmp_path": tmp_path,
        "dates": dates,
        "windows_dir": windows_dir,
        "artifacts_dir": artifacts_dir,
        "pipelines_dir": pipelines_dir,
        "evaluations_dir": evaluations_dir,
        "reports_dir": reports_dir,
        "mlflow_uri": mlflow_uri,
    }


def test_two_window_pipeline_e2e(pipeline_env):
    """Run cold start + challenge on synthetic data and verify outputs."""
    env = pipeline_env

    # Patch all paths and MLflow URI before importing runner
    with (
        patch("src.config.WINDOWS_DIR", env["windows_dir"]),
        patch("src.config.ARTIFACTS_DIR", env["artifacts_dir"]),
        patch("src.config.PIPELINES_DIR", env["pipelines_dir"]),
        patch("src.config.EVALUATIONS_DIR", env["evaluations_dir"]),
        patch("src.config.REPORTS_DIR", env["reports_dir"]),
        patch("src.data.WINDOWS_DIR", env["windows_dir"]),
        patch("src.evaluation.EVALUATIONS_DIR", env["evaluations_dir"]),
        patch("src.drift.REPORTS_DIR", env["reports_dir"]),
        patch.dict(os.environ, {"MLFLOW_TRACKING_URI": env["mlflow_uri"]}),
    ):
        # Import after patching so module-level references pick up the patches
        from runner import ChampionChallengerPipeline

        pipeline = ChampionChallengerPipeline(
            window_dates=env["dates"],
            promotion_threshold=0.01,
        )
        pipeline.run()

    # ── Assertions ──────────────────────────────────────────────

    # 1. Pipeline ran both windows
    assert len(pipeline.history) == 2

    # 2. First window is always cold_start
    assert pipeline.history[0]["outcome"] == "cold_start"
    assert pipeline.history[0]["champion"] == "2004-12-31"

    # 3. Second window is either promoted or retained (both valid)
    assert pipeline.history[1]["outcome"] in ("promoted", "retained")

    # 4. Champion date is set
    assert pipeline.champion_date is not None

    # 5. Pipeline artifacts exist on disk
    assert (env["pipelines_dir"] / "pipeline_2004-12-31.joblib").exists()
    assert (env["pipelines_dir"] / "pipeline_2005-12-31.joblib").exists()

    # 6. Evaluation JSONs were written
    eval_files = list(env["evaluations_dir"].glob("eval_*.json"))
    assert len(eval_files) >= 2  # at least one per window

    # 7. Metrics are valid
    for ef in eval_files:
        with open(ef) as f:
            metrics = json.load(f)
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    # 8. Summary file written
    summary_path = env["artifacts_dir"] / "summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert len(summary) == 2
