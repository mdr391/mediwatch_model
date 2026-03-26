# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Champion/Challenger ML retraining pipeline for the UCI 130-Hospital Diabetes readmission dataset. The system detects concept drift across five time windows (2004–2008), trains a new challenger model when drift is detected, promotes it if F1 improves by ≥1% over the champion.

## Common Commands

```bash
# Install dependencies
uv sync

# Prepare the five frozen parquet windows (one-time)
uv run generate_windows.py

# Run the full pipeline locally (sequential, single process)
uv run runner.py

# Reset MLflow experiment state
uv run runner_cleanup.py

# View MLflow results
mlflow ui   # http://localhost:5000
```

**Verification scripts:**
```bash
uv run scripts/verify_mlflow_registry.py   # Validate MLflow on synthetic data
uv run scripts/verify_mlflow_cleanup.py    # Verify experiment cleanup
```

**Airflow (containerized):**
```bash
cd airflow

# Acceptance test only (no project code needed)
docker compose up -d --build
./scripts/run_acceptance_test.sh

# Full pipeline (mounts parent project into containers)
docker compose -f docker-compose.yml -f docker-compose.pipeline.yml up -d --build
python scripts/trigger_windows.py --dag-id mediwatch_pipeline --poll-secs 30 --timeout 600

# Airflow UI: http://localhost:8080  (admin / admin)
docker compose down -v   # tear down + remove DB volume
```

## Architecture

### Two Execution Paths

1. **`runner.py` (local):** Processes all 5 windows sequentially in a single Python process. Champion state is held in memory.
2. **Airflow DAG (`airflow/dags/pipeline_dag.py`):** One isolated DAG run per window date. Champion state persists across runs via the MLflow model registry `@champion` alias.

### Pipeline Per Window (6 tasks in Airflow / equivalent steps in runner.py)

```
detect_window → drift_report → train_challenger → evaluate_models → promote_decision → log_summary
```

- **Cold start (window 1):** Train and deploy as champion unconditionally.
- **Challenge (windows 2–5):** Detect drift (Evidently), train challenger, compare F1 on eval set, promote if `challenger_f1 ≥ champion_f1 + 0.01`.

### Source Modules (`src/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | All constants: paths, feature lists, window dates, MLflow names, promotion threshold |
| `data.py` | Load frozen parquet windows; `load_sliding_train()` concatenates current + previous window |
| `preprocessing.py` | Custom sklearn Transformers (ICD-9 binning, missing values, categorical encoding); `build_preprocessor()` |
| `training.py` | Build, train, and serialize the full sklearn Pipeline (preprocessor + XGBoost) via joblib |
| `evaluation.py` | Compute metrics (accuracy, F1, precision, recall, ROC-AUC); persist JSON artifacts |
| `drift.py` | Evidently `DataDriftTable` comparing eval sets; returns `(drift_detected: bool, html_report)` |
| `mlflow_utils.py` | Helpers to restore/rename/delete soft-deleted MLflow experiments |

### Key Design Decisions

- **Single sklearn Pipeline artifact:** Preprocessor + model serialized together as one `.joblib` file — eliminates train/serving skew.
- **Sliding window training:** Training data = current window + previous window (not just current). Prevents catastrophic forgetting; doubles training size with minimal compute.
- **Ordinal encoding for trees:** Avoids one-hot explosion; handles unknown categories (→ -1) and missing (→ -2).
- **ICD-9 binning:** Three raw diagnosis columns collapsed into 8 binary clinical-category flags.
- **F1 as promotion metric:** Positive class is ~12%; accuracy is misleading.
- **`@champion` alias in MLflow registry:** The only cross-run state in the Airflow path. `detect_window` reads it; `promote_decision` updates it.
- **XCom for intra-run state:** `run_id` and evaluation metrics passed between Airflow tasks via XCom.

### MLflow Tracking

- **Experiment:** `mediwatch_champion_challenger`
- **Registered model:** defined in `src/config.py` as `REGISTERED_MODEL`
- **Logged per run:** champion/challenger metrics, `drift_detected`, `deployed`, `f1_delta`, drift HTML report, JSON summary
- In pipeline mode, MLflow data writes to host via `sqlite:////opt/mediwatch/mlflow.db`

### Airflow Stack Details

- Custom image: `apache/airflow:2.9.3-python3.11` + `uv`; `requires-python = ">=3.11"` (not pinned to 3.13 because containers use 3.11)
- Two compose modes: standalone (`docker-compose.yml`) for acceptance test; pipeline mode adds `docker-compose.pipeline.yml` override which mounts `..:/opt/mediwatch` and sets `PYTHONPATH`
- Acceptance DAG (`mediwatch_acceptance_test`): 4 tasks, no project imports — validates env/Python/uv/MLflow
- Trigger script (`airflow/scripts/trigger_windows.py`): REST API client against Airflow v1 API, basic auth admin/admin

### Data

- **Source:** `data/diabetic_data.csv` (UCI 130-Hospital dataset)
- **Windows:** `windows/window_YYYY.parquet` — five frozen year splits, 80/20 train/eval each
- **Features:** 8 numeric + 22 categorical + 12 ICD-9 derived binary flags (see `src/config.py` for exact lists)
