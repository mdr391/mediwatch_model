"""
dags/acceptance_test_dag.py
-----------------------------
Acceptance test DAG — proves the Airflow stack is healthy:

  1. env_check      : Python interpreter, PYTHONPATH, key env vars
  2. python_check   : core Python packages importable (json, subprocess, etc.)
  3. uv_check       : uv is available in container PATH
  4. mlflow_check   : mlflow package importable and tracking URI reachable

Trigger manually:
    airflow dags trigger mediwatch_acceptance_test
Or via the REST API (see scripts/trigger_windows.py).

This DAG is NOT scheduled — it is for validation only.
No external project code is imported — this is a standalone stack validation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow.operators.python import PythonOperator

from airflow import DAG

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="mediwatch_acceptance_test",
    description="Stack acceptance test — no schedule, trigger manually.",
    start_date=datetime(2024, 1, 1),
    schedule=None,           # manual trigger only
    catchup=False,
    tags=["mediwatch", "acceptance"],
) as dag:

    # ── Task 1: environment ─────────────────────────────────────────────────

    def check_env(**ctx):
        import platform
        print(f"Python     : {sys.version}")
        print(f"Platform   : {platform.platform()}")
        print(f"PYTHONPATH : {os.environ.get('PYTHONPATH', '(not set)')}")
        print(f"Working dir: {os.getcwd()}")

        required = ["AIRFLOW__CORE__EXECUTOR"]
        missing  = [v for v in required if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

        print("env_check PASSED")

    t_env = PythonOperator(
        task_id="env_check",
        python_callable=check_env,
    )

    # ── Task 2: python sanity check ───────────────────────────────────────

    def check_python(**ctx):
        import importlib
        for mod in ["json", "subprocess", "pathlib", "logging"]:
            importlib.import_module(mod)
            print(f"  {mod} OK")
        print("python_check PASSED")

    t_python = PythonOperator(
        task_id="python_check",
        python_callable=check_python,
    )

    # ── Task 3: uv available ────────────────────────────────────────────────

    def check_uv(**ctx):
        import subprocess
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "uv is not available in the container PATH.\n"
                "Add uv to the Airflow image or mount it as a binary.\n"
                f"stderr: {result.stderr}"
            )
        print(f"uv version : {result.stdout.strip()}")
        print("uv_check PASSED")

    t_uv = PythonOperator(
        task_id="uv_check",
        python_callable=check_uv,
    )

    # ── Task 4: MLflow importable ─────────────────────────────────────────

    def check_mlflow(**ctx):
        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI : {tracking_uri}")
        else:
            print(f"MLFLOW_TRACKING_URI not set — using default: {mlflow.get_tracking_uri()}")

        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        print(f"MLflow experiments visible: {len(experiments)}")
        print("mlflow_check PASSED")

    t_mlflow = PythonOperator(
        task_id="mlflow_check",
        python_callable=check_mlflow,
    )

    # ── Task ordering ───────────────────────────────────────────────────────
    #
    #  env_check → python_check → uv_check
    #                            → mlflow_check

    t_env >> t_python >> [t_uv, t_mlflow]
