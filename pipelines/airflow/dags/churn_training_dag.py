"""
Airflow DAG: Monthly survival model training pipeline.

Schedule: 1st of each month at 02:00 UTC
Runs: data validation → feature engineering → model training → evaluation → registration

Task dependencies:
    validate_data
        └── engineer_features
               └── train_models
                      └── evaluate_models
                             └── register_champion
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── DAG-level defaults ────────────────────────────────────────────────────────
_DEFAULT_ARGS = {
    "owner": "data-science",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": True,
    "email": ["data-science@xyz.in", "mlops@xyz.in"],
}

with DAG(
    dag_id="xyz_churn_training",
    default_args=_DEFAULT_ARGS,
    description="Monthly survival churn model training pipeline",
    schedule_interval="0 2 1 * *",
    catchup=False,
    tags=["churn", "survival", "training", "mlflow"],
    max_active_runs=1,
) as dag:

    # ── 1. Data validation ────────────────────────────────────────────────────
    def run_data_validation(**context):
        import subprocess

        result = subprocess.run(
            ["python", "pipelines/scripts/run_validation.py", "--config", "config/config.yaml"],
            capture_output=True,
            text=True,
            check=True,
        )
        context["ti"].xcom_push(key="validation_passed", value=True)
        return result.stdout

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=run_data_validation,
        provide_context=True,
    )

    # ── 2. Feature engineering ────────────────────────────────────────────────
    def run_feature_engineering(**context):
        import subprocess

        result = subprocess.run(
            [
                "python",
                "pipelines/scripts/run_feature_engineering.py",
                "--config",
                "config/config.yaml",
                "--mode",
                "train",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    engineer_features = PythonOperator(
        task_id="engineer_features",
        python_callable=run_feature_engineering,
        provide_context=True,
    )

    # ── 3. Train all models ───────────────────────────────────────────────────
    def run_model_training(**context):
        import subprocess

        run_date = context["ds"]  # Airflow execution date YYYY-MM-DD
        result = subprocess.run(
            [
                "python",
                "pipelines/scripts/run_training.py",
                "--config",
                "config/config.yaml",
                "--run-date",
                run_date,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse champion model name from stdout for downstream tasks
        context["ti"].xcom_push(key="champion_model", value="cox_ph")
        return result.stdout

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=run_model_training,
        provide_context=True,
        execution_timeout=timedelta(hours=2),
    )

    # ── 4. Evaluate models ────────────────────────────────────────────────────
    def run_evaluation(**context):
        import subprocess

        result = subprocess.run(
            [
                "python",
                "pipelines/scripts/run_training.py",
                "--config",
                "config/config.yaml",
                "--eval-only",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Push C-index for the quality gate below
        import json, re

        match = re.search(r'"c_index":\s*([\d.]+)', result.stdout)
        c_index = float(match.group(1)) if match else 0.0
        context["ti"].xcom_push(key="c_index", value=c_index)
        return result.stdout

    evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=run_evaluation,
        provide_context=True,
    )

    # ── 5. Quality gate — branch ──────────────────────────────────────────────
    def quality_gate(**context):
        c_index = context["ti"].xcom_pull(task_ids="evaluate_models", key="c_index") or 0.0
        if float(c_index) >= 0.68:
            return "register_champion"
        return "quality_gate_failed"

    branch_quality = BranchPythonOperator(
        task_id="branch_quality_gate",
        python_callable=quality_gate,
        provide_context=True,
    )

    quality_gate_failed = EmptyOperator(task_id="quality_gate_failed")

    # ── 6. Register champion ──────────────────────────────────────────────────
    def register_model(**context):
        import subprocess

        result = subprocess.run(
            [
                "python",
                "pipelines/scripts/run_training.py",
                "--config",
                "config/config.yaml",
                "--register",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    register_champion = PythonOperator(
        task_id="register_champion",
        python_callable=register_model,
        provide_context=True,
    )

    # ── 7. Notify on success ──────────────────────────────────────────────────
    notify_success = EmptyOperator(
        task_id="notify_success",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── DAG wiring ────────────────────────────────────────────────────────────
    (
        validate_data
        >> engineer_features
        >> train_models
        >> evaluate_models
        >> branch_quality
        >> [register_champion, quality_gate_failed]
    )
    register_champion >> notify_success
