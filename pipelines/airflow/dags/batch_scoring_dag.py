"""
Airflow DAG: Daily batch scoring pipeline.

Schedule: daily at 04:00 UTC
Runs: feature refresh → score customers → validate scores → export output

This DAG assumes the champion model is already in the MLflow Production stage.
It only triggers retraining if drift or performance degradation is detected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

_DEFAULT_ARGS = {
    "owner":            "data-science",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": True,
    "email":            ["mlops@xyz.in"],
}

with DAG(
    dag_id="xyz_churn_batch_scoring",
    default_args=_DEFAULT_ARGS,
    description="Daily batch survival scoring for all active customers",
    schedule_interval="0 4 * * *",
    catchup=False,
    tags=["churn", "survival", "scoring", "batch"],
    max_active_runs=1,
) as dag:

    # ── 1. Refresh features ───────────────────────────────────────────────────
    def refresh_features(**context):
        import subprocess
        score_date = context["ds"]
        result = subprocess.run(
            ["python", "pipelines/scripts/run_feature_engineering.py",
             "--config", "config/config.yaml",
             "--mode", "score",
             "--as-of-date", score_date],
            capture_output=True, text=True, check=True,
        )
        return result.stdout

    refresh_features_task = PythonOperator(
        task_id="refresh_features",
        python_callable=refresh_features,
        provide_context=True,
    )

    # ── 2. Run batch scoring ──────────────────────────────────────────────────
    def run_scoring(**context):
        import subprocess
        score_date = context["ds"]
        result = subprocess.run(
            ["python", "pipelines/scripts/run_scoring.py",
             "--config", "config/config.yaml",
             "--score-date", score_date],
            capture_output=True, text=True, check=True,
        )
        return result.stdout

    batch_score = PythonOperator(
        task_id="batch_score",
        python_callable=run_scoring,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    # ── 3. Validate score output ──────────────────────────────────────────────
    def validate_scores(**context):
        import subprocess
        score_date = context["ds"]
        result = subprocess.run(
            ["python", "pipelines/scripts/run_scoring.py",
             "--config", "config/config.yaml",
             "--validate-only",
             "--score-date", score_date],
            capture_output=True, text=True, check=True,
        )
        import re
        match = re.search(r'"drift_detected":\s*(true|false)', result.stdout)
        drift = match and match.group(1) == "true"
        context["ti"].xcom_push(key="drift_detected", value=drift)
        return result.stdout

    validate_scores_task = PythonOperator(
        task_id="validate_scores",
        python_callable=validate_scores,
        provide_context=True,
    )

    # ── 4. Branch: drift detected? ────────────────────────────────────────────
    def check_drift(**context):
        drift = context["ti"].xcom_pull(task_ids="validate_scores", key="drift_detected")
        if drift:
            return "trigger_retraining"
        return "export_scores"

    branch_drift = BranchPythonOperator(
        task_id="branch_drift_check",
        python_callable=check_drift,
        provide_context=True,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="xyz_churn_training",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # ── 5. Export scores to downstream systems ────────────────────────────────
    def export_scores(**context):
        score_date = context["ds"]
        # In production: copy to S3 / GCS / ADLS / downstream DB
        import subprocess
        subprocess.run(
            ["python", "pipelines/scripts/run_scoring.py",
             "--config", "config/config.yaml",
             "--export",
             "--score-date", score_date],
            capture_output=True, text=True, check=True,
        )

    export_scores_task = PythonOperator(
        task_id="export_scores",
        python_callable=export_scores,
        provide_context=True,
    )

    # ── DAG wiring ────────────────────────────────────────────────────────────
    (
        refresh_features_task
        >> batch_score
        >> validate_scores_task
        >> branch_drift
        >> [trigger_retraining, export_scores_task]
    )
