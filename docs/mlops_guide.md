# MLOps Guide

> **⚠️ Disclaimer — Dummy Data & Learning Purpose**
>
> This guide and the pipelines it describes operate on **synthetically generated dummy data**
> created for learning and testing purposes. The Airflow DAG schedules, MLflow tracking setup,
> and GitHub Actions workflows are production-pattern implementations, but all data flowing
> through them is simulated.
>
> **Before using this with real-world data**, review and adjust:
> - Data source connections in `pipelines/airflow/dags/` (currently run local scripts; point
>   to your actual data warehouse or data lake)
> - Airflow connections and secrets for your environment (database URIs, S3/ADLS credentials)
> - Row-count and schema thresholds in `src/data/data_validator.py` to match real data volumes
> - MLflow tracking URI and artifact store location for your infrastructure

---

## Overview

The XYZ churn survival system runs a fully automated ML lifecycle:

```
Dev machine / PR branch
    │
    │  GitHub Actions CI (lint + tests + model quality gate)
    │
    ▼
Airflow (scheduled pipelines)
    ├── Data Validation DAG (daily at 01:00 UTC)
    ├── Feature Engineering → Training DAG (monthly at 02:00 UTC)
    └── Batch Scoring DAG (daily at 04:00 UTC)
    │
    ▼
MLflow Model Registry
    └── [Staging] → [Production] (after quality gate passes)
    │
    ▼
Score output (Parquet) → CRM / BI tools / intervention campaigns
```

---

## MLflow Experiment Tracking

### Structure

```
Experiment: xyz-churn-survival
    └── Run: training_2024-10-01
        ├── Parameters: penalizer=0.1, train_rows=42000, ...
        ├── Metrics: cox_concordance_index=0.712, cv_c_index_mean=0.708, ...
        ├── Artifacts:
        │   ├── models/cox_ph.pkl
        │   ├── plots/km_curve.png
        │   ├── plots/km_segment.png
        │   ├── plots/cox_hazard_ratios.png
        │   └── reports/model_selection_report.csv
        └── Tags: run_status=SUCCESS, ph_violations=[...], run_date=2024-10-01
```

### Launching the UI

```bash
make mlflow-ui
# Opens http://localhost:5000
```

### Model Registry Stages

| Stage | Description |
|---|---|
| None | Model logged but not registered |
| Staging | Passed quality gate; awaiting production sign-off |
| Production | Current champion — used by batch scorer |
| Archived | Previous Production versions |

### Promoting a Model

```python
from src.utils.mlflow_utils import register_model

register_model(
    model_uri="runs:/<run_id>/cox_ph_model",
    registered_model_name="xyz_churn_cox_ph",
    stage="Production",
    description="Cox PH v3 — C-index=0.715, trained 2024-10-01",
)
```

---

## Airflow DAGs

### Prerequisites

```bash
export AIRFLOW_HOME=./airflow_home
airflow db init
airflow users create --username admin --password admin \
    --firstname Admin --lastname User --role Admin --email admin@xyz.in

# Copy DAGs
cp pipelines/airflow/dags/*.py $AIRFLOW_HOME/dags/

# Start
airflow webserver --port 8080 &
airflow scheduler &
```

### DAG: `xyz_churn_training`

- **Schedule:** 1st of each month at 02:00 UTC
- **Duration:** ~90 minutes (including 5-fold CV)
- **Quality gate:** Model is registered only if C-index ≥ 0.68
- **Failure notifications:** data-science@xyz.in, mlops@xyz.in

### DAG: `xyz_churn_batch_scoring`

- **Schedule:** Daily at 04:00 UTC
- **Duration:** ~30 minutes
- **Auto-retraining trigger:** If PSI > 0.20 detected in score distribution, triggers `xyz_churn_training` DAG
- **Output:** Parquet partitioned by score_date

### DAG: `xyz_churn_data_validation`

- **Schedule:** Daily at 01:00 UTC (before scoring)
- **Checks:** Row counts, null rates, schema, referential integrity

---

## GitHub Actions PR Workflow

### Workflow 1: `ci.yml` (every PR)

```
PR opened/updated
    ├── lint (black + isort + flake8)
    ├── type-check (mypy)
    ├── unit-tests (pytest -m unit, 2× Python versions)
    └── data-smoke-test (generate 5K rows, validate)
```
All four jobs must pass before merge is allowed.

### Workflow 2: `model-validation.yml` (PR touching models/features)

```
PR touches src/models/ or src/features/
    ├── Generate 10K sample dataset
    ├── Run feature engineering
    ├── Train Cox PH
    ├── Evaluate C-index
    └── Gate: C-index >= 0.68  (fails PR if not)
```

### Workflow 3: `batch-scoring.yml` (daily + manual)

```
04:00 UTC daily (or manual trigger)
    ├── Refresh features (scoring mode)
    ├── Run batch scoring
    ├── Validate score distribution
    ├── Drift check (PSI on score distribution)
    └── Export to downstream systems
```

---

## Adding a New Feature

1. Define the feature in `config/feature_config.yaml` under the appropriate category.
2. Implement the transformation in `src/features/feature_engineering.py`.
3. Add a unit test in `tests/unit/test_feature_engineering.py`.
4. Open a PR — the model-validation workflow will automatically check if the new feature improves C-index.
5. Update `docs/data_dictionary.md`.

---

## Retraining Triggers

| Trigger | Condition | Action |
|---|---|---|
| Scheduled | 1st of each month | Automatic retraining via Airflow |
| Drift alert | PSI > 0.20 on any key feature | Airflow `xyz_churn_batch_scoring` triggers `xyz_churn_training` |
| Performance degradation | C-index drops > 2pp vs baseline | Performance monitor raises alert → manual retraining review |
| Data volume spike | >20% change in daily row counts | Data validation DAG alerts team |

---

## Monitoring Dashboard

Key metrics logged to MLflow per scoring run:
- Score distribution (hist of survival_prob_90d)
- Risk segment proportions (High / Medium / Low %)
- PSI per feature vs training reference
- Realised C-index (computed monthly when 90d ground truth is available)

---

## Secret Management

Never commit credentials. Use environment variables or a secrets manager:

```bash
# Local development (.env file — gitignored)
MLFLOW_TRACKING_URI=http://mlflow.internal:5000
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# GitHub Actions: set in Repository Secrets
# Airflow: use Airflow Variables/Connections
```
