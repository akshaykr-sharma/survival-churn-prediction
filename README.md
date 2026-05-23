# XYZ India — Survival-Based Churn Model

Production-grade customer churn prediction system for XYZ's India pre-built PC market.
Uses **survival analysis** (Kaplan-Meier, Cox Proportional Hazards, Weibull/Log-Normal/Log-Logistic)
to model *when* a customer will churn, not just *if*.

---

> **⚠️ Disclaimer — Dummy Data & Learning Purpose**
>
> All data used in this project is **synthetically generated** for learning and testing purposes only.
> It does not represent any real customer, transaction, or business record.
> If you integrate this pipeline with **real-world data**, changes will likely be required — including
> schema adjustments, churn definition recalibration, feature relevance review, and model retuning
> — depending on the structure and characteristics of your actual dataset.

---

## Why Survival Analysis?

Traditional binary classifiers (logistic regression, XGBoost) answer:
> "Will this customer churn?" → Yes / No

Survival models answer:
> "What is the probability that this customer survives (stays active) beyond **t** days?"

This gives the business:
- **Time-to-churn estimates** — when to intervene, not just who to target
- **Correct handling of censored customers** — active customers have incomplete observation
- **Actionable risk curves** — S(t) at 30, 60, 90, 180-day horizons

---

## Architecture

```
                     Raw Data Layer
    ┌────────────────────────────────────────────────┐
    │ customers/ transactions/ clickstream/ support/ │  ← Parquet (50K+ cust, 1M+ events)
    └─────────────────────┬──────────────────────────┘
                          │  DataLoader (PySpark)
                    Data Validation (Pandera)
                          │
                          ▼
    ┌────────────────────────────────────────────────┐
    │        Feature Engineering (PySpark)           │
    │  RFM · Behavioural · Trend · Support · Derived │
    └─────────────────────┬──────────────────────────┘
                          │  Feature Matrix (Parquet)
                          ▼
    ┌────────────────────────────────────────────────┐
    │           Model Training (lifelines)           │
    │  Kaplan-Meier │ Cox PH │ Weibull │ Log-Normal  │
    └──────────┬─────────────────────────────────────┘
               │  MLflow experiment tracking
               │  Model Registry (Production stage)
               ▼
    ┌────────────────────────────────────────────────┐
    │         Batch Scoring (PySpark)                │
    │   S(30d)  S(60d)  S(90d)  S(180d)  Risk Seg   │
    └──────────┬─────────────────────────────────────┘
               │  Scores Parquet
               ▼
    ┌────────────────────────────────────────────────┐
    │         Monitoring                             │
    │   PSI drift · C-index degradation · Brier      │
    └────────────────────────────────────────────────┘

Orchestration: Apache Airflow (3 DAGs)
CI/CD: GitHub Actions (lint → test → model-gate → deploy)
```

---

## Dataset

> **Note:** All data is synthetically generated via `src/data/data_generator.py` and is used
> purely for learning and testing. No real customer data is included in this repository.
> See [docs/data_dictionary.md](docs/data_dictionary.md) for full column definitions.

| Table | Rows (approx.) | Key columns |
|---|---|---|
| `customers` | 55,000 | customer_id, segment, city_tier, product_category, duration_days, is_churned |
| `transactions` | ~230,000 | customer_id, transaction_date, transaction_type, amount_inr |
| `clickstream` | ~660,000 | customer_id, event_timestamp, event_type, session_duration_s |
| `support_tickets` | ~99,000 | customer_id, ticket_date, category, is_escalated, resolution_days |

**Churn definition** (combined behavioural + transactional):
- No transaction for ≥ 270 days **AND**
- Trailing-90d avg weekly web sessions ≤ 0.5 **AND**
- No support contact for ≥ 180 days

---

## Models

| Model | Type | Covariates | Use case |
|---|---|---|---|
| Kaplan-Meier | Non-parametric | None | Population & segment-level curves, log-rank tests |
| Cox PH | Semi-parametric | Yes (static + TVC) | Individual predictions, hazard ratios, feature importance |
| Weibull AFT | Parametric | Yes | Smooth curves, extrapolation beyond observation window |
| Log-Normal AFT | Parametric | Yes | Mid-lifecycle churn patterns |
| Log-Logistic AFT | Parametric | Yes | Heavy-tailed survival, comparison with Weibull |

Champion model selection: highest C-index on held-out test set among covariate-adjusted models.
KM is kept as a non-adjustable baseline.

---

## Prerequisites

| Tool | Version | How to check | Install if missing |
|---|---|---|---|
| Python | 3.10 or 3.11 | `python --version` | https://www.python.org/downloads/ |
| Java JDK | 17 (LTS) | `java -version` | `winget install EclipseAdoptium.Temurin.17.JDK` (Windows) |
| Git | any recent | `git --version` | https://git-scm.com/download/win |

> **Why Java?** PySpark runs on the JVM. Without Java 11/17, PySpark will not start.

---

## Quick Start

### 1 — Create & activate virtual environment

```powershell
# Windows (PowerShell)
cd D:\self_projects\xyz-churn-survival

python -m venv churn_pred
churn_pred\Scripts\activate
```

```bash
# macOS / Linux
cd xyz-churn-survival
python3 -m venv churn_pred
source churn_pred/bin/activate
```

Your prompt will show `(churn_pred)` when the environment is active.

### 2 — Install all packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt       # production deps
pip install -r requirements-dev.txt   # adds pytest, black, etc.
```

### 3 — Set JAVA_HOME for PySpark

**Windows — set once, persists across sessions:**

```powershell
# Find your JDK folder first
Get-ChildItem "C:\Program Files\Eclipse Adoptium\"

# Then set (replace the folder name with what you see above)
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
$env:PATH      = "$env:JAVA_HOME\bin;" + $env:PATH

# Make it permanent for your user account (no admin needed)
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", $env:JAVA_HOME, "User")
$up = [System.Environment]::GetEnvironmentVariable("PATH","User")
[System.Environment]::SetEnvironmentVariable("PATH", $up + ";$env:JAVA_HOME\bin", "User")

# Verify
java -version
```

**macOS / Linux:**
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # macOS
# or
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64  # Ubuntu
export PATH=$JAVA_HOME/bin:$PATH
```

> **Tip (Windows):** `src/utils/spark_utils.py` auto-detects `JAVA_HOME` by reading
> the Windows registry and scanning common JDK install paths, so you usually don't
> need to set it every session once it's been installed.

### 3b — Install Hadoop winutils (Windows only)

PySpark on Windows requires `winutils.exe` + `hadoop.dll` (Hadoop filesystem shim).
Without these, every Spark job fails on Windows with `The system cannot find the path specified`.

```powershell
New-Item -ItemType Directory -Force -Path "C:\hadoop\bin"
Invoke-WebRequest "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe" `
    -OutFile "C:\hadoop\bin\winutils.exe"
Invoke-WebRequest "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll" `
    -OutFile "C:\hadoop\bin\hadoop.dll"
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
$env:HADOOP_HOME = "C:\hadoop"
```

`spark_utils.py` automatically wires `C:\hadoop\bin` into `PATH` so the Windows
DLL loader finds `hadoop.dll`. Not required on macOS/Linux.

### 4 — Run the pipeline

```powershell
# Generate synthetic dataset — 55K customers, 4M+ events (~4 min)
python pipelines/scripts/run_data_generation.py --config config/config.yaml

# Feature engineering via PySpark (~3-5 min, first run is slower)
python pipelines/scripts/run_feature_engineering.py --config config/config.yaml --mode train

# Train KM + Cox PH + Weibull/Log-Normal/Log-Logistic, log to MLflow
python pipelines/scripts/run_training.py --config config/config.yaml --run-date 2024-06-30

# Register champion model in MLflow Model Registry
python pipelines/scripts/run_training.py --config config/config.yaml --run-date 2024-06-30 --register

# Run batch scoring for all active customers
python pipelines/scripts/run_scoring.py --config config/config.yaml --score-date 2024-12-31
```

### 5 — View results in MLflow UI

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open **http://localhost:5000** in your browser. Press `Ctrl+C` to stop.

### Run unit tests

```powershell
python -m pytest tests/unit -m unit -v
# Expected: 32 passed
```

> For detailed Windows troubleshooting see [`LOCAL_SETUP.md`](LOCAL_SETUP.md).

---

## Project Structure

```
xyz-churn-survival/
├── config/                 # Master config (config.yaml, feature_config.yaml)
├── data/
│   ├── raw/                # Generated Parquet datasets (gitignored)
│   └── processed/          # Feature matrix + scores (gitignored)
├── docs/
│   ├── model_explainers/   # KM, Cox, Parametric deep dives
│   ├── data_dictionary.md
│   └── mlops_guide.md
├── pipelines/
│   ├── airflow/dags/       # Training, scoring, validation DAGs
│   └── scripts/            # CLI entry points for each pipeline stage
├── src/
│   ├── data/               # Generator, validator, loader
│   ├── features/           # PySpark feature engineering
│   ├── models/             # KM, Cox PH, Parametric, selector
│   ├── scoring/            # Batch scorer (PySpark)
│   ├── monitoring/         # Drift detection, performance monitor
│   └── utils/              # Spark, MLflow, logging utilities
├── tests/
│   ├── unit/               # Fast tests (no Spark / MLflow)
│   └── integration/        # Slower end-to-end tests
└── .github/workflows/      # CI, model validation gate, batch scoring trigger
```

---

## MLOps Workflow

### Training (monthly)
```
Airflow DAG: xyz_churn_training  (cron: 0 2 1 * *)
  validate_data → engineer_features → train_models
    → evaluate_models → [quality_gate]
      → register_champion (if C-index ≥ 0.68)
```

### Scoring (daily)
```
Airflow DAG: xyz_churn_batch_scoring  (cron: 0 4 * * *)
  refresh_features → batch_score → validate_scores
    → [drift_check]
      → export_scores  OR  trigger_retraining
```

### PR Gates (GitHub Actions)
| Workflow | Trigger | Gate |
|---|---|---|
| `ci.yml` | Every PR | Lint + type-check + unit tests (coverage ≥ 80%) |
| `model-validation.yml` | PR touching `src/models/` or `src/features/` | C-index ≥ 0.68 on 10K sample |
| `batch-scoring.yml` | Daily 04:00 UTC + manual | Score output validation + drift check |

---

## Key Metrics

| Metric | Definition | Target |
|---|---|---|
| C-index (Harrell's C) | Concordance of predicted risk vs actual event ordering | ≥ 0.70 |
| Brier Score at 90d | Calibration + discrimination at 90-day horizon | ≤ 0.20 |
| PSI (feature drift) | Population Stability Index per feature | < 0.20 |
| C-index degradation | Drop in C-index vs baseline run | < 0.02 |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Big data | PySpark 3.5 |
| Survival models | lifelines 0.29 (KM, Cox PH, Weibull, Log-Normal, Log-Logistic) |
| Experiment tracking | MLflow 2.13 |
| Orchestration | Apache Airflow 2.9 |
| Data validation | Pandera 0.19 |
| CI/CD | GitHub Actions |
| Serialisation | Parquet (PyArrow, Snappy) |
| Logging | structlog |

---

## Platform Compatibility Notes

These are non-obvious behaviours the code already handles so you don't have to.

**Windows / PySpark bootstrapping** — `src/utils/spark_utils.py` runs a one-time
setup at import:
- Pins `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` to `sys.executable` (PySpark's
  default of `python3` doesn't exist on Windows).
- Reads `JAVA_HOME` and `HADOOP_HOME` from the user-scope Windows registry
  (HKCU\Environment) and falls back to scanning common JDK install paths.
- Prepends `%JAVA_HOME%\bin` and `%HADOOP_HOME%\bin` to `PATH` so the Windows
  DLL loader can find `java.exe` and `hadoop.dll`.

**Parquet type discipline** — `src/data/data_generator.py` writes columns with
explicit PyArrow types so the strict Spark schema enforcement in
`src/data/data_loader.py` never trips on physical-type mismatches:
- Date columns → `pa.date32()` (INT32 + DATE logical type)
- Timestamp columns → `pa.timestamp('us')` (INT64 + TIMESTAMP_MICROS, microsecond
  precision — Spark cannot read nanosecond Parquet timestamps)
- Integer columns → downcast `int64` → `int32` when values fit, matching the
  DataLoader's `IntegerType` schema width

Spark's vectorized Parquet reader is also disabled (`enableVectorizedReader=false`)
because the row-based reader is more tolerant of legacy files. This costs little
on a 55K-row dataset.

**Cox PH convergence safety** — `src/models/cox_ph.py` `_prepare_features` runs
defensive preprocessing before handing data to `lifelines.CoxPHFitter`:
- Replaces `±inf` with `0` (derived-feature divisions can produce these)
- Drops near-zero-variance columns (`std < 1e-6`) — these cause Newton-Raphson
  to produce NaN deltas
- Z-scores all numeric features so the penalizer applies uniformly and the
  optimizer doesn't overshoot on large-scale features like `product_price_inr`

Means and standard deviations are captured at fit time and reapplied at inference.

**Schoenfeld residuals on training data** — `cox.test_proportional_hazard_assumption()`
takes no arguments. The Cox model caches its prepared training matrix at fit time
and uses it automatically; passing a different dataset is statistically incorrect
(residuals are defined relative to the fit) and triggers a lifelines length-mismatch
error.
