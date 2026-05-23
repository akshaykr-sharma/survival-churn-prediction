# Local Setup Guide — XYZ Churn Survival Model

Step-by-step instructions to run the full pipeline on a Windows machine.

---

## Prerequisites (install once)

| Tool | Version | Download |
|---|---|---|
| Python | 3.10 or 3.11 | https://www.python.org/downloads/ |
| Java JDK | 17 (LTS) | https://adoptium.net/temurin/releases/ |
| Git | latest | https://git-scm.com/download/win |

> **Why Java?** PySpark runs on the JVM. Without Java 11/17 on `PATH`, PySpark will fail.

---

## Step 1 — Verify prerequisites

Open **PowerShell** (or Command Prompt):

```powershell
# Check Python version (must be 3.10 or 3.11)
python --version

# Check Java version (must be 11 or 17)
java -version

# Check Git
git --version
```

If `java -version` says "not recognized", install JDK 17 from https://adoptium.net and
add `C:\Program Files\Eclipse Adoptium\jdk-17.x.x.x-hotspot\bin` to your `PATH`.

---

## Step 2 — Navigate to the project folder

```powershell
cd D:\self_projects\xyz-churn-survival
```

---

## Step 3 — Create virtual environment named `churn_pred`

```powershell
python -m venv churn_pred
```

This creates an isolated Python environment at `D:\self_projects\xyz-churn-survival\churn_pred\`.
All packages you install here will not affect your system Python.

---

## Step 4 — Activate the virtual environment

```powershell
churn_pred\Scripts\activate
```

Your prompt will change to `(churn_pred) PS D:\self_projects\xyz-churn-survival>`.

> **Every time you open a new terminal**, run this activate command before anything else.

---

## Step 5 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## Step 6 — Install all dependencies

```powershell
pip install -r requirements.txt
```

This installs: lifelines, PySpark, scikit-learn, pandas, numpy, scipy, MLflow, pandera,
pyarrow, structlog, matplotlib, seaborn, plotly, pyyaml, rich.

Expected time: 3–8 minutes depending on internet speed.

To also install dev/test tools:

```powershell
pip install -r requirements-dev.txt
```

---

## Step 7 — Set JAVA_HOME (required for PySpark)

```powershell
# Replace the path with your actual JDK installation path
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

# Verify PySpark can see Java
python -c "import pyspark; print('PySpark OK:', pyspark.__version__)"
```

To make `JAVA_HOME` permanent (so you don't have to set it each session):
1. Open **System Properties → Environment Variables**
2. Add `JAVA_HOME` = `C:\Program Files\Eclipse Adoptium\jdk-17.x.x.x-hotspot`
3. Edit `Path` → add `%JAVA_HOME%\bin`

> **Note:** `src/utils/spark_utils.py` automatically reads `JAVA_HOME` and `HADOOP_HOME`
> from the Windows registry (HKCU\Environment) and falls back to scanning common JDK
> install locations if the variable isn't set. You can usually run the pipeline without
> setting them manually each session.

---

## Step 7b — Install Hadoop winutils (Windows only, required for PySpark)

PySpark on Windows requires `winutils.exe` + `hadoop.dll` — small Hadoop shim binaries
that handle filesystem operations the JVM expects from a Unix-like environment.
Without them, every Spark job fails with cryptic errors like
`The system cannot find the path specified` or `Can not find winutils binary`.

```powershell
# Create Hadoop bin directory
New-Item -ItemType Directory -Force -Path "C:\hadoop\bin"

# Download winutils.exe + hadoop.dll for Hadoop 3.3.x (compatible with PySpark 3.4/3.5)
Invoke-WebRequest -Uri "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe" `
    -OutFile "C:\hadoop\bin\winutils.exe"
Invoke-WebRequest -Uri "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll" `
    -OutFile "C:\hadoop\bin\hadoop.dll"

# Persist HADOOP_HOME for your user account (no admin required)
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
$env:HADOOP_HOME = "C:\hadoop"

# Verify
Test-Path "C:\hadoop\bin\winutils.exe"   # → True
Test-Path "C:\hadoop\bin\hadoop.dll"     # → True
```

`spark_utils.py` automatically adds `C:\hadoop\bin` to `PATH` at import time so the
Windows DLL loader can find `hadoop.dll` when `winutils.exe` runs.

---

## Step 8 — Run the pipeline

> **Note — Dummy Data:** The pipeline below generates and uses **synthetically created dummy data**
> for learning and testing purposes only. No real customer or business data is involved.
> If you plan to use real-world data, refer to `docs/data_dictionary.md` for the expected schema
> and adjust the generator, validator, and feature engineering scripts to match your actual dataset.

### 8a. Generate synthetic dataset (55K customers, ~5M events)

```powershell
python pipelines/scripts/run_data_generation.py --config config/config.yaml
```

Expected output:
```
Customers generated   total=55000  churn_rate=56%
Transactions generated total=194278
Clickstream generated  total=4343728
Support tickets        total=156578
Data generation complete
```

Generated files appear in `data/raw/`.

---

### 8b. Feature engineering (PySpark)

```powershell
python pipelines/scripts/run_feature_engineering.py --config config/config.yaml --mode train
```

Expected output:
```
SparkSession ready
Feature matrix built  rows=55000  cols=45
Write complete  path=data/processed/features
```

Generated file: `data/processed/features/part-*.parquet`

> First run is slow (~3 min) while Spark downloads its own dependencies.

---

### 8c. Train all survival models + log to MLflow

```powershell
python pipelines/scripts/run_training.py --config config/config.yaml --run-date 2024-06-30
```

This fits KM, Cox PH, Weibull, Log-Normal, Log-Logistic; runs 5-fold CV; logs everything to MLflow.

Expected output:
```
Fitting Kaplan-Meier ...
  km_median_survival_days = 487
Fitting Cox PH ...
  cox_concordance_index = 0.71
Fitting parametric models ...
Cross-validation started  folds=5
  cv_c_index_mean = 0.708  std = 0.012
```

---

### 8d. View results in MLflow UI

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open your browser at **http://localhost:5000**

You will see:
- All training runs with parameters and metrics
- KM survival curve plots
- Cox PH hazard ratio forest plot
- Model selection report CSV

Press `Ctrl+C` to stop the MLflow UI.

---

### 8e. Register champion model (after reviewing MLflow)

```powershell
python pipelines/scripts/run_training.py --config config/config.yaml --run-date 2024-06-30 --register
```

This promotes the Cox PH model to the `Production` stage in the MLflow Model Registry
(only if C-index >= 0.68).

---

### 8f. Run batch scoring

```powershell
python pipelines/scripts/run_scoring.py --config config/config.yaml --score-date 2024-12-31
```

Scores are written to `data/processed/scores/score_date=2024-12-31/`.

---

### 8g. Run unit tests

```powershell
python -m pytest tests/unit -m unit -v
```

Expected: **32 passed** in ~10 seconds.

---

## Step 9 — One-command full pipeline

```powershell
# Generate data + feature engineering + train (skip Spark for feature eng if PySpark not set up)
python pipelines/scripts/run_data_generation.py && `
python pipelines/scripts/run_feature_engineering.py --mode train && `
python pipelines/scripts/run_training.py --run-date 2024-06-30
```

---

## Quick reference — daily commands

```powershell
# Activate venv (always do this first)
churn_pred\Scripts\activate

# Run tests
python -m pytest tests/unit -m unit -v

# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000

# Deactivate venv when done
deactivate
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `java.lang.RuntimeException: java.io.IOException` | JAVA_HOME not set. Run `$env:JAVA_HOME = "C:\Program Files\..."` |
| `ModuleNotFoundError: No module named 'pyspark'` | Virtual env not activated. Run `churn_pred\Scripts\activate` |
| `ModuleNotFoundError: No module named 'pandera.pandas'` | Older pandera installed. Code includes a backward-compatible fallback — run `pip install -r requirements.txt` to upgrade |
| `The system cannot find the path specified` when running PySpark scripts | Missing `winutils.exe`/`hadoop.dll`. Complete **Step 7b** above |
| `Missing Python executable 'python3'` | Older PySpark default. `spark_utils.py` already pins `PYSPARK_PYTHON` to `sys.executable` — ensure you're running through the project scripts, not raw `pyspark-submit` |
| `Parquet column cannot be converted: Expected: timestamp, Found: INT64` | Old Parquet files with wrong physical types. **Delete `data/raw/*` and regenerate** — the fixed `data_generator.py` writes correct PyArrow types (`date32`, `timestamp[us]`, `int32`) |
| `Unable to create Parquet converter for ... TIMESTAMP(NANOS,false)` | Same root cause — regenerate data. The DataLoader also has defensive nanos→micros casting for legacy files |
| `MutableInt cannot be cast to MutableLong` | Schema width mismatch (file has `int64`, DataLoader expects `int32`). Regenerate data with the fixed generator |
| `ConvergenceError: delta contains nan value(s)` (Cox PH) | Already handled in `cox_ph.py` — `_prepare_features` drops near-zero-variance columns, replaces `±inf`, and z-scores. If it still fails, increase `penalizer` in `config/config.yaml` |
| `ValueError: Length of values (N) does not match length of index (M)` in PH test | Schoenfeld residuals must use *training* data. The Cox model caches it automatically — just call `cox.test_proportional_hazard_assumption()` with no args |
| `FileNotFoundError: data/processed/features/part-00000.parquet` | Spark writes UUID-suffixed part files. Already fixed — `load_features()` now reads the directory. Pull latest code |
| `pandera.errors.SchemaErrors` during validation | Data generation produced unexpected values; re-run with `--seed 42` |
| MLflow UI blank after training | Training did not complete; check terminal output for errors |
| PySpark slow on first run | Normal — Spark downloads dependencies to `~/.ivy2/` on first start |
| `permission denied` on `data/` directory | Run PowerShell as Administrator, or change `output_dir` in `config/config.yaml` |
| `NativeCommandError: Setting default log level to "WARN"` in PowerShell | Cosmetic — PowerShell flags Java's stderr output as an error. The pipeline still works; ignore it |
