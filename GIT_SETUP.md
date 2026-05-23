# Git Setup — Push to Remote Repository

Commands to initialise a local Git repo and push to GitHub (or any remote).

---

## One-time setup: configure Git identity (skip if already done)

```powershell
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
```

---

## Step 1 — Initialise local repository

```powershell
cd D:\self_projects\xyz-churn-survival

git init
git branch -M main
```

---

## Step 2 — Stage all project files

```powershell
git add .
```

Verify what will be committed (data files are excluded by `.gitignore`):

```powershell
git status
```

You should see only source code, configs, docs, tests — **not** `data/raw/`, `data/processed/`, `mlruns/`, or `churn_pred/`.

---

## Step 3 — First commit

```powershell
git commit -m "feat: initial production survival churn model

- KM, Cox PH, Weibull/Log-Normal/Log-Logistic survival models
- 55K synthetic India PC market dataset generator
- PySpark feature engineering pipeline (RFM + behavioural + support)
- MLflow experiment tracking + model registry
- Airflow DAGs: training (monthly), scoring (daily), validation (daily)
- GitHub Actions: CI, model quality gate (C-index >= 0.68), batch scoring
- 32 unit tests, 100% pass rate
- Full documentation: README, model explainers, data dictionary, MLOps guide"
```

---

## Step 4 — Create a repository on GitHub

1. Go to https://github.com/new
2. Repository name: `xyz-churn-survival`
3. Visibility: **Private** (recommended — contains business logic)
4. Do NOT initialise with README (you already have one)
5. Click **Create repository**
6. Copy the repository URL shown (e.g. `https://github.com/YOUR_USERNAME/xyz-churn-survival.git`)

---

## Step 5 — Add remote and push

```powershell
# Replace with your actual GitHub URL
git remote add origin https://github.com/YOUR_USERNAME/xyz-churn-survival.git

# Push main branch and set upstream tracking
git push -u origin main
```

GitHub will prompt for credentials (use a **Personal Access Token**, not your password).
To create a token: GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic) → check `repo` scope.

---

## Step 6 — Verify on GitHub

Open `https://github.com/YOUR_USERNAME/xyz-churn-survival` — you should see all files.

---

## Ongoing workflow (after initial push)

```powershell
# Always activate venv first
churn_pred\Scripts\activate

# Create a feature branch for new work
git checkout -b feature/add-time-varying-covariates

# ... make changes ...

# Stage and commit
git add src/models/cox_ph.py tests/unit/test_models.py
git commit -m "feat: add time-varying covariate support to Cox PH"

# Push feature branch
git push -u origin feature/add-time-varying-covariates

# Open a Pull Request on GitHub → GitHub Actions CI will run automatically
```

---

## Protect the main branch (recommended)

On GitHub: **Settings → Branches → Add branch protection rule**
- Branch name pattern: `main`
- ✅ Require status checks to pass before merging
  - Select: `lint`, `type-check`, `unit-tests`, `data-smoke-test`
- ✅ Require pull request reviews before merging (1 reviewer)
- ✅ Do not allow bypassing the above settings

This enforces that every PR must pass the full CI pipeline before merge.

---

## Useful Git commands

```powershell
# See all branches
git branch -a

# See last 5 commits
git log --oneline -5

# See what changed before committing
git diff

# Undo staged changes (safe — does not delete files)
git restore --staged .

# Pull latest from remote
git pull origin main

# See remote URL
git remote -v
```
