# ── XYZ Churn Survival — project Makefile ──────────────────────────────────
.PHONY: help install install-dev lint format type-check test test-unit \
        test-integration generate-data validate-data train score docs clean

PYTHON      := python
PYTEST      := pytest
SRC_DIR     := src
TEST_DIR    := tests
DATA_DIR    := data
CONFIG      := config/config.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install dev + production dependencies
	pip install -r requirements-dev.txt
	pre-commit install

# ── Code quality ────────────────────────────────────────────────────────────
lint: ## Run flake8 linter
	flake8 $(SRC_DIR) $(TEST_DIR) pipelines --max-line-length=100

format: ## Auto-format with black + isort
	black $(SRC_DIR) $(TEST_DIR) pipelines --line-length=100
	isort $(SRC_DIR) $(TEST_DIR) pipelines

type-check: ## Run mypy type checker
	mypy $(SRC_DIR) --ignore-missing-imports

# ── Tests ───────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=xml -v

test-unit: ## Run unit tests only (fast, no Spark)
	$(PYTEST) $(TEST_DIR)/unit -m unit -v

test-integration: ## Run integration tests (Spark required)
	$(PYTEST) $(TEST_DIR)/integration -m integration -v

# ── Data pipeline ───────────────────────────────────────────────────────────
generate-data: ## Generate synthetic 50K+ customer dataset
	$(PYTHON) pipelines/scripts/run_data_generation.py --config $(CONFIG)

validate-data: ## Run data quality checks on generated data
	$(PYTHON) pipelines/scripts/run_validation.py --config $(CONFIG)

# ── Model pipeline ──────────────────────────────────────────────────────────
train: ## Train KM + Cox PH + Parametric models and log to MLflow
	$(PYTHON) pipelines/scripts/run_training.py --config $(CONFIG)

score: ## Run batch scoring on latest data
	$(PYTHON) pipelines/scripts/run_scoring.py --config $(CONFIG)

# ── MLflow ──────────────────────────────────────────────────────────────────
mlflow-ui: ## Launch MLflow tracking UI
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Full pipeline ────────────────────────────────────────────────────────────
pipeline: generate-data validate-data train score ## Run end-to-end pipeline

# ── Docs ─────────────────────────────────────────────────────────────────────
docs: ## Open model docs in browser
	@echo "Docs are in docs/ — open docs/model_explainers/ to read them."

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove generated artefacts (NOT raw data)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf spark-warehouse metastore_db derby.log
