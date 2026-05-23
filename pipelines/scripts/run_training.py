"""
CLI entry point: train survival models and log to MLflow.

Steps:
  1. Load feature matrix from Parquet
  2. Train/test split (temporal: train ≤ train_cutoff, test > train_cutoff)
  3. Fit KM, Cox PH, Weibull, Log-Normal, Log-Logistic
  4. Run model selection (C-index, AIC)
  5. Cross-validate champion
  6. Log all metrics + artifacts to MLflow
  7. Register champion model in MLflow Model Registry (if --register)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.cox_ph import CoxPHModel
from src.models.kaplan_meier import KaplanMeierModel
from src.models.model_selector import ModelSelector
from src.models.parametric import fit_all_parametric
from src.utils.logger import configure_logging, get_logger
from src.utils.mlflow_utils import (
    log_artifact_path,
    log_metrics,
    log_params,
    mlflow_run,
    register_model,
    setup_mlflow,
)

log = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train XYZ churn survival models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-date", default=None, help="Execution date YYYY-MM-DD")
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training; just evaluate the last run"
    )
    parser.add_argument(
        "--register", action="store_true", help="Register champion model to MLflow registry"
    )
    return parser.parse_args()


def load_features(cfg: dict) -> pd.DataFrame:
    """Load the Spark-written feature matrix.

    Spark writes Parquet as a directory of UUID-suffixed part files, not a
    single named file.  pyarrow/pandas read_parquet() handles a directory
    natively — it concatenates all part-*.parquet files into one DataFrame.
    """
    feat_dir = Path(cfg["features"]["output_dir"]) / "features"
    if not feat_dir.exists() or not any(feat_dir.glob("part-*.parquet")):
        raise FileNotFoundError(
            f"Feature matrix not found at {feat_dir}. " "Run run_feature_engineering.py first."
        )
    log.info("Loading feature matrix", path=str(feat_dir))
    return pd.read_parquet(feat_dir)


def temporal_split(df: pd.DataFrame, train_cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by first_purchase_date — customers acquired before cutoff = train."""
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    train = df[df["first_purchase_date"] <= pd.Timestamp(train_cutoff)].copy()
    test = df[df["first_purchase_date"] > pd.Timestamp(train_cutoff)].copy()
    log.info(
        "Train/test split",
        train=len(train),
        test=len(test),
        train_churn_rate=f"{train['is_churned'].mean():.2%}",
        test_churn_rate=f"{test['is_churned'].mean():.2%}",
    )
    return train, test


def main():
    args = parse_args()
    configure_logging(level="INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["mlflow"]
    obs_cfg = cfg["observation"]
    model_cfg = cfg["models"]

    setup_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

    df = load_features(cfg)
    train_df, test_df = temporal_split(df, obs_cfg["train_cutoff_date"])

    run_name = f"training_{args.run_date or 'manual'}"

    with mlflow_run(
        run_name=run_name, tags={"mode": "training", "run_date": args.run_date or "manual"}
    ):

        # ── Log run metadata ──────────────────────────────────────────────────
        log_params(
            {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_cutoff": obs_cfg["train_cutoff_date"],
                "churn_rate_train": round(float(train_df["is_churned"].mean()), 4),
                "churn_rate_test": round(float(test_df["is_churned"].mean()), 4),
            }
        )

        # ── 1. Kaplan-Meier (baseline) ────────────────────────────────────────
        log.info("Fitting Kaplan-Meier …")
        km = KaplanMeierModel(config=model_cfg["kaplan_meier"])
        km.fit(train_df)
        km_metrics = km.get_summary_metrics()
        log_metrics({f"km_{k}": v for k, v in km_metrics.items()})

        # Save KM curve plot
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = os.path.join(tmp, "km_curve.png")
            km.plot_survival_curve(output_path=curve_path)
            log_artifact_path(curve_path, artifact_path="plots")

            # Stratified curves
            for strat_col in ["segment", "product_category"]:
                if strat_col in train_df.columns:
                    km.fit_stratified(train_df, stratify_col=strat_col)
                    strat_path = os.path.join(tmp, f"km_{strat_col}.png")
                    km.plot_stratified_curves(strat_col, output_path=strat_path)
                    log_artifact_path(strat_path, artifact_path="plots")

        # ── 2. Cox PH ─────────────────────────────────────────────────────────
        log.info("Fitting Cox PH …")
        cox_cfg = model_cfg["cox_ph"]
        cox = CoxPHModel(
            penalizer=cox_cfg["penalizer"],
            l1_ratio=cox_cfg.get("l1_ratio", 0.0),
            baseline_estimation_method=cox_cfg["baseline_estimation_method"],
        )
        cox.fit(train_df)
        cox_c_index = cox.compute_concordance_index(test_df)

        log_params(cox.get_model_params())
        log_metrics({"cox_concordance_index": cox_c_index})
        log.info("Cox PH C-index", c_index=round(cox_c_index, 4))

        # PH assumption test (Schoenfeld residuals — computed on training data)
        ph_result = cox.test_proportional_hazard_assumption()
        mlflow.set_tag("ph_violations", str(ph_result["ph_violations"]))

        # Save Cox model artifact
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "cox_ph.pkl")
            cox.save(model_path)
            log_artifact_path(model_path, artifact_path="models")

            forest_path = os.path.join(tmp, "cox_hazard_ratios.png")
            cox.plot_coefficients(top_n=25, output_path=forest_path)
            log_artifact_path(forest_path, artifact_path="plots")

            # Log to MLflow models so it's registered-able
            mlflow.sklearn.log_model(cox, artifact_path="cox_ph_model")

        # ── 3. Parametric models ──────────────────────────────────────────────
        log.info("Fitting parametric models …")
        param_models = fit_all_parametric(train_df)
        for dist_name, model in param_models.items():
            c_idx = model.compute_concordance_index(test_df)
            log_metrics(
                {
                    f"{dist_name}_c_index": c_idx,
                    f"{dist_name}_aic": model.aic,
                    f"{dist_name}_bic": model.bic,
                }
            )

        # ── 4. Cross-validation ───────────────────────────────────────────────
        log.info("Running cross-validation …")
        selector = ModelSelector(cv_folds=model_cfg["model_selection"]["cv_folds"])
        cv_result = selector.cross_validate(train_df, model_name="cox_ph")
        log_metrics(
            {
                "cv_c_index_mean": cv_result["c_index_mean"],
                "cv_c_index_std": cv_result["c_index_std"],
            }
        )
        log.info("CV result", **{k: v for k, v in cv_result.items() if k != "c_indices"})

        # ── 5. Full model selection report ────────────────────────────────────
        report_df = selector.run(train_df, test_df)
        log.info("Model selection report:\n%s", report_df.to_string())

        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "model_selection_report.csv")
            report_df.to_csv(report_path)
            log_artifact_path(report_path, artifact_path="reports")

        # ── 6. Register champion if requested ─────────────────────────────────
        if args.register and cox_c_index >= 0.68:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/cox_ph_model"
            version = register_model(
                model_uri=model_uri,
                registered_model_name=mlflow_cfg["registered_model_name"],
                stage="Production",
                description=f"Cox PH champion — C-index={cox_c_index:.4f}, date={args.run_date}",
            )
            log.info("Champion registered", version=version)
        elif args.register:
            log.warning(
                "C-index below 0.68 — model NOT registered",
                c_index=cox_c_index,
            )


if __name__ == "__main__":
    main()
