"""CLI entry point: run batch survival scoring for all active customers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.drift_detector import DriftDetector
from src.scoring.batch_scorer import BatchScorer
from src.utils.logger import configure_logging, get_logger
from src.utils.spark_utils import get_spark_session, stop_spark

log = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="XYZ churn batch scoring")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--score-date", default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run score validation / drift check, skip scoring",
    )
    parser.add_argument("--export", action="store_true", help="Export scores to downstream systems")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(level="INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    spark_cfg = cfg["spark"]
    scoring_cfg = cfg["scoring"]
    mlflow_cfg = cfg["mlflow"]
    monitor_cfg = cfg["monitoring"]

    # ── 1. Spark session ──────────────────────────────────────────────────────
    spark = get_spark_session(
        app_name=spark_cfg["app_name"],
        master=spark_cfg["master"],
        executor_memory=spark_cfg["executor_memory"],
        driver_memory=spark_cfg["driver_memory"],
        shuffle_partitions=spark_cfg["shuffle_partitions"],
        log_level=spark_cfg["log_level"],
    )

    try:
        feat_path = str(Path(cfg["features"]["output_dir"]) / "features")

        if not args.validate_only:
            # ── 2. Load features ──────────────────────────────────────────────
            log.info("Loading feature matrix", path=feat_path)
            features_df = spark.read.parquet(feat_path)

            # Filter to active customers only (not yet churned)
            active_df = features_df.filter(features_df["is_churned"] == 0)
            log.info("Active customers to score", count=active_df.count())

            # ── 3. Batch scoring ──────────────────────────────────────────────
            import mlflow

            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

            scorer = BatchScorer(
                spark=spark,
                score_date=args.score_date,
                model_name=mlflow_cfg["registered_model_name"],
                model_stage=scoring_cfg["model_stage"],
                high_risk_threshold=scoring_cfg["thresholds"]["high_risk"],
                medium_risk_threshold=scoring_cfg["thresholds"]["medium_risk"],
                output_dir=scoring_cfg["output_dir"],
            )
            scores_df = scorer.score(active_df)
            output_path = scorer.write_scores(scores_df)
            log.info("Scores written", path=output_path)

        # ── 4. Drift detection ────────────────────────────────────────────────
        score_path = str(Path(scoring_cfg["output_dir"]) / f"score_date={args.score_date}")
        if Path(score_path).exists():
            current_scores = pd.read_parquet(score_path)

            # Compare to a reference (previous batch — load most recent)
            ref_path_candidates = sorted(
                Path(scoring_cfg["output_dir"]).glob("score_date=*"),
                reverse=True,
            )
            drift_detected = False
            if len(ref_path_candidates) >= 2:
                ref_path = str(ref_path_candidates[1])
                ref_scores = pd.read_parquet(ref_path)

                detector = DriftDetector(psi_threshold=monitor_cfg["psi_threshold"])
                # Check score distribution drift
                score_check = detector.score_distribution_check(
                    ref_scores["survival_prob_90d"],
                    current_scores["survival_prob_90d"],
                )
                log.info("Score distribution drift check", **score_check)
                drift_detected = score_check["is_drifted"]

            print(json.dumps({"drift_detected": drift_detected}))

        # ── 5. Export (stub — plug in your export target) ─────────────────────
        if args.export:
            log.info("Exporting scores to downstream system …", score_date=args.score_date)
            # Example: copy to S3 / write to BigQuery / push to CRM
            log.info("Export complete (stub)")

    finally:
        stop_spark(spark)


if __name__ == "__main__":
    main()
