"""CLI entry point: run PySpark feature engineering pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.utils.logger import configure_logging, get_logger
from src.utils.spark_utils import get_spark_session, stop_spark

log = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="XYZ churn feature engineering")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode",
        choices=["train", "score"],
        default="train",
        help="train: use train_cutoff_date; score: use scoring_as_of_date",
    )
    parser.add_argument("--as-of-date", default=None, help="Override as_of_date (YYYY-MM-DD)")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(level="INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    obs_cfg = cfg["observation"]
    spark_cfg = cfg["spark"]
    feat_cfg = cfg["features"]

    if args.as_of_date:
        as_of_date = args.as_of_date
    elif args.mode == "train":
        as_of_date = obs_cfg["train_cutoff_date"]
    else:
        as_of_date = obs_cfg["scoring_as_of_date"]

    log.info("Feature engineering config", mode=args.mode, as_of_date=as_of_date)

    spark = get_spark_session(
        app_name=spark_cfg["app_name"],
        master=spark_cfg["master"],
        executor_memory=spark_cfg["executor_memory"],
        driver_memory=spark_cfg["driver_memory"],
        shuffle_partitions=spark_cfg["shuffle_partitions"],
        log_level=spark_cfg["log_level"],
    )

    try:
        loader = DataLoader(spark, raw_data_dir=cfg["data_generation"]["output_dir"])
        datasets = loader.load_all()

        pipeline = FeatureEngineeringPipeline(
            spark=spark,
            as_of_date=as_of_date,
            lookback_windows=feat_cfg["lookback_windows_days"],
            output_dir=feat_cfg["output_dir"],
        )
        features_df = pipeline.run(
            customers=datasets["customers"],
            transactions=datasets["transactions"],
            clickstream=datasets["clickstream"],
            support_tickets=datasets["support_tickets"],
            write_output=True,
        )
        log.info("Feature engineering done", rows=features_df.count())
    finally:
        stop_spark(spark)


if __name__ == "__main__":
    main()
