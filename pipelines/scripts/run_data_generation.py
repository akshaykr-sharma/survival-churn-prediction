"""CLI entry point: generate synthetic 50K+ customer dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data_generator import SyntheticDataGenerator
from src.data.data_validator import DataValidator
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate XYZ churn synthetic dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n-customers", type=int, default=None,
                        help="Override n_customers from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(level="INFO")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gen_cfg = cfg["data_generation"]
    obs_cfg = cfg["observation"]

    n_customers = args.n_customers or gen_cfg["n_customers"]
    seed = args.seed or gen_cfg["random_seed"]

    log.info("Data generation config", n_customers=n_customers, seed=seed)

    generator = SyntheticDataGenerator(
        n_customers=n_customers,
        random_seed=seed,
        obs_start=obs_cfg["start_date"],
        obs_end=obs_cfg["end_date"],
        churn_rate_target=gen_cfg["churn_rate_target"],
        output_dir=gen_cfg["output_dir"],
    )
    datasets = generator.generate_all()

    if not args.skip_validation:
        log.info("Running data validation on generated datasets …")
        validator = DataValidator()
        reports = validator.validate_all(datasets, raise_on_failure=True)
        for name, report in reports.items():
            status = "PASS" if report.passed else "FAIL"
            log.info("Validation result", dataset=name, status=status,
                     errors=report.errors, warnings=report.warnings)

    log.info("Data generation complete")


if __name__ == "__main__":
    main()
