"""
Schema + statistical data validation using Pandera.

Validates all four raw datasets before feature engineering runs.
Raises DataValidationError on hard failures; logs warnings on soft checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# pandera.pandas submodule exists in >= 0.18.3; fall back to top-level for older installs
try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:
    import pandera as pa  # type: ignore[no-redef]
    from pandera import Check, Column, DataFrameSchema  # type: ignore[no-redef]

try:
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # older pandera only has SchemaError
    from pandera.errors import SchemaError  # type: ignore[assignment]

    SchemaErrors = SchemaError  # type: ignore[misc,assignment]

from src.utils.logger import get_logger

log = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when a dataset fails hard schema constraints."""


@dataclass
class ValidationReport:
    dataset: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    null_summary: dict = field(default_factory=dict)


# ── Pandera schemas ───────────────────────────────────────────────────────────

CUSTOMERS_SCHEMA = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False, unique=True),
        "registration_date": Column(
            pa.DateTime,
            nullable=False,
            checks=Check.greater_than_or_equal_to(pd.Timestamp("2022-01-01")),
        ),
        "first_purchase_date": Column(pa.DateTime, nullable=False),
        "segment": Column(
            str,
            nullable=False,
            checks=Check.isin(["student", "gamer", "professional", "smb", "creative"]),
        ),
        "city": Column(str, nullable=False),
        "state": Column(str, nullable=False),
        "city_tier": Column(int, nullable=False, checks=Check.isin([1, 2, 3])),
        "gender": Column(str, nullable=False, checks=Check.isin(["M", "F", "Other"])),
        "age_years": Column(
            "Int64",
            nullable=True,
            checks=[Check.greater_than_or_equal_to(18), Check.less_than_or_equal_to(80)],
        ),
        "acquisition_channel": Column(str, nullable=False),
        "product_category": Column(
            str,
            nullable=False,
            checks=Check.isin(["budget_pc", "mid_range_pc", "gaming_pc", "workstation"]),
        ),
        "product_price_inr": Column(float, nullable=False, checks=Check.greater_than(0)),
        "warranty_months": Column(int, nullable=False, checks=Check.isin([12, 24])),
        "amc_active": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "is_churned": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "duration_days": Column(int, nullable=False, checks=Check.greater_than(0)),
    },
    coerce=True,
    strict=False,
)

TRANSACTIONS_SCHEMA = DataFrameSchema(
    columns={
        "transaction_id": Column(str, nullable=False, unique=True),
        "customer_id": Column(str, nullable=False),
        "transaction_date": Column(pa.DateTime, nullable=False),
        "transaction_type": Column(str, nullable=False),
        "amount_inr": Column(float, nullable=False, checks=Check.greater_than(0)),
        "is_returned": Column(int, nullable=False, checks=Check.isin([0, 1])),
    },
    coerce=True,
    strict=False,
)

CLICKSTREAM_SCHEMA = DataFrameSchema(
    columns={
        "event_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "event_timestamp": Column(pa.DateTime, nullable=False),
        "event_type": Column(str, nullable=False),
        "session_duration_s": Column(int, nullable=False, checks=Check.greater_than_or_equal_to(0)),
        "pages_in_session": Column(int, nullable=False, checks=Check.greater_than(0)),
    },
    coerce=True,
    strict=False,
)

SUPPORT_SCHEMA = DataFrameSchema(
    columns={
        "ticket_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "ticket_date": Column(pa.DateTime, nullable=False),
        "resolution_days": Column(int, nullable=False, checks=Check.greater_than_or_equal_to(0)),
        "is_escalated": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "is_resolved": Column(int, nullable=False, checks=Check.isin([0, 1])),
    },
    coerce=True,
    strict=False,
)

SCHEMA_MAP = {
    "customers": CUSTOMERS_SCHEMA,
    "transactions": TRANSACTIONS_SCHEMA,
    "clickstream": CLICKSTREAM_SCHEMA,
    "support_tickets": SUPPORT_SCHEMA,
}


# ── Validator class ───────────────────────────────────────────────────────────


class DataValidator:
    """Validates raw datasets before they enter the feature-engineering pipeline."""

    # Default minimums for production; tests can pass a smaller override via min_row_overrides
    DEFAULT_MIN_ROWS = {
        "customers": 40_000,
        "transactions": 100_000,
        "clickstream": 400_000,
        "support_tickets": 50_000,
    }

    def __init__(
        self,
        expected_customer_ids: Optional[set] = None,
        min_row_overrides: Optional[dict] = None,
    ):
        self._customer_ids = expected_customer_ids
        self._min_rows = {**self.DEFAULT_MIN_ROWS, **(min_row_overrides or {})}

    def validate(self, df: pd.DataFrame, dataset_name: str) -> ValidationReport:
        report = ValidationReport(dataset=dataset_name, passed=True, row_count=len(df))

        # 1. Schema validation
        schema = SCHEMA_MAP.get(dataset_name)
        if schema is None:
            report.warnings.append(
                f"No Pandera schema registered for '{dataset_name}' — skipping schema check."
            )
        else:
            try:
                schema.validate(df, lazy=True)
                log.info("Schema validation passed", dataset=dataset_name)
            except (SchemaError, SchemaErrors) as exc:
                report.passed = False
                report.errors.append(f"Schema error: {exc}")
                log.error("Schema validation failed", dataset=dataset_name, error=str(exc)[:200])

        # 2. Null audit
        null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        report.null_summary = null_pct
        high_null_cols = {k: v for k, v in null_pct.items() if v > 5.0}
        for col, pct in high_null_cols.items():
            report.warnings.append(f"Column '{col}' has {pct:.1f}% nulls — exceeds 5% threshold.")
            log.warning("High null rate", dataset=dataset_name, column=col, pct=pct)

        # 3. Row count checks
        min_count = self._min_rows.get(dataset_name, 0)
        if len(df) < min_count:
            report.passed = False
            msg = f"Row count {len(df)} < minimum {min_count}"
            report.errors.append(msg)
            log.error("Insufficient rows", dataset=dataset_name, count=len(df), minimum=min_count)

        # 4. Referential integrity: all customer_ids must exist in customers table
        if dataset_name != "customers" and self._customer_ids is not None:
            orphan_ids = set(df["customer_id"].unique()) - self._customer_ids
            if orphan_ids:
                pct = len(orphan_ids) / df["customer_id"].nunique() * 100
                if pct > 1.0:
                    report.passed = False
                    report.errors.append(f"{len(orphan_ids)} orphan customer_ids ({pct:.2f}%)")
                else:
                    report.warnings.append(
                        f"{len(orphan_ids)} orphan customer_ids ({pct:.2f}%) — within tolerance."
                    )

        # 5. Dataset-specific statistical checks
        if dataset_name == "customers":
            self._check_customers(df, report)
        elif dataset_name == "transactions":
            self._check_transactions(df, report)

        log.info(
            "Validation complete",
            dataset=dataset_name,
            passed=report.passed,
            errors=len(report.errors),
            warnings=len(report.warnings),
        )
        return report

    def _check_customers(self, df: pd.DataFrame, report: ValidationReport) -> None:
        churn_rate = df["is_churned"].mean()
        if not (0.10 <= churn_rate <= 0.55):
            report.warnings.append(
                f"Churn rate {churn_rate:.2%} is outside expected range [10%, 55%]."
            )

        neg_duration = (df["duration_days"] <= 0).sum()
        if neg_duration > 0:
            report.passed = False
            report.errors.append(f"{neg_duration} customers have duration_days <= 0.")

        if df["customer_id"].duplicated().any():
            report.passed = False
            report.errors.append("Duplicate customer_ids detected.")

    def _check_transactions(self, df: pd.DataFrame, report: ValidationReport) -> None:
        neg_amounts = (df["amount_inr"] <= 0).sum()
        if neg_amounts > 0:
            report.passed = False
            report.errors.append(f"{neg_amounts} transactions with amount_inr <= 0.")

    def validate_all(
        self,
        datasets: dict[str, pd.DataFrame],
        raise_on_failure: bool = True,
    ) -> dict[str, ValidationReport]:
        # Seed referential integrity checker with customer IDs
        if "customers" in datasets:
            self._customer_ids = set(datasets["customers"]["customer_id"])

        reports = {}
        for name, df in datasets.items():
            reports[name] = self.validate(df, name)

        failed = [name for name, r in reports.items() if not r.passed]
        if failed and raise_on_failure:
            raise DataValidationError(
                f"Data validation failed for datasets: {failed}. "
                "Check the validation reports for details."
            )
        return reports
