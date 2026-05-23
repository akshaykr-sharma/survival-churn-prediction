"""Unit tests for DataValidator."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_validator import DataValidationError, DataValidator


@pytest.mark.unit
class TestDataValidator:

    def _make_customers(self, n=200):
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "customer_id": [f"CUST{i:07d}" for i in range(n)],
                "registration_date": pd.date_range("2022-01-01", periods=n, freq="D"),
                "first_purchase_date": pd.date_range("2022-01-15", periods=n, freq="D"),
                "segment": rng.choice(["student", "gamer", "professional", "smb", "creative"], n),
                "city": ["Mumbai"] * n,
                "state": ["Maharashtra"] * n,
                "city_tier": rng.choice([1, 2, 3], n),
                "gender": rng.choice(["M", "F", "Other"], n),
                "age_years": rng.integers(18, 60, n),
                "acquisition_channel": rng.choice(["organic_search", "referral"], n),
                "product_category": rng.choice(
                    ["budget_pc", "mid_range_pc", "gaming_pc", "workstation"], n
                ),
                "product_price_inr": rng.uniform(30_000, 200_000, n),
                "warranty_months": rng.choice([12, 24], n),
                "amc_active": rng.choice([0, 1], n),
                "is_churned": rng.choice([0, 1], n),
                "duration_days": rng.integers(1, 1000, n),
            }
        )

    # Small min-row override so unit tests (200 rows) aren't blocked by production thresholds
    _SMALL_MIN = {"customers": 0, "transactions": 0, "clickstream": 0, "support_tickets": 0}

    def test_valid_customer_df_passes(self):
        df = self._make_customers()
        validator = DataValidator(
            expected_customer_ids=set(df["customer_id"]),
            min_row_overrides=self._SMALL_MIN,
        )
        report = validator.validate(df, "customers")
        assert report.passed, f"Expected pass, got errors: {report.errors}"

    def test_duplicate_customer_ids_fail(self):
        df = self._make_customers(50)
        df = pd.concat([df, df.iloc[:5]])
        validator = DataValidator(min_row_overrides=self._SMALL_MIN)
        report = validator.validate(df, "customers")
        assert not report.passed
        assert any("Schema error" in e or "Duplicate" in e for e in report.errors)

    def test_zero_duration_fails(self):
        df = self._make_customers(50)
        df.loc[0, "duration_days"] = 0
        validator = DataValidator(min_row_overrides=self._SMALL_MIN)
        report = validator.validate(df, "customers")
        assert not report.passed

    def test_invalid_segment_fails(self):
        df = self._make_customers(50)
        df.loc[0, "segment"] = "UNKNOWN_SEGMENT"
        validator = DataValidator(min_row_overrides=self._SMALL_MIN)
        report = validator.validate(df, "customers")
        assert not report.passed

    def test_high_null_rate_raises_warning(self):
        df = self._make_customers(100)
        # Add 10% nulls to age_years (should warn, not fail since age_years is nullable)
        df.loc[:9, "age_years"] = None
        validator = DataValidator(min_row_overrides=self._SMALL_MIN)
        report = validator.validate(df, "customers")
        assert any("age_years" in w for w in report.warnings)

    def test_referential_integrity_orphan_customers(self):
        customers = self._make_customers(100)
        txn = pd.DataFrame(
            {
                "transaction_id": [f"TX{i:09d}" for i in range(10)],
                "customer_id": ["ORPHAN_001"] * 10,
                "transaction_date": pd.date_range("2022-01-01", periods=10),
                "transaction_type": ["initial_purchase"] * 10,
                "amount_inr": [50_000.0] * 10,
                "is_returned": [0] * 10,
            }
        )
        customer_ids = set(customers["customer_id"])
        validator = DataValidator(
            expected_customer_ids=customer_ids,
            min_row_overrides=self._SMALL_MIN,
        )
        report = validator.validate(txn, "transactions")
        assert not report.passed

    def test_validate_all_raises_on_failure(self):
        customers = self._make_customers(100)
        bad_txn = pd.DataFrame(
            {
                "transaction_id": ["TX1"],
                "customer_id": ["CUST0000001"],
                "transaction_date": [pd.Timestamp("2022-01-01")],
                "transaction_type": ["initial_purchase"],
                "amount_inr": [-100.0],  # invalid negative amount
                "is_returned": [0],
            }
        )
        with pytest.raises(DataValidationError):
            validator = DataValidator(
                expected_customer_ids={"CUST0000001"},
                min_row_overrides=self._SMALL_MIN,
            )
            validator.validate_all({"customers": customers, "transactions": bad_txn})
