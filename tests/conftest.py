"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def small_customer_df():
    """Minimal customer DataFrame for unit tests (no Spark needed)."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "customer_id":          [f"CUST{i:07d}" for i in range(n)],
        "duration_days":        rng.integers(30, 1095, n).tolist(),
        "is_churned":           rng.choice([0, 1], n, p=[0.70, 0.30]).tolist(),
        "segment":              rng.choice(
            ["student", "gamer", "professional", "smb", "creative"], n
        ).tolist(),
        "city_tier":            rng.choice([1, 2, 3], n).tolist(),
        "product_category":     rng.choice(
            ["budget_pc", "mid_range_pc", "gaming_pc", "workstation"], n
        ).tolist(),
        "product_price_inr":    rng.uniform(30_000, 250_000, n).tolist(),
        "warranty_months":      rng.choice([12, 24], n).tolist(),
        "amc_active":           rng.choice([0, 1], n).tolist(),
        "tenure_days":          rng.integers(30, 1095, n).tolist(),
        "first_purchase_date":  pd.date_range("2022-01-01", periods=n, freq="8h").tolist(),
        "observation_end_date": pd.date_range("2024-01-01", periods=n, freq="8h").tolist(),
        # RFM
        "total_transactions":   rng.integers(1, 15, n).tolist(),
        "total_spend_inr":      rng.uniform(30_000, 400_000, n).tolist(),
        "days_since_last_transaction": rng.integers(0, 400, n).tolist(),
        "avg_transaction_value_inr":   rng.uniform(500, 50_000, n).tolist(),
        "tx_count_30d":         rng.integers(0, 4, n).tolist(),
        "tx_count_90d":         rng.integers(0, 8, n).tolist(),
        "tx_amount_30d":        rng.uniform(0, 50_000, n).tolist(),
        "tx_amount_90d":        rng.uniform(0, 150_000, n).tolist(),
        # Behavioural
        "avg_weekly_sessions_30d": rng.uniform(0, 5, n).tolist(),
        "avg_weekly_sessions_90d": rng.uniform(0, 5, n).tolist(),
        "avg_session_dur_s_90d":   rng.uniform(30, 600, n).tolist(),
        "product_comparison_events_30d": rng.integers(0, 10, n).tolist(),
        "days_since_last_web_visit": rng.integers(0, 200, n).tolist(),
        "session_frequency_trend_90d": rng.uniform(-2, 2, n).tolist(),
        # Support
        "total_support_tickets":       rng.integers(0, 8, n).tolist(),
        "support_tickets_30d":         rng.integers(0, 3, n).tolist(),
        "support_tickets_90d":         rng.integers(0, 5, n).tolist(),
        "avg_resolution_days":         rng.uniform(1, 10, n).tolist(),
        "escalation_rate":             rng.uniform(0, 0.5, n).tolist(),
        "days_since_last_support_contact": rng.integers(0, 400, n).tolist(),
        # Derived
        "rfm_score":          rng.integers(3, 16, n).tolist(),
        "engagement_score":   rng.uniform(0, 1, n).tolist(),
        "warranty_remaining_days": rng.integers(-100, 730, n).tolist(),
        "is_warranty_expired":     rng.choice([0, 1], n).tolist(),
        "months_since_last_transaction": rng.uniform(0, 14, n).tolist(),
        "rfm_recency_score":  rng.integers(1, 6, n).tolist(),
        "rfm_frequency_score": rng.integers(1, 6, n).tolist(),
        "rfm_monetary_score": rng.integers(1, 6, n).tolist(),
        "acquisition_channel": rng.choice(
            ["organic_search", "paid_search", "social_media", "referral",
             "retail_store", "email_campaign"], n
        ).tolist(),
    })


@pytest.fixture(scope="session")
def train_test_split(small_customer_df):
    """Return (train, test) split — 80/20."""
    n = len(small_customer_df)
    split = int(n * 0.8)
    train = small_customer_df.iloc[:split].copy()
    test  = small_customer_df.iloc[split:].copy()
    return train, test
