"""Unit tests for DriftDetector."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector


@pytest.mark.unit
class TestDriftDetector:

    def _make_df(self, n=1000, shifted=False, seed=42):
        rng = np.random.default_rng(seed)
        shift = 5.0 if shifted else 0.0
        return pd.DataFrame(
            {
                "days_since_last_transaction": rng.normal(90 + shift, 60, n).clip(0),
                "avg_weekly_sessions_90d": rng.exponential(2 + shift, n).clip(0),
                "engagement_score": rng.beta(2, 5, n),
                "segment": rng.choice(["student", "gamer", "professional", "smb", "creative"], n),
                "city_tier": rng.choice([1, 2, 3], n),
            }
        )

    def test_no_drift_on_same_distribution(self):
        ref = self._make_df(1000, shifted=False, seed=0)
        cur = self._make_df(1000, shifted=False, seed=1)
        detector = DriftDetector(psi_threshold=0.20)
        report = detector.run(
            ref,
            cur,
            numeric_cols=["days_since_last_transaction", "avg_weekly_sessions_90d"],
            categorical_cols=["segment"],
        )
        drifted = report[report["is_drifted"]]
        # Should detect very few or zero drifted features
        assert len(drifted) <= 1

    def test_drift_detected_on_shifted_distribution(self):
        ref = self._make_df(2000, shifted=False, seed=0)
        cur = self._make_df(2000, shifted=True, seed=99)
        detector = DriftDetector(psi_threshold=0.10)
        report = detector.run(
            ref, cur, numeric_cols=["days_since_last_transaction", "avg_weekly_sessions_90d"]
        )
        drifted = report[report["is_drifted"]]
        assert len(drifted) >= 1

    def test_report_has_expected_columns(self):
        ref = self._make_df(500)
        cur = self._make_df(500)
        detector = DriftDetector()
        report = detector.run(
            ref, cur, numeric_cols=["engagement_score"], categorical_cols=["city_tier"]
        )
        for col in ["feature", "type", "psi", "is_drifted"]:
            assert col in report.columns

    def test_score_distribution_check(self):
        rng = np.random.default_rng(0)
        ref_scores = pd.Series(rng.beta(5, 2, 1000))
        cur_scores = pd.Series(rng.beta(2, 5, 1000))  # shifted distribution
        detector = DriftDetector(psi_threshold=0.10)
        result = detector.score_distribution_check(ref_scores, cur_scores)
        assert result["is_drifted"] is True
        assert "psi" in result

    def test_psi_zero_identical_distributions(self):
        rng = np.random.default_rng(7)
        data = pd.Series(rng.normal(50, 10, 5000))
        detector = DriftDetector()
        psi = detector._psi(data, data)
        assert psi == pytest.approx(0.0, abs=0.01)
