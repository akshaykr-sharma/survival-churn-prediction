"""
Data drift detection for the churn survival pipeline.

Metrics:
  - PSI (Population Stability Index) per feature
  - KL Divergence for score distributions
  - Chi-square test for categorical features
  - Kolmogorov-Smirnov test for continuous features

PSI interpretation:
  < 0.10  → no significant shift
  0.10–0.20 → moderate shift, monitor
  > 0.20  → significant shift, investigate / retrain
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.utils.logger import get_logger

log = get_logger(__name__)

NUMERIC_FEATURES = [
    "days_since_last_transaction",
    "total_transactions",
    "avg_weekly_sessions_90d",
    "session_frequency_trend_90d",
    "total_support_tickets",
    "engagement_score",
    "rfm_score",
    "product_price_inr",
    "days_since_last_web_visit",
]

CATEGORICAL_FEATURES = [
    "segment",
    "city_tier",
    "product_category",
    "acquisition_channel",
]


class DriftDetector:
    """
    Compares a reference (training) distribution to a current (scoring) distribution.

    Usage:
        detector = DriftDetector(psi_threshold=0.20)
        report = detector.run(reference_df, current_df)
    """

    def __init__(
        self,
        psi_threshold: float = 0.20,
        ks_alpha: float = 0.05,
        n_bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.n_bins = n_bins

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute drift metrics for all features.
        Returns a DataFrame with one row per feature and columns:
            feature, psi, ks_stat, ks_pvalue, chi2_stat, chi2_pvalue, is_drifted
        """
        num_cols = [c for c in (numeric_cols or NUMERIC_FEATURES) if c in reference.columns]
        cat_cols = [c for c in (categorical_cols or CATEGORICAL_FEATURES) if c in reference.columns]

        records = []
        for col in num_cols:
            row = self._numeric_drift(reference[col], current[col], col)
            records.append(row)

        for col in cat_cols:
            row = self._categorical_drift(reference[col], current[col], col)
            records.append(row)

        report = pd.DataFrame(records)
        drifted = report[report["is_drifted"]]
        log.info(
            "Drift detection complete",
            features_checked=len(report),
            drifted=len(drifted),
            drifted_features=drifted["feature"].tolist(),
        )
        return report

    def _numeric_drift(self, ref: pd.Series, cur: pd.Series, col: str) -> dict:
        ref_clean = ref.dropna()
        cur_clean = cur.dropna()

        psi = self._psi(ref_clean, cur_clean)
        ks_stat, ks_pval = ks_2samp(ref_clean, cur_clean)

        return {
            "feature": col,
            "type": "numeric",
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "chi2_stat": None,
            "chi2_pvalue": None,
            "is_drifted": (psi > self.psi_threshold) or (ks_pval < self.ks_alpha),
        }

    def _categorical_drift(self, ref: pd.Series, cur: pd.Series, col: str) -> dict:
        categories = set(ref.dropna().unique()) | set(cur.dropna().unique())
        ref_counts = ref.value_counts().reindex(categories, fill_value=0)
        cur_counts = cur.value_counts().reindex(categories, fill_value=0)

        contingency = pd.DataFrame({"ref": ref_counts, "cur": cur_counts})
        chi2, pval, _, _ = chi2_contingency(contingency.T)

        psi = self._psi_categorical(ref_counts, cur_counts)

        return {
            "feature": col,
            "type": "categorical",
            "psi": round(psi, 4),
            "ks_stat": None,
            "ks_pvalue": None,
            "chi2_stat": round(float(chi2), 4),
            "chi2_pvalue": round(float(pval), 4),
            "is_drifted": (psi > self.psi_threshold) or (pval < self.ks_alpha),
        }

    def _psi(self, ref: pd.Series, cur: pd.Series) -> float:
        """PSI for numeric feature via equal-frequency binning on reference."""
        bins = np.percentile(ref, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=bins)[0] / len(cur)

        # Avoid log(0)
        ref_pct = np.clip(ref_pct, 1e-6, None)
        cur_pct = np.clip(cur_pct, 1e-6, None)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    def _psi_categorical(self, ref_counts: pd.Series, cur_counts: pd.Series) -> float:
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        ref_pct = ref_pct.clip(lower=1e-6)
        cur_pct = cur_pct.clip(lower=1e-6)
        return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())

    def score_distribution_check(
        self,
        ref_scores: pd.Series,
        cur_scores: pd.Series,
    ) -> dict:
        """KS + PSI on the churn risk score distribution."""
        ks_stat, ks_pval = ks_2samp(ref_scores.dropna(), cur_scores.dropna())
        psi = self._psi(ref_scores.dropna(), cur_scores.dropna())
        return {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "is_drifted": psi > self.psi_threshold,
        }
