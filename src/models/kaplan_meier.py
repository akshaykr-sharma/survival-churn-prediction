"""
Kaplan-Meier survival estimator.

Used for:
    * Population-level and stratified survival curves (descriptive)
    * Establishing baseline churn rates at key time horizons
    * Log-rank tests between customer segments

KM is non-parametric — it makes no distributional assumption and handles
right-censoring correctly.  It does NOT support covariate adjustment;
use CoxPH for that.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from src.models.base_model import BaseSurvivalModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class KaplanMeierModel(BaseSurvivalModel):
    """
    Wrapper around lifelines.KaplanMeierFitter with stratification,
    log-rank testing, and MLflow-friendly artefact export.
    """

    def __init__(self, config: Optional[dict] = None, alpha: float = 0.05):
        super().__init__("kaplan_meier", config)
        self.alpha = alpha
        self.fitter = KaplanMeierFitter(alpha=alpha, label="XYZ Churn KM")
        self._stratified_fitters: dict[str, KaplanMeierFitter] = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        **kwargs,
    ) -> "KaplanMeierModel":
        log.info("Fitting Kaplan-Meier", rows=len(df))
        self.fitter.fit(df[duration_col], event_observed=df[event_col])
        self._duration_col = duration_col
        self._event_col = event_col
        self._fitted = True
        log.info(
            "KM fit complete",
            median_survival=self.fitter.median_survival_time_,
        )
        return self

    def fit_stratified(
        self,
        df: pd.DataFrame,
        stratify_col: str,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
    ) -> dict[str, KaplanMeierFitter]:
        """Fit one KM fitter per stratum and store internally."""
        log.info("Fitting stratified KM", stratify_col=stratify_col)
        self._stratified_fitters = {}

        for group, gdf in df.groupby(stratify_col):
            kmf = KaplanMeierFitter(alpha=self.alpha, label=str(group))
            kmf.fit(gdf[duration_col], event_observed=gdf[event_col])
            self._stratified_fitters[str(group)] = kmf
            log.debug(
                "Stratum KM fitted",
                group=group,
                n=len(gdf),
                median=kmf.median_survival_time_,
            )

        return self._stratified_fitters

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For KM, the survival function is population-level (not individualised).
        Returns the fitted S(t) curve as a single-column DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_survival_function()")
        return self.fitter.survival_function_

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """All individuals share the population-level median for KM."""
        median = self.fitter.median_survival_time_
        return pd.Series([median] * len(df), name="median_survival_days")

    def survival_prob_at(self, t: int) -> float:
        """Return S(t) at a given time horizon."""
        sf = self.fitter.survival_function_
        idx = sf.index.get_indexer([t], method="nearest")[0]
        return float(sf.iloc[idx].values[0])

    # ── Log-rank tests ────────────────────────────────────────────────────────

    def log_rank_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
    ) -> dict:
        """
        Run a log-rank test across all pairs of groups in group_col.
        Returns dict with p-values and test statistics.
        """
        groups = df[group_col].unique()
        results = {}

        if len(groups) == 2:
            g1, g2 = groups
            d1 = df[df[group_col] == g1]
            d2 = df[df[group_col] == g2]
            test = logrank_test(
                d1[duration_col],
                d2[duration_col],
                event_observed_A=d1[event_col],
                event_observed_B=d2[event_col],
            )
            results[f"{g1}_vs_{g2}"] = {
                "test_statistic": float(test.test_statistic),
                "p_value": float(test.p_value),
                "significant_at_05": test.p_value < 0.05,
            }
        else:
            # Multivariate log-rank
            result = multivariate_logrank_test(
                df[duration_col], df[group_col], event_col=df[event_col]
            )
            results["multivariate"] = {
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "significant_at_05": result.p_value < 0.05,
            }

        log.info("Log-rank tests complete", group_col=group_col, results=results)
        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_model_params(self) -> dict[str, Any]:
        return {
            "model_type": "kaplan_meier",
            "alpha": self.alpha,
            "median_survival_days": (
                float(self.fitter.median_survival_time_) if self._fitted else None
            ),
        }

    def get_summary_metrics(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        return {
            "median_survival_days": float(self.fitter.median_survival_time_),
            "survival_prob_90d": self.survival_prob_at(90),
            "survival_prob_180d": self.survival_prob_at(180),
            "survival_prob_365d": self.survival_prob_at(365),
        }

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot_survival_curve(
        self,
        title: str = "XYZ India PC — Kaplan-Meier Survival Curve",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        if not self._fitted:
            raise RuntimeError("Call fit() before plotting.")

        fig, ax = plt.subplots(figsize=(10, 6))
        self.fitter.plot_survival_function(
            ax=ax,
            ci_show=True,
            at_risk_counts=True,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Days since first purchase")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_ylim(0, 1.05)
        ax.axvline(90, color="orange", linestyle="--", alpha=0.6, label="90d")
        ax.axvline(180, color="red", linestyle="--", alpha=0.6, label="180d")
        ax.axvline(365, color="darkred", linestyle="--", alpha=0.6, label="365d")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info("KM curve saved", path=output_path)

        return fig

    def plot_stratified_curves(
        self,
        stratify_col: str,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        if not self._stratified_fitters:
            raise RuntimeError("Call fit_stratified() first.")

        fig, ax = plt.subplots(figsize=(11, 7))
        for label, kmf in self._stratified_fitters.items():
            kmf.plot_survival_function(ax=ax, ci_show=False)

        ax.set_title(title or f"KM Survival by {stratify_col}", fontsize=14)
        ax.set_xlabel("Days since first purchase")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig
