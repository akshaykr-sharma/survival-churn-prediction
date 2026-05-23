"""
Parametric survival models: Weibull, Log-Normal, Log-Logistic.

Parametric models assume a specific distributional form for the baseline hazard
and estimate its parameters from data.  They are:
  * More data-efficient than KM/Cox when the distributional assumption holds
  * Extrapolatable beyond the observed follow-up period
  * Better for generating smooth survival curves

All three models are fitted and compared via AIC/BIC to select the best
distributional family for this dataset.

Distributions:
  Weibull     — captures both increasing (γ>1) and decreasing (γ<1) hazards;
                most common for hardware churn.
  Log-Normal  — hazard rises then falls; appropriate for mid-lifecycle churn.
  Log-Logistic — similar to Log-Normal but with heavier tails.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

from src.models.base_model import BaseSurvivalModel
from src.utils.logger import get_logger

log = get_logger(__name__)

_COVARIATE_COLS = [
    "tenure_days",
    "product_price_inr",
    "warranty_months",
    "amc_active",
    "total_transactions",
    "days_since_last_transaction",
    "avg_transaction_value_inr",
    "tx_count_30d",
    "tx_count_90d",
    "tx_amount_90d",
    "avg_weekly_sessions_90d",
    "days_since_last_web_visit",
    "session_frequency_trend_90d",
    "total_support_tickets",
    "escalation_rate",
    "rfm_score",
    "engagement_score",
    "is_warranty_expired",
]

_DISTRIBUTION_MAP = {
    "weibull": WeibullAFTFitter,
    "log_normal": LogNormalAFTFitter,
    "log_logistic": LogLogisticAFTFitter,
}


class ParametricSurvivalModel(BaseSurvivalModel):
    """
    Accelerated Failure Time (AFT) parametric model.

    AFT formulation:
        log(T) = β'x + σε
    where ε follows a distribution-specific error (Gumbel for Weibull,
    Normal for Log-Normal, Logistic for Log-Logistic).

    This is equivalent to:
        S(t | x) = S_0(t * exp(-β'x))
    meaning covariates accelerate or decelerate the time to event.
    """

    SUPPORTED_DISTRIBUTIONS = list(_DISTRIBUTION_MAP.keys())

    def __init__(
        self,
        distribution: str = "weibull",
        penalizer: float = 0.0,
        config: Optional[dict] = None,
    ):
        if distribution not in _DISTRIBUTION_MAP:
            raise ValueError(
                f"distribution must be one of {self.SUPPORTED_DISTRIBUTIONS}, got '{distribution}'"
            )
        super().__init__(f"parametric_{distribution}", config)
        self.distribution = distribution
        self.penalizer = penalizer
        self.fitter = _DISTRIBUTION_MAP[distribution](penalizer=penalizer)
        self._covariate_cols: list[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        **kwargs,
    ) -> "ParametricSurvivalModel":
        cov_cols = [c for c in _COVARIATE_COLS if c in df.columns]
        self._covariate_cols = cov_cols
        self._duration_col = duration_col
        self._event_col = event_col

        keep_cols = [duration_col, event_col] + cov_cols
        fit_df = df[keep_cols].fillna(0)

        log.info(
            "Fitting parametric model",
            distribution=self.distribution,
            rows=len(fit_df),
            covariates=len(cov_cols),
        )
        self.fitter.fit(
            fit_df,
            duration_col=duration_col,
            event_col=event_col,
        )
        self._fitted = True

        log.info(
            "Parametric fit complete",
            distribution=self.distribution,
            aic=round(float(self.fitter.AIC_), 2),
            bic=round(float(self.fitter.BIC_), 2),
            concordance=round(float(self.fitter.concordance_index_), 4),
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        pred_df = df[self._covariate_cols].fillna(0)
        return self.fitter.predict_survival_function(pred_df)

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        pred_df = df[self._covariate_cols].fillna(0)
        return self.fitter.predict_median(pred_df)

    def predict_expectation(self, df: pd.DataFrame) -> pd.Series:
        """Expected survival time E[T] per individual."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        pred_df = df[self._covariate_cols].fillna(0)
        return self.fitter.predict_expectation(pred_df)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def compute_concordance_index(self, df: pd.DataFrame) -> float:
        sf = self.predict_survival_function(df)
        idx = sf.index.get_indexer([90], method="nearest")[0]
        risk_score = 1.0 - sf.iloc[idx].values
        c_idx = concordance_index(df[self._duration_col], -risk_score, df[self._event_col])
        return float(c_idx)

    @property
    def aic(self) -> float:
        return float(self.fitter.AIC_) if self._fitted else float("inf")

    @property
    def bic(self) -> float:
        return float(self.fitter.BIC_) if self._fitted else float("inf")

    def get_model_params(self) -> dict[str, Any]:
        params = {
            "model_type": f"parametric_{self.distribution}",
            "distribution": self.distribution,
            "penalizer": self.penalizer,
            "n_covariates": len(self._covariate_cols),
        }
        if self._fitted:
            params["aic"] = self.aic
            params["bic"] = self.bic
            params["concordance_index"] = float(self.fitter.concordance_index_)
        return params

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_survival_curves_sample(
        self,
        df: pd.DataFrame,
        n_samples: int = 50,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot individual survival curves for a random sample."""
        sample = df.sample(min(n_samples, len(df)), random_state=42)
        sf = self.predict_survival_function(sample)

        fig, ax = plt.subplots(figsize=(11, 6))
        for col in sf.columns:
            ax.plot(sf.index, sf[col], alpha=0.2, color="steelblue", linewidth=0.8)

        # Overlay population mean
        ax.plot(sf.index, sf.mean(axis=1), color="darkred", linewidth=2.5, label="Population mean")

        ax.set_xlabel("Days since first purchase")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_title(title or f"Parametric ({self.distribution}) — Individual Curves")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Parametric model saved", path=path, distribution=self.distribution)

    @classmethod
    def load(cls, path: str) -> "ParametricSurvivalModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.info("Parametric model loaded", path=path)
        return model


# ── Convenience: fit all three and compare ────────────────────────────────────


def fit_all_parametric(
    df: pd.DataFrame,
    duration_col: str = "duration_days",
    event_col: str = "is_churned",
) -> dict[str, ParametricSurvivalModel]:
    """Fit Weibull, Log-Normal, Log-Logistic and return all three."""
    results = {}
    for dist in ParametricSurvivalModel.SUPPORTED_DISTRIBUTIONS:
        model = ParametricSurvivalModel(distribution=dist)
        model.fit(df, duration_col=duration_col, event_col=event_col)
        results[dist] = model

    comparison = {
        dist: {"aic": m.aic, "bic": m.bic, "c_index": m.compute_concordance_index(df)}
        for dist, m in results.items()
    }
    best = min(comparison, key=lambda d: comparison[d]["aic"])
    log.info("Parametric model comparison", comparison=comparison, best_by_aic=best)

    return results
