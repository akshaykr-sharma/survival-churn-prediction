"""
Cox Proportional Hazards model with time-varying covariate (TVC) support.

The Cox PH model is the primary production model because:
  * It handles censored observations correctly
  * It is semi-parametric (no baseline hazard assumption)
  * It produces individual-level survival curves given a covariate vector
  * It supports time-varying covariates via the start/stop (counting process) formulation

Proportional Hazards assumption (PH):
    h(t | x) = h0(t) * exp(β'x)
where:
    h0(t)  = baseline hazard (estimated via Breslow/Nelson-Aalen)
    β      = coefficient vector learned from data
    exp(β) = hazard ratio per unit change in x

Model is fitted via lifelines.CoxPHFitter which uses a penalised
partial likelihood approach.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from src.models.base_model import BaseSurvivalModel
from src.utils.logger import get_logger

log = get_logger(__name__)

# Feature columns used in the Cox model
_NUMERIC_COVARIATES = [
    # Lifecycle
    "tenure_days",
    "product_price_inr",
    "warranty_months",
    "amc_active",
    # RFM
    "total_transactions",
    "total_spend_inr",
    "days_since_last_transaction",
    "avg_transaction_value_inr",
    # Windowed transaction
    "tx_count_30d",
    "tx_count_90d",
    "tx_amount_30d",
    "tx_amount_90d",
    # Behavioural
    "avg_weekly_sessions_30d",
    "avg_weekly_sessions_90d",
    "avg_session_dur_s_90d",
    "product_comparison_events_30d",
    "days_since_last_web_visit",
    "session_frequency_trend_90d",
    # Support
    "total_support_tickets",
    "support_tickets_30d",
    "support_tickets_90d",
    "avg_resolution_days",
    "escalation_rate",
    "days_since_last_support_contact",
    # Derived
    "rfm_score",
    "engagement_score",
    "warranty_remaining_days",
    "is_warranty_expired",
]

_CATEGORICAL_COVARIATES = [
    "segment",
    "city_tier",
    "product_category",
    "acquisition_channel",
]


class CoxPHModel(BaseSurvivalModel):
    """
    Production Cox PH model wrapping lifelines.CoxPHFitter.

    Supports:
        * Standard (static covariate) fitting
        * Time-varying covariate (TVC) fitting via start/stop encoding
        * PH assumption testing (Schoenfeld residuals)
        * Individual-level survival function prediction
        * Concordance index (C-index) evaluation
        * MLflow artifact export (pickle + params)
    """

    def __init__(
        self,
        penalizer: float = 0.1,
        l1_ratio: float = 0.0,
        baseline_estimation_method: str = "breslow",
        config: Optional[dict] = None,
    ):
        super().__init__("cox_ph", config)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.baseline_estimation_method = baseline_estimation_method
        self.fitter = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            baseline_estimation_method=baseline_estimation_method,
        )
        self._numeric_covs: list[str] = []
        self._categorical_covs: list[str] = []
        self._dummy_cols: list[str] = []
        self._train_summary: Optional[pd.DataFrame] = None
        # Standardization stats — populated at fit time, used at inference
        self._num_means: Optional[pd.Series] = None
        self._num_stds: Optional[pd.Series] = None
        # Cache prepared training matrix for Schoenfeld residual / PH test
        # (lifelines requires the SAME data the model was fit on)
        self._train_prepared: Optional[pd.DataFrame] = None

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_features(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        fit_mode: bool = True,
    ) -> pd.DataFrame:
        """
        Select, one-hot encode, clean, and standardize features.

        In fit_mode=True the selected columns and standardization stats are
        stored on the instance for use at inference time.

        Robustness steps required for Cox PH convergence:
          1. Replace ±inf with NaN, then fillna(0)
          2. Drop near-zero-variance numeric columns (std < 1e-6) — these
             cause NaN deltas in Newton-Raphson optimization
          3. Z-score numeric features so penalizer applies uniformly and the
             optimizer doesn't blow up on features with vastly different scales
        """
        num_cols = [c for c in _NUMERIC_COVARIATES if c in df.columns]
        cat_cols = [c for c in _CATEGORICAL_COVARIATES if c in df.columns]

        # 1. Build numeric block, clean inf/NaN
        num_df = df[num_cols].copy().astype(float)
        num_df = num_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if fit_mode:
            # 2. Drop low-variance columns
            stds = num_df.std()
            kept_num_cols = stds[stds > 1e-6].index.tolist()
            dropped = [c for c in num_cols if c not in kept_num_cols]
            if dropped:
                log.warning(
                    "Dropping low-variance numeric covariates",
                    dropped=dropped,
                )
            num_df = num_df[kept_num_cols]

            # 3. Compute and store z-score stats
            self._numeric_covs = kept_num_cols
            self._num_means = num_df.mean()
            self._num_stds = num_df.std().replace(0.0, 1.0)
        else:
            # Inference: use stored columns + stats
            num_df = num_df.reindex(columns=self._numeric_covs, fill_value=0.0)

        # 3 (cont.) Apply z-score using the fit-time stats
        num_df = (num_df - self._num_means) / self._num_stds

        # 4. One-hot encode categoricals (drop_first avoids multicollinearity)
        subset = pd.concat(
            [df[[duration_col, event_col]].reset_index(drop=True),
             num_df.reset_index(drop=True)],
            axis=1,
        )
        for cat in cat_cols:
            dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)
            subset = pd.concat([subset, dummies.reset_index(drop=True)], axis=1)
            if fit_mode:
                self._dummy_cols.extend(dummies.columns.tolist())

        if fit_mode:
            self._categorical_covs = cat_cols

        # Final NaN sweep (categorical encoding edge cases)
        subset = subset.fillna(0.0)
        return subset

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        **kwargs,
    ) -> "CoxPHModel":
        log.info("Fitting Cox PH", rows=len(df), penalizer=self.penalizer)

        train_df = self._prepare_features(df, duration_col, event_col, fit_mode=True)
        self.fitter.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )
        self._duration_col = duration_col
        self._event_col = event_col
        self._train_summary = self.fitter.summary
        self._train_prepared = train_df  # needed for Schoenfeld residual test
        self._fitted = True

        log.info(
            "Cox PH fit complete",
            concordance_index=float(self.fitter.concordance_index_),
            n_params=len(self.fitter.params_),
        )
        return self

    def fit_time_varying(
        self,
        long_df: pd.DataFrame,
        start_col: str = "start",
        stop_col: str = "stop",
        event_col: str = "is_churned",
        id_col: str = "customer_id",
        covariate_cols: Optional[list[str]] = None,
    ) -> "CoxPHModel":
        """
        Fit Cox model with time-varying covariates using the counting-process
        (start, stop] formulation.

        `long_df` must be in long format: one row per (customer, time interval)
        with columns: id_col, start_col, stop_col, event_col, covariate_cols.
        """
        log.info("Fitting Cox PH with TVC", rows=len(long_df))
        cols = covariate_cols or [
            c for c in _NUMERIC_COVARIATES if c in long_df.columns
        ]
        keep = [id_col, start_col, stop_col, event_col] + cols
        fit_df = long_df[keep].fillna(0)

        self.fitter.fit(
            fit_df,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col,
            show_progress=False,
        )
        self._fitted = True
        log.info("TVC Cox fit complete", concordance=float(self.fitter.concordance_index_))
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of shape (n_timepoints, n_customers) where
        index = timeline days and columns = row indices of df.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_survival_function()")
        pred_df = self._prepare_features(df, self._duration_col, self._event_col, fit_mode=False)
        # Drop labels from prediction input
        feat_cols = [c for c in pred_df.columns
                     if c not in (self._duration_col, self._event_col)]
        return self.fitter.predict_survival_function(pred_df[feat_cols])

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        pred_df = self._prepare_features(df, self._duration_col, self._event_col, fit_mode=False)
        feat_cols = [c for c in pred_df.columns
                     if c not in (self._duration_col, self._event_col)]
        return self.fitter.predict_median(pred_df[feat_cols])

    def predict_partial_hazard(self, df: pd.DataFrame) -> pd.Series:
        """Returns exp(β'x) — relative risk score per individual."""
        pred_df = self._prepare_features(df, self._duration_col, self._event_col, fit_mode=False)
        feat_cols = [c for c in pred_df.columns
                     if c not in (self._duration_col, self._event_col)]
        return self.fitter.predict_partial_hazard(pred_df[feat_cols])

    # ── Evaluation ────────────────────────────────────────────────────────────

    def compute_concordance_index(self, df: pd.DataFrame) -> float:
        """Harrell's C-index on held-out data."""
        sf = self.predict_survival_function(df)
        # Use survival at 90d as a scalar risk score
        idx = sf.index.get_indexer([90], method="nearest")[0]
        risk_score = 1.0 - sf.iloc[idx].values  # higher = more risk

        c_index = concordance_index(
            df[self._duration_col],
            -risk_score,  # negate: higher risk → shorter survival
            df[self._event_col],
        )
        log.info("C-index computed", c_index=round(c_index, 4))
        return float(c_index)

    def test_proportional_hazard_assumption(
        self, df: Optional[pd.DataFrame] = None, p_threshold: float = 0.05
    ) -> dict:
        """
        Schoenfeld residual test for the PH assumption.

        Schoenfeld residuals are defined relative to the data the model was
        fitted on, so this always uses the cached training matrix.  The `df`
        argument is accepted for backwards compatibility but ignored.

        Flags covariates where p < threshold as PH violators (they need
        time-varying treatment).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        if self._train_prepared is None:
            raise RuntimeError(
                "Training matrix unavailable — refit the model before "
                "calling test_proportional_hazard_assumption()."
            )
        if df is not None:
            log.warning(
                "test_proportional_hazard_assumption ignores its `df` arg; "
                "Schoenfeld residuals are computed on the training data."
            )
        result = proportional_hazard_test(
            self.fitter, self._train_prepared, time_transform="rank"
        )
        violations = result.summary[result.summary["p"] < p_threshold].index.tolist()
        log.info(
            "PH test complete",
            violating_covariates=violations,
            n_violations=len(violations),
        )
        return {
            "test_summary": result.summary.to_dict(),
            "ph_violations": violations,
        }

    # ── Model introspection ───────────────────────────────────────────────────

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Return DataFrame with covariate, coef, exp(coef), p-value, CI."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        summary = self.fitter.summary.copy()
        summary.index.name = "covariate"
        return summary.reset_index()[
            ["covariate", "coef", "exp(coef)", "p", "coef lower 95%", "coef upper 95%"]
        ]

    def get_model_params(self) -> dict[str, Any]:
        return {
            "model_type":                  "cox_ph",
            "penalizer":                   self.penalizer,
            "l1_ratio":                    self.l1_ratio,
            "baseline_estimation_method":  self.baseline_estimation_method,
            "n_covariates":                len(self._numeric_covs) + len(self._dummy_cols),
        }

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_coefficients(
        self,
        top_n: int = 20,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Forest plot of top-N hazard ratios."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        summary = self.get_hazard_ratios().nlargest(top_n, "coef", keep="all")

        fig, ax = plt.subplots(figsize=(10, 0.5 * len(summary) + 2))
        ax.barh(summary["covariate"], summary["exp(coef)"], xerr=[
            summary["exp(coef)"] - np.exp(summary["coef lower 95%"]),
            np.exp(summary["coef upper 95%"]) - summary["exp(coef)"],
        ], color="steelblue", ecolor="black", capsize=4)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Hazard Ratio (exp(coef))")
        ax.set_title(f"Cox PH — Top {top_n} Hazard Ratios")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("CoxPH model saved", path=path)

    @classmethod
    def load(cls, path: str) -> "CoxPHModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.info("CoxPH model loaded", path=path)
        return model
