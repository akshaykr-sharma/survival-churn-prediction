"""
Model performance monitoring — tracks C-index over time and triggers alerts.

Compares the current batch's realised churn labels (lagged by 90d)
against the model's predicted survival probabilities to compute the
realised concordance index and Brier score.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from lifelines.utils import concordance_index

from src.utils.logger import get_logger

log = get_logger(__name__)


class PerformanceMonitor:
    """
    Computes and tracks model performance metrics over time.

    Requires a 'realised' DataFrame with actual churn outcomes for
    customers scored N days ago (once we can observe the ground truth).
    """

    def __init__(
        self,
        c_index_threshold: float = 0.70,
        degradation_threshold: float = 0.02,
        horizon_days: int = 90,
    ):
        self.c_index_threshold = c_index_threshold
        self.degradation_threshold = degradation_threshold
        self.horizon_days = horizon_days
        self._history: list[dict] = []

    def evaluate(
        self,
        realised_df: pd.DataFrame,
        score_col: str = "survival_prob_90d",
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        run_date: Optional[str] = None,
    ) -> dict:
        """
        Compute C-index and Brier score on realised outcomes.

        realised_df must contain:
            - score_col: predicted survival probability at horizon
            - duration_col: actual survival time (days)
            - event_col: actual churn indicator (1 = churned)
        """
        run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")

        risk_scores = 1.0 - realised_df[score_col]  # convert S(t) to risk
        c_idx = concordance_index(
            realised_df[duration_col],
            -risk_scores,
            realised_df[event_col],
        )

        brier = self._brier_score(
            realised_df[event_col].values,
            realised_df[score_col].values,
            self.horizon_days,
            realised_df[duration_col].values,
        )

        record = {
            "run_date":         run_date,
            "c_index":          round(float(c_idx), 4),
            "brier_score":      round(float(brier), 4),
            "n_customers":      len(realised_df),
            "n_churned":        int(realised_df[event_col].sum()),
            "below_threshold":  c_idx < self.c_index_threshold,
        }
        self._history.append(record)

        if record["below_threshold"]:
            log.warning(
                "C-index below threshold — consider retraining",
                c_index=c_idx,
                threshold=self.c_index_threshold,
                run_date=run_date,
            )
        else:
            log.info("Model performance OK", c_index=c_idx, run_date=run_date)

        # Degradation check
        if len(self._history) >= 2:
            prev_c = self._history[-2]["c_index"]
            drop = prev_c - c_idx
            if drop > self.degradation_threshold:
                log.warning(
                    "C-index degradation detected",
                    prev_c_index=prev_c,
                    current_c_index=c_idx,
                    drop=round(drop, 4),
                )
                record["degradation_alert"] = True

        return record

    def _brier_score(
        self,
        events: "np.ndarray",
        survival_probs: "np.ndarray",
        horizon: int,
        durations: "np.ndarray",
    ) -> float:
        """
        Censoring-adjusted Brier score at `horizon` days.
        Simplified version (no IPCW weighting).
        """
        import numpy as np
        # Only evaluate on customers whose observation covers the horizon
        mask = (durations >= horizon) | (events == 1)
        if mask.sum() == 0:
            return float("nan")

        e = events[mask]
        s = survival_probs[mask]
        d = durations[mask]

        # Indicator: did the event happen before horizon?
        y = ((e == 1) & (d <= horizon)).astype(float)
        brier = np.mean((y - (1.0 - s)) ** 2)
        return float(brier)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def should_retrain(self) -> bool:
        """Return True if the last evaluation triggered a retraining signal."""
        if not self._history:
            return False
        last = self._history[-1]
        return last.get("below_threshold", False) or last.get("degradation_alert", False)
