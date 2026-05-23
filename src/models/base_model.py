"""Abstract base class for all survival models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class BaseSurvivalModel(ABC):
    """
    Contract that every survival model in this project must satisfy.

    Concrete models implement fit() and predict_survival_function().
    The base class provides shared helpers for persistence and metric logging.
    """

    def __init__(self, model_name: str, config: Optional[dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self._fitted = False

    @abstractmethod
    def fit(
        self, df: pd.DataFrame, duration_col: str, event_col: str, **kwargs
    ) -> "BaseSurvivalModel":
        """Fit the model on the training DataFrame."""

    @abstractmethod
    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame where index = timeline (days) and
        columns = customer_id (or integer index), values = S(t).
        """

    @abstractmethod
    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """Return median survival time (days) per individual."""

    @abstractmethod
    def get_model_params(self) -> dict[str, Any]:
        """Return serialisable dict of model hyperparameters."""

    def is_fitted(self) -> bool:
        return self._fitted

    def survival_at_horizons(
        self, df: pd.DataFrame, horizons: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """
        Return S(t) at specific time horizons for each individual.
        Output columns: customer_id (if present), survival_prob_Xd for each horizon.
        """
        if horizons is None:
            horizons = [30, 60, 90, 180]
        sf = self.predict_survival_function(df)
        result_cols = {}
        for t in horizons:
            idx = sf.index.get_indexer([t], method="nearest")[0]
            result_cols[f"survival_prob_{t}d"] = sf.iloc[idx].values
        out = pd.DataFrame(result_cols)
        if "customer_id" in df.columns:
            out.insert(0, "customer_id", df["customer_id"].values)
        return out
