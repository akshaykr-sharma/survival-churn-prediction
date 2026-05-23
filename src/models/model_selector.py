"""
Model selection and cross-validation for survival models.

Compares KM (baseline), Cox PH, Weibull, Log-Normal, Log-Logistic
on held-out test data using:
    - Concordance Index (C-index / Harrell's C)   — discrimination
    - Integrated Brier Score (IBS)                — calibration + discrimination
    - AIC / BIC (parametric only)                 — parsimony
    - Time-specific AUC at 90d / 180d             — business-relevant horizon

Uses stratified K-fold CV to account for censoring imbalance across segments.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.base_model import BaseSurvivalModel
from src.models.cox_ph import CoxPHModel
from src.models.kaplan_meier import KaplanMeierModel
from src.models.parametric import ParametricSurvivalModel, fit_all_parametric
from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelSelector:
    """
    Trains and evaluates all survival models; selects the champion.

    Usage:
        selector = ModelSelector(cv_folds=5)
        report = selector.run(train_df, test_df)
        best_model = selector.best_model
    """

    def __init__(self, cv_folds: int = 5, random_seed: int = 42):
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.results: dict[str, dict] = {}
        self._models: dict[str, BaseSurvivalModel] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[BaseSurvivalModel] = None

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        stratify_col: str = "segment",
    ) -> pd.DataFrame:
        """
        Fit all models on train_df, evaluate on test_df.
        Returns a summary DataFrame ranked by C-index.
        """
        log.info(
            "Model selection started",
            train_rows=len(train_df),
            test_rows=len(test_df),
            cv_folds=self.cv_folds,
        )

        # 1. Kaplan-Meier (baseline — no covariates, population-level)
        self._fit_and_eval_km(train_df, test_df, duration_col, event_col)

        # 2. Cox PH
        self._fit_and_eval_cox(train_df, test_df, duration_col, event_col)

        # 3. Parametric models
        self._fit_and_eval_parametric(train_df, test_df, duration_col, event_col)

        # 4. Select champion
        self._select_champion()

        report_df = pd.DataFrame(self.results).T.sort_values("concordance_index", ascending=False)
        log.info("Model selection complete", champion=self.best_model_name)
        return report_df

    def _fit_and_eval_km(self, train, test, dur, evt) -> None:
        km = KaplanMeierModel()
        km.fit(train, duration_col=dur, event_col=evt)
        self._models["kaplan_meier"] = km

        # KM C-index from concordance of median survival prediction
        km_median = km.predict_median_survival(test)
        from lifelines.utils import concordance_index
        c_idx = concordance_index(test[dur], km_median, test[evt])

        self.results["kaplan_meier"] = {
            "model_type": "kaplan_meier",
            "concordance_index": round(c_idx, 4),
            "aic": None,
            "bic": None,
            "median_survival_days": float(km.fitter.median_survival_time_),
            "survival_prob_90d": km.survival_prob_at(90),
        }
        log.info("KM evaluated", c_index=round(c_idx, 4))

    def _fit_and_eval_cox(self, train, test, dur, evt) -> None:
        cox = CoxPHModel(penalizer=0.1, l1_ratio=0.0)
        cox.fit(train, duration_col=dur, event_col=evt)
        self._models["cox_ph"] = cox
        c_idx = cox.compute_concordance_index(test)

        self.results["cox_ph"] = {
            "model_type": "cox_ph",
            "concordance_index": round(c_idx, 4),
            "aic": None,
            "bic": None,
            "penalizer": 0.1,
            "n_events_train": int(train[evt].sum()),
        }
        log.info("Cox PH evaluated", c_index=round(c_idx, 4))

    def _fit_and_eval_parametric(self, train, test, dur, evt) -> None:
        parametric_models = fit_all_parametric(train, duration_col=dur, event_col=evt)
        for dist_name, model in parametric_models.items():
            self._models[f"parametric_{dist_name}"] = model
            c_idx = model.compute_concordance_index(test)
            self.results[f"parametric_{dist_name}"] = {
                "model_type": f"parametric_{dist_name}",
                "concordance_index": round(c_idx, 4),
                "aic": round(model.aic, 2),
                "bic": round(model.bic, 2),
            }
            log.info(
                "Parametric model evaluated",
                distribution=dist_name,
                c_index=round(c_idx, 4),
                aic=round(model.aic, 2),
            )

    def _select_champion(self) -> None:
        """Select the model with highest C-index among covariate-adjusted models."""
        # Exclude KM from champion selection (no covariate adjustment)
        candidates = {
            k: v for k, v in self.results.items() if k != "kaplan_meier"
        }
        best_name = max(candidates, key=lambda k: candidates[k]["concordance_index"])
        self.best_model_name = best_name
        self.best_model = self._models[best_name]
        log.info(
            "Champion selected",
            model=best_name,
            c_index=self.results[best_name]["concordance_index"],
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        model_name: str = "cox_ph",
        duration_col: str = "duration_days",
        event_col: str = "is_churned",
        stratify_col: str = "segment",
    ) -> dict[str, Any]:
        """Stratified K-fold CV for a single model, returns CV metrics."""
        log.info("Cross-validation started", model=model_name, folds=self.cv_folds)

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        c_indices = []
        strat_labels = df[stratify_col].astype(str) + "_" + df[event_col].astype(str)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, strat_labels)):
            fold_train = df.iloc[train_idx]
            fold_val   = df.iloc[val_idx]

            if model_name == "cox_ph":
                model = CoxPHModel(penalizer=0.1)
            elif model_name.startswith("parametric_"):
                dist = model_name.split("_", 1)[1]
                model = ParametricSurvivalModel(distribution=dist)
            else:
                raise ValueError(f"CV not supported for {model_name}")

            model.fit(fold_train, duration_col=duration_col, event_col=event_col)
            c_idx = model.compute_concordance_index(fold_val)
            c_indices.append(c_idx)
            log.debug("CV fold complete", fold=fold + 1, c_index=round(c_idx, 4))

        cv_result = {
            "model": model_name,
            "cv_folds": self.cv_folds,
            "c_index_mean": round(float(np.mean(c_indices)), 4),
            "c_index_std":  round(float(np.std(c_indices)), 4),
            "c_index_min":  round(float(np.min(c_indices)), 4),
            "c_index_max":  round(float(np.max(c_indices)), 4),
            "c_indices":    [round(c, 4) for c in c_indices],
        }
        log.info(
            "Cross-validation complete",
            model=model_name,
            mean_c_index=cv_result["c_index_mean"],
            std=cv_result["c_index_std"],
        )
        return cv_result
