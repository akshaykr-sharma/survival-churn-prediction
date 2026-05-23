"""Unit tests for KM, Cox PH, and parametric survival models."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
class TestKaplanMeier:

    def test_fit_returns_self(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        result = km.fit(small_customer_df)
        assert result is km

    def test_fitted_flag(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        assert not km.is_fitted()
        km.fit(small_customer_df)
        assert km.is_fitted()

    def test_survival_prob_at_t0_is_one(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        km.fit(small_customer_df)
        sf = km.predict_survival_function(small_customer_df)
        assert float(sf.iloc[0].values[0]) == pytest.approx(1.0, abs=0.01)

    def test_survival_prob_monotone_decreasing(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        km.fit(small_customer_df)
        sf = km.predict_survival_function(small_customer_df)
        vals = sf.iloc[:, 0].values
        assert np.all(vals[:-1] >= vals[1:] - 1e-9)

    def test_summary_metrics_keys(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        km.fit(small_customer_df)
        metrics = km.get_summary_metrics()
        for key in [
            "median_survival_days",
            "survival_prob_90d",
            "survival_prob_180d",
            "survival_prob_365d",
        ]:
            assert key in metrics

    def test_stratified_fit(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        km.fit(small_customer_df)
        strat = km.fit_stratified(small_customer_df, "segment")
        assert len(strat) == 5  # 5 segments

    def test_predict_before_fit_raises(self):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        with pytest.raises(RuntimeError):
            km.predict_survival_function(pd.DataFrame())

    def test_log_rank_two_groups(self, small_customer_df):
        from src.models.kaplan_meier import KaplanMeierModel

        km = KaplanMeierModel()
        km.fit(small_customer_df)
        subset = small_customer_df[small_customer_df["city_tier"].isin([1, 2])].copy()
        result = km.log_rank_test(subset, group_col="city_tier")
        assert "p_value" in list(result.values())[0]


@pytest.mark.unit
class TestCoxPH:

    def test_fit_and_concordance(self, train_test_split):
        from src.models.cox_ph import CoxPHModel

        train, test = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        c_idx = cox.compute_concordance_index(test)
        # C-index should be above random (0.5) with these features
        assert c_idx > 0.50

    def test_predict_survival_function_shape(self, train_test_split):
        from src.models.cox_ph import CoxPHModel

        train, test = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        sf = cox.predict_survival_function(test)
        # Rows = time points, cols = customers in test
        assert sf.shape[1] == len(test)

    def test_survival_probs_in_0_1(self, train_test_split):
        from src.models.cox_ph import CoxPHModel

        train, test = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        sf = cox.predict_survival_function(test)
        assert (sf.values >= 0).all() and (sf.values <= 1 + 1e-9).all()

    def test_partial_hazard_positive(self, train_test_split):
        from src.models.cox_ph import CoxPHModel

        train, test = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        ph = cox.predict_partial_hazard(test)
        assert (ph.values > 0).all()

    def test_get_hazard_ratios_columns(self, train_test_split):
        from src.models.cox_ph import CoxPHModel

        train, _ = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        hr = cox.get_hazard_ratios()
        for col in ["covariate", "coef", "exp(coef)", "p"]:
            assert col in hr.columns

    def test_model_params_serialisable(self, train_test_split):
        import json

        from src.models.cox_ph import CoxPHModel

        train, _ = train_test_split
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train)
        params = cox.get_model_params()
        # Should be JSON-serialisable
        json.dumps(params)


@pytest.mark.unit
class TestParametric:

    @pytest.mark.parametrize("distribution", ["weibull", "log_normal", "log_logistic"])
    def test_fit_and_aic(self, small_customer_df, distribution):
        from src.models.parametric import ParametricSurvivalModel

        model = ParametricSurvivalModel(distribution=distribution)
        model.fit(small_customer_df)
        assert model.aic < 0 or model.aic > 0  # finite float

    def test_invalid_distribution_raises(self):
        from src.models.parametric import ParametricSurvivalModel

        with pytest.raises(ValueError):
            ParametricSurvivalModel(distribution="invalid_dist")

    def test_concordance_above_random(self, train_test_split):
        from src.models.parametric import ParametricSurvivalModel

        train, test = train_test_split
        model = ParametricSurvivalModel(distribution="weibull")
        model.fit(train)
        c_idx = model.compute_concordance_index(test)
        assert c_idx > 0.50

    def test_fit_all_parametric_returns_three(self, small_customer_df):
        from src.models.parametric import fit_all_parametric

        models = fit_all_parametric(small_customer_df)
        assert set(models.keys()) == {"weibull", "log_normal", "log_logistic"}
        for m in models.values():
            assert m.is_fitted()
