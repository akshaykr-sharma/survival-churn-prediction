# Cox Proportional Hazards Model — Model Explainer

## What It Is

The Cox Proportional Hazards (Cox PH) model is a **semi-parametric** regression model that estimates the hazard function — the instantaneous rate of churn at time t given a set of covariates.

It is the **primary production model** for individual-level survival scoring.

## Mathematical Foundation

$$h(t \mid \mathbf{x}) = h_0(t) \cdot \exp(\boldsymbol{\beta}^\top \mathbf{x})$$

Where:
- $h(t \mid \mathbf{x})$ = individual hazard (rate of churn at time t)
- $h_0(t)$ = **baseline hazard** (estimated non-parametrically via Breslow estimator) — shared across all customers
- $\exp(\boldsymbol{\beta}^\top \mathbf{x})$ = multiplicative adjustment per individual based on their features
- $\boldsymbol{\beta}$ = vector of coefficients estimated by maximising the partial likelihood

## Why "Semi-parametric"?

- The baseline hazard $h_0(t)$ is **not** assumed to follow any distribution (non-parametric part)
- The covariate effect $\exp(\boldsymbol{\beta}^\top \mathbf{x})$ is parametric (linear on log scale)

This gives flexibility: no need to correctly specify the shape of the baseline hazard.

## The Proportional Hazards Assumption

The model assumes that the **hazard ratio between any two customers is constant over time**:

$$\frac{h(t \mid \mathbf{x}_1)}{h(t \mid \mathbf{x}_2)} = \exp\big(\boldsymbol{\beta}^\top (\mathbf{x}_1 - \mathbf{x}_2)\big) = \text{constant}$$

This means if a gamer customer has 30% lower hazard than a student at day 90, they still have 30% lower hazard at day 365. We test this assumption using **Schoenfeld residuals**.

## Hazard Ratios (HR)

For covariate $x_j$, the hazard ratio is $\text{HR}_j = \exp(\beta_j)$.

| HR | Interpretation |
|---|---|
| HR = 1.0 | No effect on churn risk |
| HR = 1.5 | 50% higher churn rate per unit increase |
| HR = 0.7 | 30% lower churn rate per unit increase (protective) |

### Expected HRs for XYZ India

| Covariate | Expected HR | Direction | Business meaning |
|---|---|---|---|
| `days_since_last_transaction` | > 1.0 | Increases risk | Silence → approaching churn |
| `amc_active` | < 1.0 | Decreases risk | AMC customers are retained longer |
| `avg_weekly_sessions_90d` | < 1.0 | Decreases risk | Web-engaged customers churn less |
| `escalation_rate` | > 1.0 | Increases risk | Escalations signal dissatisfaction |
| `session_frequency_trend_90d` | < 1.0 | Decreases risk | Growing engagement = lower hazard |
| `warranty_remaining_days` | < 1.0 | Decreases risk | Active warranty = reason to stay |

## Individual Survival Curves

From Cox PH, the individual survival function is:

$$S(t \mid \mathbf{x}) = S_0(t)^{\exp(\boldsymbol{\beta}^\top \mathbf{x})}$$

where $S_0(t) = \exp\!\left(-\int_0^t h_0(u)\,du\right)$ is the baseline survival estimated via Breslow.

This gives a personalised S(t) curve per customer.

## Time-Varying Covariates (TVC)

Standard Cox assumes covariates are measured once (at baseline). But features like `monthly_sessions_trend` and `support_ticket_count_30d` change over time.

TVC is handled via the **counting process** (start, stop] formulation:
- One row per customer per time interval
- Covariates can take different values in each interval
- The model conditions on the covariate value at each observed event time

```
customer_id  start  stop  is_churned  sessions_per_week  tickets_30d
CUST0000001    0     90       0             3.2              0
CUST0000001   90    180       0             2.1              1
CUST0000001  180    270       1             0.4              3   ← churn
```

## Penalisation

The model uses an **L2 (ridge) penalty** (`penalizer=0.1`) to prevent overfitting with many covariates. This shrinks coefficients toward zero but does not zero them out (unlike L1/Lasso).

The penalised partial log-likelihood:

$$\ell_p(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \frac{\lambda}{2} \|\boldsymbol{\beta}\|^2$$

## Model Evaluation

### C-index (Harrell's Concordance Index)

$$C = P\left(\hat{T}_i < \hat{T}_j \mid T_i < T_j, \delta_i = 1\right)$$

Measures discrimination: do customers with higher predicted risk actually churn sooner?
- C = 0.5 → random
- C = 1.0 → perfect discrimination
- Target: **C ≥ 0.70**

### Brier Score at 90d

Measures calibration: are the predicted survival probabilities well-calibrated?
- BS = 0 → perfect predictions
- BS = 0.25 → uninformative (predicting 0.5 for everyone)
- Target: **BS ≤ 0.20**

## PH Assumption Testing

We test the proportional hazards assumption using the Schoenfeld residual method. Under PH, residuals should be uncorrelated with time.

```python
from lifelines.statistics import proportional_hazard_test
result = proportional_hazard_test(cox_fitter, train_df, time_transform="rank")
```

If a covariate violates PH (p < 0.05), options:
1. Add a time-interaction term (e.g., `x * log(t)`)
2. Stratify the model by that covariate
3. Use it as a time-varying covariate

## Implementation

```python
from src.models.cox_ph import CoxPHModel

cox = CoxPHModel(penalizer=0.1, l1_ratio=0.0)
cox.fit(train_df, duration_col="duration_days", event_col="is_churned")

# C-index on test set
c_index = cox.compute_concordance_index(test_df)

# Individual survival probabilities
sf = cox.predict_survival_function(test_df)        # S(t) matrix
median = cox.predict_median_survival(test_df)       # median days
risk = cox.predict_partial_hazard(test_df)          # relative risk score

# Horizon-specific predictions for scoring
horizons = cox.survival_at_horizons(test_df, horizons=[30, 60, 90, 180])

# PH test
ph_result = cox.test_proportional_hazard_assumption(test_df)
```
