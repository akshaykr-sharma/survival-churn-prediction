# Kaplan-Meier Estimator — Model Explainer

## What It Is

The Kaplan-Meier (KM) estimator is a **non-parametric** method that estimates the survival function S(t) — the probability that a customer has not churned by time t — directly from observed data, without assuming any particular distributional form.

## Mathematical Foundation

$$S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$

Where:
- $t_i$ = each distinct event time in the data
- $d_i$ = number of churn events at time $t_i$
- $n_i$ = number of customers still active ("at risk") just before $t_i$

The product form means S(t) drops only at observed churn events, and the estimate is a **step function** rather than a smooth curve.

## Handling Censoring

A customer is **right-censored** if they haven't churned by the observation end date. KM handles this correctly: censored customers contribute to the at-risk count up until their last observation, then drop out. Critically, they are **not** treated as churned — this is the key advantage over naive retention rate calculations.

## What KM Is Used For in This Project

| Purpose | Details |
|---|---|
| Population-level baseline | Overall S(t) for all 55K customers — establishes expected churn rates at 90d, 180d, 365d |
| Segment curves | Separate KM curves per segment (student, gamer, professional, SMB, creative) |
| Product-tier curves | Separate curves per product_category |
| City-tier curves | Tier-1 vs Tier-2 vs Tier-3 survival differences |
| Log-rank tests | Statistical tests comparing curves between groups |

## Log-Rank Test

The log-rank test asks: "Are the survival curves for these groups statistically different?"

$$\chi^2 = \sum_j \frac{(O_j - E_j)^2}{E_j}$$

- $O_j$ = observed churn events in group j at each time point
- $E_j$ = expected events under the null (same hazard in all groups)
- p < 0.05 → groups have significantly different churn rates

**Business interpretation**: If the log-rank test shows p < 0.05 between Gaming PC and Budget PC customers, it means the timing of churn differs and they should be modelled/managed separately.

## KM Stratification Findings (expected for India PC market)

| Segment | Expected median survival | Interpretation |
|---|---|---|
| Professional | ~800 days | Low churn — repeat service visits, AMC renewals |
| Gamer | ~700 days | Engaged community, upgrade cycle drives retention |
| Creative | ~720 days | High spend, low churn |
| SMB | ~650 days | Moderate churn — budget-driven decisions |
| Student | ~550 days | Highest churn — device replaced when graduating |

## Limitations

1. **No covariate adjustment** — KM curves are population-level; they cannot predict individual churn risk given a customer's features.
2. **Step function** — not smooth; can't extrapolate beyond the observed follow-up period.
3. **Assumes independent censoring** — customers who are censored must not be systematically different from those who stayed (Missing Completely At Random assumption).

For individual-level predictions, use the **Cox PH** or **Parametric** models.

## How to Read the KM Plot

```
S(t)
1.0 ──────┐
          │  Confidence bands (95% CI)
0.8       └─────┐
                │
0.6             └──────────────┐
                               │  Step drops at each churn event
0.4                            └───────────────┐
                                               │
0.2                                            └──────
    ─────────────────────────────────────────────────► t (days)
    0      90     180     270     365
```

- The curve starts at 1.0 (no one has churned yet)
- Steps down at each observed churn event
- Confidence bands widen as the number at risk decreases
- Vertical dashes on the curve mark censored observations

## Implementation

```python
from src.models.kaplan_meier import KaplanMeierModel

km = KaplanMeierModel(alpha=0.05)
km.fit(train_df, duration_col="duration_days", event_col="is_churned")

# Population S(t)
survival_func = km.predict_survival_function(train_df)

# S(90d)
prob_90d = km.survival_prob_at(90)  # e.g. 0.78 → 78% survive past 90 days

# Stratified by segment
km.fit_stratified(train_df, stratify_col="segment")

# Log-rank test between city tiers
results = km.log_rank_test(train_df, group_col="city_tier")
```
