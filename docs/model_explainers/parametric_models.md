# Parametric Survival Models — Model Explainer

## Overview

Parametric models assume the survival time T follows a specific probability distribution. Compared to Cox PH:

| Property | Cox PH | Parametric |
|---|---|---|
| Baseline hazard | Non-parametric (flexible) | Fixed distributional form |
| Data efficiency | Moderate | Higher (fewer parameters) |
| Extrapolation | Limited to observed follow-up | Can extrapolate beyond |
| Smooth curves | No (step function baseline) | Yes |
| AIC/BIC comparison | No | Yes — enables distributional model selection |

We fit **three distributions** and select the best via AIC.

---

## Accelerated Failure Time (AFT) Framework

All three models are estimated as **AFT (Accelerated Failure Time)** models:

$$\log(T_i) = \boldsymbol{\beta}^\top \mathbf{x}_i + \sigma \varepsilon_i$$

Where:
- $T_i$ = survival time for customer i
- $\mathbf{x}_i$ = feature vector
- $\sigma$ = scale parameter (spread of the distribution)
- $\varepsilon_i$ = random error following a distribution-specific form

The covariate vector $\mathbf{x}$ **accelerates or decelerates** time to event. A positive coefficient means that feature extends survival time (lower churn risk).

**Relationship to Cox PH:** For the Weibull distribution, the AFT and PH formulations are equivalent (just different parameterisations). For Log-Normal and Log-Logistic, AFT is the natural form.

---

## 1. Weibull Distribution

### Hazard Function

$$h(t) = \frac{\gamma}{\lambda} \left(\frac{t}{\lambda}\right)^{\gamma - 1}$$

Where:
- $\gamma$ (shape) controls the hazard trend over time
- $\lambda$ (scale) is the characteristic life

### Shape Parameter Interpretation

| Shape (γ) | Hazard Pattern | Business meaning for PC churn |
|---|---|---|
| γ < 1 | Decreasing hazard | Early-life failures (infant mortality) — customers who churn quickly |
| γ = 1 | Constant hazard | Exponential — memoryless churn |
| γ > 1 | Increasing hazard | Wear-out — churn increases as device ages; **most common for PCs** |

**Expected for XYZ India:** γ ≈ 1.2–1.6 (increasing hazard with tenure — hardware ages, competition increases).

### Survival Function

$$S(t \mid \mathbf{x}) = \exp\!\left[-\left(\frac{t}{\lambda \cdot \exp(\boldsymbol{\beta}^\top \mathbf{x})}\right)^\gamma\right]$$

---

## 2. Log-Normal Distribution

### Hazard Function

$$h(t) = \frac{\phi(\sigma^{-1}\log(t/\mu))}{t \Phi(-\sigma^{-1}\log(t/\mu))}$$

Where $\phi$ and $\Phi$ are the PDF and CDF of the standard normal.

### Hazard Pattern

The log-normal hazard **rises then falls** — it peaks at an intermediate time and then decreases. This models:
- Customers who churn after an initial honeymoon period
- Returns/exchanges resolved early, followed by a period of higher churn risk mid-lifecycle, then declining risk for loyal long-tenured customers

### When to Prefer Log-Normal

- If exploratory analysis (KM curves + smoothed hazard) shows a non-monotone hazard
- If your customer lifecycle has a "danger zone" (e.g., 3–6 months post-purchase when novelty wears off)

---

## 3. Log-Logistic Distribution

### Hazard Function

$$h(t) = \frac{(\gamma/\lambda)(t/\lambda)^{\gamma-1}}{1 + (t/\lambda)^\gamma}$$

### Characteristics

- Similar to log-normal but with **heavier tails** (more probability in extreme survival times)
- Hazard also rises then falls (for γ > 1)
- Often preferred over log-normal when you have a small number of very late churners

### Practical Difference vs Log-Normal

| Feature | Log-Normal | Log-Logistic |
|---|---|---|
| Tails | Lighter | Heavier |
| Hazard peak | Later | Earlier |
| Computation | Slightly slower | Faster (closed-form) |

---

## Model Selection

We compare all three models using:

1. **AIC (Akaike Information Criterion):** $\text{AIC} = 2k - 2\ell$ — penalises complexity
2. **BIC (Bayesian Information Criterion):** $\text{BIC} = k\log(n) - 2\ell$ — stronger penalty for larger datasets
3. **C-index** on held-out test set

**Rule:** Select the model with lowest AIC among the three as the parametric champion.
Then compare the parametric champion against Cox PH on C-index. The final production model is whichever achieves the higher C-index.

```python
from src.models.parametric import fit_all_parametric

models = fit_all_parametric(train_df)
# Returns: {"weibull": ..., "log_normal": ..., "log_logistic": ...}

# Compare
for name, m in models.items():
    print(f"{name}: AIC={m.aic:.1f}, BIC={m.bic:.1f}, C={m.compute_concordance_index(test_df):.3f}")
```

---

## Feature Interpretation in AFT

In the AFT formulation, the coefficient $\beta_j$ has this interpretation:

> "A one-unit increase in feature $x_j$ multiplies the expected survival time by $\exp(\beta_j)$."

- $\exp(\beta_j) > 1$ → covariate extends survival (protective)
- $\exp(\beta_j) < 1$ → covariate shortens survival (risk factor)

Example: If `amc_active` has $\beta = 0.25$, then $\exp(0.25) \approx 1.28$ — customers with active AMC are expected to survive **28% longer** before churning.

---

## Implementation

```python
from src.models.parametric import ParametricSurvivalModel

# Fit a single model
weibull = ParametricSurvivalModel(distribution="weibull")
weibull.fit(train_df, duration_col="duration_days", event_col="is_churned")

# Key metrics
print(f"AIC: {weibull.aic:.2f}")
print(f"BIC: {weibull.bic:.2f}")
print(f"C-index: {weibull.compute_concordance_index(test_df):.4f}")

# Individual predictions
sf = weibull.predict_survival_function(test_df)      # S(t) matrix
median = weibull.predict_median_survival(test_df)    # median days to churn
expected = weibull.predict_expectation(test_df)      # E[T] per customer

# Plot sample individual curves
weibull.plot_survival_curves_sample(test_df, n_samples=100)
```
