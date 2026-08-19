# Customer Churn Survival Analysis

Time-to-event analysis of customer churn using the **IBM Telco Customer Churn** sample. Instead of predicting only whether a customer churns, this project models **retention over time** and identifies factors associated with the timing of churn.

## Business question

How does customer retention evolve over tenure, which contract groups have meaningfully different survival patterns, and which customer characteristics are associated with higher or lower churn hazard?

## Dataset

IBM Telco Customer Churn sample.

- Customers analyzed: **7,032**
- Observed churn events: **1,869**
- Overall churn rate: **26.58%**

Customers who had not churned by the end of their observed tenure are treated as right-censored observations.

## Analytical workflow

1. Prepared tenure as the time variable and churn as the event indicator.
2. Estimated the overall retention curve using **Kaplan-Meier** survival analysis.
3. Compared retention curves across contract types.
4. Used a **log-rank test** to compare month-to-month and two-year contracts.
5. Fitted a penalized **Cox proportional-hazards model** using a compact set of interpretable predictors.
6. Interpreted hazard ratios as conditional associations with churn hazard rather than causal effects.

## Verified results

| Result | Value |
|---|---:|
| Customers analyzed | **7,032** |
| Churn events | **1,869** |
| Overall churn rate | **26.58%** |
| Cox concordance index | **0.826** |
| Significant Cox predictors at 5% | **6** |

The Kaplan-Meier median survival time was **not reached within the observed horizon**, meaning the estimated survival probability remained above 50% throughout the available follow-up period.

The month-to-month versus two-year contract survival curves were extremely different statistically; the computed p-value was so small that it displayed as numerical zero in the notebook output.

## Selected hazard-ratio findings

Relative to the model's reference categories and conditional on the included covariates:

- **Two-year contract:** hazard ratio ≈ **0.048**, associated with substantially lower churn hazard.
- **One-year contract:** hazard ratio ≈ **0.175**, also associated with lower churn hazard.
- **Fiber-optic internet service:** hazard ratio ≈ **3.154**, associated with substantially higher churn hazard than the reference internet-service category.

These are **associations, not causal effects**.

## Business interpretation

- Contract structure is strongly associated with retention differences in this sample.
- Survival analysis adds information that a simple churn classification model does not: it models *when* churn occurs and handles censored customers correctly.
- High-risk groups identified by the Cox model are useful for prioritizing further investigation or retention experiments, but interventions should be tested before assuming they will reduce churn.

## Technology stack

- Python
- Pandas, NumPy, Matplotlib
- `lifelines`
- Kaplan-Meier estimation
- Log-rank testing
- Cox proportional-hazards modeling
- Hazard ratios and concordance index

## Repository structure

```text
customer-churn-survival-analysis/
├── README.md
├── analysis.py
└── requirements.txt
```

## Reproducing the analysis

1. Install the packages in `requirements.txt`.
2. Run `analysis.py` in an IPython/Jupyter environment or adapt the `display()` calls to `print()` for a standard Python shell.
3. Inspect the Kaplan-Meier plots, contract summary, log-rank test, and Cox-model output.

## Limitations

- Observational data does not establish that contract type or service choice causes churn.
- Cox-model interpretation depends on the proportional-hazards framework and the predictors included.
- The dataset is a sample and may not represent another telecom customer population or future period.
