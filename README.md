# Customer Churn Survival Analysis

**Exploratory survival-method demonstration** on IBM's fictional Telco Customer Churn sample. The code maps observed tenure to a duration and a *last-month churn indicator* to a constructed event/censoring variable. Because this is a cross-sectional snapshot, not an inception cohort with observed customer exit times, the resulting Kaplan–Meier/log-rank/Cox outputs **are not validated estimates of population customer lifetimes or retention curves**. The original numeric demonstration is retained, not retroactively certified.

## Business question

What patterns arise if tenure and the last-month churn label are fed into standard survival estimators, and which diagnostics expose the limits of that construction?

## Dataset

IBM Telco Customer Churn sample (fictional company). IBM describes `Churn` as whether the customer left **within the last month**, not whether an exit was witnessed at the row's lifetime tenure. [IBM dataset definition](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn).

- Customers analyzed: **7,032**
- Last-month churn-flag positives: **1,869**
- Overall churn rate: **26.58%**

**Construction, not verified censoring:** the script treats `Churn=No` as right-censored at tenure and `Churn=Yes` as an event at tenure. The source does not establish this as a longitudinal event/censoring history. Independent censoring, entry-cohort comparability, and exact exit-time identification are **not established**.

## Analytical workflow

1. Constructed a tenure-duration / last-month-churn indicator mapping for an educational sensitivity demonstration (not observed time-to-exit records).
2. Computed a **Kaplan–Meier curve under the constructed event/censoring mapping**; it must not be presented as validated population retention.
3. Compared model-implied curves across contract types within the same unverified mapping.
4. Computed a **log-rank statistic** for month-to-month versus two-year contracts under the same assumptions; its p-value is not a valid established longitudinal group-survival inference.
5. Fitted a penalized **Cox proportional-hazards model** and reported only in-sample diagnostics and constructed-risk associations.
6. Reported hazard-ratio coefficients within the constructed model; the snapshot does not identify actual lifetime hazards or causal effects.

## Verified results

| Result | Value |
|---|---:|
| Customers analyzed | **7,032** |
| Last-month churn-flag positives | **1,869** |
| Overall churn rate | **26.58%** |
| Cox apparent/in-sample concordance on fitting customers | **0.826** |
| Nominal Cox p-values below 5% under the constructed model | **6** |

The constructed Kaplan–Meier curve does not cross 50% in the plotted tenure range. This is **not** an established median customer lifetime: the input label captures last-month departure, not recorded exit-at-tenure for a longitudinal cohort.

The log-rank output for contract groups is numerically tiny under the constructed mapping. It does **not** validate differences in actual longitudinal survival without an appropriate cohort, event dates and censoring design.

## Selected hazard-ratio findings

Relative to the model's reference categories and conditional on the included covariates:

- **Two-year contract:** hazard ratio ≈ **0.048**, associated with substantially lower churn hazard.
- **One-year contract:** hazard ratio ≈ **0.175**, also associated with lower churn hazard.
- **Fiber-optic internet service:** hazard ratio ≈ **3.154**, associated with substantially higher churn hazard than the reference internet-service category.

These are **within-model descriptive coefficients**, neither causal effects nor independently validated population hazard ratios; the survival-data construction itself is not identified.

## Business interpretation

- Contract type is associated with the snapshot churn flag, but the plotted survival curves do not establish observed longitudinal retention differences.
- Kaplan–Meier, log-rank and Cox are educational examples here; **correct censoring adjustment cannot be claimed** without a valid event-time/censoring design.
- The in-sample score and nominal coefficients are unsuitable for ranking real customers or estimating intervention effects. A genuine cohort with exit dates, censoring reasons, time-varying tenure and held-out validation is required before decision use.

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

1. Install the packages in `requirements.txt`. Data loading requires access to the linked IBM-hosted CSV; no offline cache or source hash check is implemented.
2. Run `analysis.py` in an IPython/Jupyter environment or adapt the `display()` calls to `print()` for a standard Python shell.
3. Inspect the Kaplan-Meier plots, contract summary, log-rank test, and Cox-model output.

## Limitations

- **Primary identification failure:** IBM defines the response as last-month churn at a cross-sectional snapshot. Mapping it to events occurring at the observed total tenure does not recover event times of an inception cohort. Corresponding survival curves, tests and hazard ratios are exploratory numerical demonstrations only.
- **Not validated out of sample:** the displayed Cox concordance **0.826** is `cph.concordance_index_` measured on fitting customers. There is no held-out customer evaluation or calibrated forecast.
- Cox-model interpretation additionally depends on proportional hazards, valid non-informative censoring and correctly measured survival time; those conditions have not been demonstrated.
- Observational input does not establish causal effects or reliable intervention targets; fictional sample cannot establish external transportability.
- Results were generated by the prior script, not rerun or retuned in this disclosure correction.
