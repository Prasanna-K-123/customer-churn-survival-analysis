"""Historical numerical demonstration, NOT identified customer survival times.

IBM Telco uses a last-month Churn snapshot. Mapping that label to an event at
recorded tenure does not establish cohort event times or independent censoring.
All output curves/statistics are conditional on this invalidated data mapping;
they must not be used as population retention, actual hazard, or policy evidence.
"""
import warnings
warnings.warn("This cross-sectional churn label does not identify lifetime event times or valid censoring; outputs are educational pseudo-survival calculations only.", RuntimeWarning, stacklevel=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(URL)

print("Shape:", df.shape)
display(df.head())

# Construct pseudo-event variables from a snapshot (NOT observed exit times).
df["event"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

surv = df.dropna(subset=["tenure", "event", "MonthlyCharges"]).copy()
surv = surv[surv["tenure"] > 0].copy()

print("Rows in constructed pseudo-survival demo:", len(surv))
print("Last-month churn=Yes labels (NOT verified time-of-exit events):", surv["event"].sum())
print("Churn=No labels (NOT established independently censored rows):", len(surv) - surv["event"].sum())

# Kaplan-Meier estimator evaluated on unverified constructed event/censoring mapping.
kmf = KaplanMeierFitter()
kmf.fit(
    durations=surv["tenure"],
    event_observed=surv["event"],
    label="Constructed KM (not observed retention)",
)

ax = kmf.plot_survival_function(figsize=(8, 5))
ax.set_title("Constructed KM: Cross-sectional Event-Mapping Demo")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Constructed pseudo-survival (NOT validated retention)")
plt.tight_layout()
plt.show()

print("Median of constructed KM output (NOT observed lifetime):", kmf.median_survival_time_)

# Contract pseudo-survival comparison; NOT an identified longitudinal cohort comparison.
plt.figure(figsize=(9, 5))
for contract, group in surv.groupby("Contract"):
    km = KaplanMeierFitter()
    km.fit(group["tenure"], group["event"], label=str(contract))
    km.plot_survival_function()

plt.title("Constructed KM Curves by Contract (NOT validated retention)")
plt.xlabel("Tenure (months)")
plt.ylabel("Constructed pseudo-survival (NOT validated retention)")
plt.tight_layout()
plt.show()

contract_summary = (
    surv.groupby("Contract", as_index=False)
    .agg(
        customers=("customerID", "size"),
        churn_rate=("event", "mean"),
        median_observed_tenure=("tenure", "median"),
    )
)
display(contract_summary)

# Historical log-rank numeric demo; p-value cannot establish actual longitudinal hazards.
g1 = surv[surv["Contract"] == "Month-to-month"]
g2 = surv[surv["Contract"] == "Two year"]

lr = logrank_test(
    g1["tenure"],
    g2["tenure"],
    event_observed_A=g1["event"],
    event_observed_B=g2["event"],
)

print("Log-rank test statistic:", lr.test_statistic)
print("Pseudo-event log-rank p-value (invalid for actual lifetimes):", lr.p_value)

# Cox proportional-hazards model.
cox = surv[[
    "tenure",
    "event",
    "MonthlyCharges",
    "SeniorCitizen",
    "PaperlessBilling",
    "Contract",
    "InternetService",
]].copy()

cox["PaperlessBilling"] = (cox["PaperlessBilling"] == "Yes").astype(int)
cox["SeniorCitizen"] = pd.to_numeric(cox["SeniorCitizen"], errors="coerce").fillna(0)
cox = pd.get_dummies(
    cox,
    columns=["Contract", "InternetService"],
    drop_first=True,
    dtype=int,
)

cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox, duration_col="tenure", event_col="event")

summary = cph.summary[[
    "coef",
    "exp(coef)",
    "se(coef)",
    "p",
    "coef lower 95%",
    "coef upper 95%",
]].copy()
summary = summary.rename(columns={"exp(coef)": "hazard_ratio"})

display(summary.sort_values("p"))
print("Apparent Cox concordance on FITTING rows (not held out):", cph.concordance_index_)

sig = summary[summary["p"] < 0.05].sort_values("p").copy()
print("Nominal pseudo-survival Cox p-values < 5% (NOT validated population significance):")
display(sig)

print("=== HISTORICAL NUMERIC DEMO ONLY — EVENT TIMES / CENSORING UNVERIFIED ===")
print(f"Customers analyzed: {len(surv):,}")
print(f"Last-month churn positive flags: {int(surv['event'].sum()):,}")
print(f"Overall churn rate: {surv['event'].mean()*100:.2f}%")
print(f"Constructed KM median (not customer lifetime): {kmf.median_survival_time_}")
print(f"Pseudo-event log-rank statistic p-value (unvalidated inference): {lr.p_value:.3g}")
print(f"In-sample Cox concordance (unvalidated model): {cph.concordance_index_:.3f}")
print(f"Nominal constructed-model Cox p-values < 0.05: {len(sig)}")
