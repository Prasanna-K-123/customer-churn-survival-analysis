import warnings
warnings.filterwarnings("ignore")

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

# Prepare survival variables.
df["event"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

surv = df.dropna(subset=["tenure", "event", "MonthlyCharges"]).copy()
surv = surv[surv["tenure"] > 0].copy()

print("Rows for survival analysis:", len(surv))
print("Observed churn events:", surv["event"].sum())
print("Censored observations:", len(surv) - surv["event"].sum())

# Overall Kaplan-Meier retention curve.
kmf = KaplanMeierFitter()
kmf.fit(
    durations=surv["tenure"],
    event_observed=surv["event"],
    label="Overall retention",
)

ax = kmf.plot_survival_function(figsize=(8, 5))
ax.set_title("Kaplan-Meier Estimated Customer Retention")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Estimated probability of remaining active")
plt.tight_layout()
plt.show()

print("Estimated median survival time:", kmf.median_survival_time_)

# Contract-level survival comparison.
plt.figure(figsize=(9, 5))
for contract, group in surv.groupby("Contract"):
    km = KaplanMeierFitter()
    km.fit(group["tenure"], group["event"], label=str(contract))
    km.plot_survival_function()

plt.title("Retention Curves by Contract Type")
plt.xlabel("Tenure (months)")
plt.ylabel("Estimated probability of remaining active")
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

# Log-rank test: month-to-month versus two-year contracts.
g1 = surv[surv["Contract"] == "Month-to-month"]
g2 = surv[surv["Contract"] == "Two year"]

lr = logrank_test(
    g1["tenure"],
    g2["tenure"],
    event_observed_A=g1["event"],
    event_observed_B=g2["event"],
)

print("Log-rank test statistic:", lr.test_statistic)
print("Log-rank p-value:", lr.p_value)

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
print("Concordance index:", cph.concordance_index_)

sig = summary[summary["p"] < 0.05].sort_values("p").copy()
print("Statistically significant predictors at 5%:")
display(sig)

print("=== VERIFIED PROJECT FACTS ===")
print(f"Customers analyzed: {len(surv):,}")
print(f"Observed churn events: {int(surv['event'].sum()):,}")
print(f"Overall churn rate: {surv['event'].mean()*100:.2f}%")
print(f"Kaplan-Meier median survival time: {kmf.median_survival_time_}")
print(f"Month-to-month vs two-year log-rank p-value: {lr.p_value:.3g}")
print(f"Cox concordance index: {cph.concordance_index_:.3f}")
print(f"Significant Cox predictors at 5%: {len(sig)}")
