import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

df = pd.read_csv("merged_subjects_with_hrv.csv") 

if df["outcome_label"].dtype == "object":
    print("🔹 Detected text outcome labels, converting to numeric (0/1)...")
    df["outcome_label"] = df["outcome_label"].str.lower().map({
        "survivor": 0,
        "nonsurvivor": 1,
        "scd": 1,
        "alive": 0,
        "dead": 1,
        "0": 0,
        "1": 1
    })

if df["outcome_label"].isna().any():
    print(" (warning) Dropping rows with unrecognized or missing outcome labels.")
    df = df.dropna(subset=["outcome_label"])

df["outcome_label"] = df["outcome_label"].astype(int)
print(" (correct) Outcome variable standardized to binary (0/1).")

hrv_vars = [
    "sdnn_ms", "rmssd_ms", "sdnn_index", "lf_ms2", "pnn50_pct", "nn50",
    "tinn_ms", "lf_instability", "sample_entropy", "dfa_alpha1", "dfa_alpha2", "sdsd_ms"
]

meds = [c for c in df.columns if "blocker" in c.lower() or "med" in c.lower() or
        "inhibitor" in c.lower() or "diuretic" in c.lower() or
        "statin" in c.lower() or "nitro" in c.lower() or "digoxin" in c.lower()]

print(f"(correct)Detected {len(meds)} medication covariates:")
print(meds)

df[hrv_vars + meds] = df[hrv_vars + meds].apply(pd.to_numeric, errors="coerce")
print("(correct) All HRV + medication variables cleaned and numeric.")
print(f" Running logistic regressions for {len(hrv_vars)} HRV variables...")

results = []

for var in hrv_vars:
    try:
        X = df[[var] + meds].copy()
        y = df["outcome_label"]
        data = pd.concat([X, y], axis=1).dropna()

        X = sm.add_constant(data[[var] + meds])
        y = data["outcome_label"]

        if len(y.unique()) < 2:
            print(f"(warning) Skipping {var}: only one outcome class present.")
            continue

        model = sm.Logit(y, X).fit(disp=False)
        params = model.params[var]
        conf = model.conf_int().loc[var]
        OR = np.exp(params)
        CI_lower, CI_upper = np.exp(conf)
        results.append({"Metric": var, "OR": OR, "CI_lower": CI_lower, "CI_upper": CI_upper})
    except Exception as e:
        print(f"(warning) Skipping {var} due to error: {e}")
        continue

if len(results) == 0:
    print("(wrong) No models ran successfully. Check your HRV data formatting.")
    exit()

results_df = pd.DataFrame(results).sort_values("OR", ascending=False)
print("\n(correct) Logistic regression complete. Results preview:")
print(results_df.head())

plt.figure(figsize=(8, 6))
plt.errorbar(results_df["OR"], results_df["Metric"],
             xerr=[results_df["OR"] - results_df["CI_lower"],
                   results_df["CI_upper"] - results_df["OR"]],
             fmt="o", color="blue", ecolor="gray", capsize=4)
plt.axvline(x=1, color="red", linestyle="--", label="Null (OR=1)")
plt.xscale("log")
plt.xlabel("Adjusted Odds Ratio (log scale)")
plt.title("Medication-Adjusted HRV Associations with SCD")
plt.legend()
plt.tight_layout()
plt.show()

print("\n🔹 Computing AUC values for HRV variables...")

auc_list = []
for var in hrv_vars:
    try:
        X = df[[var] + meds].copy()
        y = df["outcome_label"]
        data = pd.concat([X, y], axis=1).dropna()
        X = data[[var] + meds]
        y = data["outcome_label"]

        if len(y.unique()) < 2:
            print(f"(warning) Skipping {var} for AUC: Only one class present.")
            continue

        model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
        y_pred = model.predict(sm.add_constant(X))

        if np.isnan(y_pred).any():
            print(f"(wanring) Skipping {var} for AUC: NaN in predictions.")
            continue

        auc = roc_auc_score(y, y_pred)
        auc_list.append({"Metric": var, "AUC": auc})
    except Exception as e:
        print(f"(warning) Skipping {var} for AUC: {e}")
        continue

if len(auc_list) == 0:
    print("(wrong) No valid AUC results — check missing values or data balance.")
else:
    auc_df = pd.DataFrame(auc_list).sort_values("AUC", ascending=False)
    print("\n(correct) AUC Results:")
    print(auc_df)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="AUC", y="Metric", data=auc_df, palette="viridis")
    plt.title("AUC of Medication-Adjusted HRV Models")
    plt.xlabel("AUC")
    plt.tight_layout()
    plt.show()
