
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import os

OUTPUT_DIR = "hrv_scd_nyha_stratified_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv("merged_subjects_with_hrv.csv", low_memory=False)


print(" Columns in merged_subjects_with_hrv.csv:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2}. '{col}'")

post_mi_col = None
for col in df.columns:
    if "myocardial infarction" in col.lower() or "post mi" in col.lower() or "prior mi" in col.lower():
        post_mi_col = col
        break

if post_mi_col is None:
    raise ValueError(" No 'Prior Myocardial Infarction' column found. Check your CSV headers.")

print(f" Using Post-MI column: '{post_mi_col}'")


nyha_col = None
for col in df.columns:
    if "nyha" in col.lower():
        nyha_col = col
        break

if nyha_col is None:
    raise ValueError(" No NYHA column found. Ensure 'nyha_class' or similar exists.")

print(f" Using NYHA column: '{nyha_col}'")


valid_outcomes = ["SCD", "PumpFailure", "Survivor", "NonCardiacDeath"]
df = df[df["outcome_label"].isin(valid_outcomes)].copy()
df["is_scd"] = (df["outcome_label"] == "SCD").astype(int)

hrv_metrics = ['sdnn_ms', 'rmssd_ms', 'lf_ms2', 'lf_instability', 'sample_entropy','dfa_alpha1','dfa_alpha2']

missing_hrv = [m for m in hrv_metrics if m not in df.columns]
if missing_hrv:
    raise ValueError(f"Missing HRV columns: {missing_hrv}")

df[post_mi_col] = pd.to_numeric(df[post_mi_col], errors='coerce')
df = df.dropna(subset=[post_mi_col, nyha_col])

df_postmi = df[df[post_mi_col] == 1].copy()

df_postmi[nyha_col] = pd.to_numeric(df_postmi[nyha_col], errors='coerce')
df_postmi = df_postmi.dropna(subset=[nyha_col])

df_postmi = df_postmi[df_postmi[nyha_col].isin([2, 3])].copy()

print(f"\n Total Post-MI patients: {len(df)} → Post-MI + NYHA II/III: {len(df_postmi)}")

if len(df_postmi) == 0:
    raise ValueError("No patients meet criteria: Post-MI + NYHA II/III")


subgroups = {
    "NYHA_II": df_postmi[df_postmi[nyha_col] == 2],
    "NYHA_III": df_postmi[df_postmi[nyha_col] == 3]
}

def analyze_subgroup(sub_df, group_name, output_dir):
    if len(sub_df) == 0:
        print(f" Skipping {group_name}: no data")
        return

    n_scd = sub_df['is_scd'].sum()
    n_total = len(sub_df)
    print(f"  • {group_name}: N={n_total}, SCD={n_scd}")

    plt.figure(figsize=(15, 12)) 
    for i, metric in enumerate(hrv_metrics, 1):
        plt.subplot(3, 3, i)  
        sns.boxplot(data=sub_df, x="outcome_label", y=metric, order=valid_outcomes)
        plt.title(f"{metric.replace('_', ' ').title()} ({group_name})")
        plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplots_{group_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    scd_stats = sub_df[sub_df['is_scd'] == 1][hrv_metrics].describe()
    non_scd_stats = sub_df[sub_df['is_scd'] == 0][hrv_metrics].describe()
    scd_stats.to_csv(os.path.join(output_dir, f"{group_name}_scd_hrv_descriptives.csv"))
    non_scd_stats.to_csv(os.path.join(output_dir, f"{group_name}_non_scd_hrv_descriptives.csv"))

    corr = sub_df[hrv_metrics].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"HRV Correlations ({group_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"correlation_{group_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    if n_scd >= 5 and (n_total - n_scd) >= 10:
        X = sub_df[['sdnn_ms', 'lf_ms2']].dropna()
        y = sub_df.loc[X.index, 'is_scd']
        if len(y) >= 10 and y.sum() >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_scaled, y)
            probs = clf.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, probs)

            fpr, tpr, _ = roc_curve(y, probs)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"Static HRV AUC = {auc:.3f}", linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {group_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"roc_{group_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()


            with open(os.path.join(output_dir, f"auc_{group_name}.txt"), "w") as f:
                f.write(f"AUC (SDNN + LF): {auc:.3f}\nTotal: {len(y)}\nSCD: {int(y.sum())}")
        else:
            print(f"    → Not enough data for ROC in {group_name}")
    else:
        print(f"    → Too few SCD cases ({n_scd}) for ROC in {group_name}")

summary_rows = []
for name, data in subgroups.items():
    subgroup_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(subgroup_dir, exist_ok=True)
    analyze_subgroup(data, name, subgroup_dir)

    counts = data['outcome_label'].value_counts()
    row = {
        'Group': name,
        'Total': len(data),
        'SCD': int(counts.get('SCD', 0)),
        'PumpFailure': int(counts.get('PumpFailure', 0)),
        'Survivor': int(counts.get('Survivor', 0)),
        'NonCardiacDeath': int(counts.get('NonCardiacDeath', 0))
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "nyha_postmi_outcome_summary.csv"), index=False)

print("\n" + "="*60)
print(" NYHA-Stratified Analysis Complete!")
print(f" Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\n Summary:")
print(summary_df.to_string(index=False))
print("="*60)
