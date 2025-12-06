import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("merged_subjects_with_hrv.csv")


fig1_vars = ['sdnn_ms', 'rmssd_ms', 'lf_ms2', 'lf_instability', 'sample_entropy', 'nn50']
fig2_vars = ['pnn50_pct', 'sdsd_ms', 'sdnn_index', 'tinn_ms', 'dfa_alpha1', 'dfa_alpha2']


def annotate_box(ax, data, cat_col, value_col):
    groups = data.groupby(cat_col)[value_col]
    cats = list(groups.groups.keys())
    for i, cat in enumerate(cats):
        vals = groups.get_group(cat).dropna()
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        if abs(med) < 0.01 and med != 0:
            fmt = "{:.3e}"
        elif abs(med) < 1:
            fmt = "{:.4f}"
        else:
            fmt = "{:.2f}"

        ax.text(i, med, f"Med={fmt.format(med)}",
                ha='center', va='center', color='white',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='navy', ec='white', lw=0.8))
        ax.text(i, q1, f"Q1={fmt.format(q1)}", ha='center', va='top',
                color='black', fontsize=8, alpha=0.8)
        ax.text(i, q3, f"Q3={fmt.format(q3)}", ha='center', va='bottom',
                color='black', fontsize=8, alpha=0.8)

def plot_6vars(df, vars_list, fig_title, filename):
    sns.set(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for idx, col in enumerate(vars_list):
        ax = axes[idx]
        data = df.copy()

   
        if col.lower() == 'tinn_ms':
            ax.set_yscale('log')
            ax.set_ylim(1000, 50000)
            ax.set_ylabel("Tinn (ms, log scale)", fontsize=11)
        sns.boxplot(x="outcome_label", y=col, data=data, ax=ax,
                    palette="Set2", showfliers=True, linewidth=1.1, width=0.6)
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_xlabel("Outcome", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        annotate_box(ax, data, "outcome_label", col)

        ax.tick_params(axis='x', labelrotation=10, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.03)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


plot_6vars(df, fig1_vars, "HRV Metrics – Group 1", "HRV_Boxplot_Group1.png")
plot_6vars(df, fig2_vars, "HRV Metrics – Group 2", "HRV_Boxplot_Group2.png")
