import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon, t
from statsmodels.stats.multitest import multipletests
import os


os.makedirs("quali_analysis_results", exist_ok=True)

print("Current working directory:", os.getcwd())
# =========================================================
# FILES DIRECTORY
# =========================================================
file_pairs = [
    ("Task1_PC_Qualitative.xlsx", "Task1_MR_Qualitative.xlsx"),
    ("Task2_PC_Qualitative.xlsx", "Task2_MR_Qualitative.xlsx"),
    ("Task3_PC_Qualitative.xlsx", "Task3_MR_Qualitative.xlsx"),
    ("Task4_PC_Qualitative.xlsx", "Task4_MR_Qualitative.xlsx")
]

task_names = ["T1", "T2", "T3", "T4"]

# =========================================================
# SETTINGS
# =========================================================
columns_to_ignore = [
    "Which visualization you use to complete the task? (multiple answers are allowed)",
    "Which one of the plots was the most helpful in completing the task?"
]

# Hardcoded expert indices (0-based, matching iloc)
expert_indices = [0, 3, 7, 8, 9, 11, 13, 14, 15]

# Minimum paired observations required
MIN_N = 5

# =========================================================
# HELPERS
# =========================================================
def cohen_dz(diff):
    """Paired-samples effect size: dz = mean(diff) / sd(diff)."""
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return diff.mean() / sd

def paired_mean_ci(diff, alpha=0.05):
    """95% CI for the paired mean difference."""
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return (mean_diff, mean_diff)
    se = sd / np.sqrt(n)
    tcrit = t.ppf(1 - alpha/2, df=n - 1)
    low = mean_diff - tcrit * se
    high = mean_diff + tcrit * se
    return (low, high)

def iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return q3 - q1

def safe_numeric_pair(series_a, series_b):
    """Convert to numeric, align, drop NaNs."""
    a = pd.to_numeric(series_a, errors="coerce")
    b = pd.to_numeric(series_b, errors="coerce")
    combined = pd.concat([a, b], axis=1).dropna()
    combined.columns = ["2D", "MR"]
    return combined

def choose_test(data_2d, data_mr):
    """
    Use Shapiro on paired differences.
    If normal -> paired t-test
    Else -> Wilcoxon
    """
    diff = data_2d - data_mr

    # Shapiro can fail for constant arrays or certain edge cases
    try:
        if len(diff) >= 3 and diff.nunique() > 1:
            _, p_norm = shapiro(diff)
        else:
            p_norm = 0.0
    except Exception:
        p_norm = 0.0

    if p_norm > 0.05:
        stat, p_val = ttest_rel(data_2d, data_mr, nan_policy="omit")
        test_name = "Paired t-test"
        stat_label = f"t({len(diff) - 1})"
    else:
        try:
            stat, p_val = wilcoxon(data_2d, data_mr, zero_method="wilcox", correction=False)
            test_name = "Wilcoxon signed-rank"
            stat_label = "W"
        except ValueError:
            # Happens when all paired differences are zero
            stat, p_val = 0.0, 1.0
            test_name = "Wilcoxon signed-rank"
            stat_label = "W"

    return diff, p_norm, test_name, stat_label, stat, p_val

def rank_biserial_paired(diff):
    """
    Matched-pairs rank-biserial correlation for Wilcoxon signed-rank data.
    Difference convention: 2D - MR.

    Negative r_rb -> MR tends to be higher
    Positive r_rb -> 2D tends to be higher
    """
    diff = diff.dropna()
    diff = diff[diff != 0]

    if len(diff) == 0:
        return np.nan

    ranks = diff.abs().rank(method="average")

    w_plus = ranks[diff > 0].sum()   # 2D > MR
    w_minus = ranks[diff < 0].sum()  # MR > 2D

    denominator = w_plus + w_minus

    if denominator == 0:
        return np.nan

    return (w_plus - w_minus) / denominator
# =========================================================
# MAIN
# =========================================================
for task_idx, ((file_2d, file_mr), task_name) in enumerate(zip(file_pairs, task_names), start=1):
    try:
        df_2d = pd.read_excel(file_2d)
        df_mr = pd.read_excel(file_mr)
    except FileNotFoundError:
        print(f"Skipping {task_name}: file not found.")
        continue

    # Clean column names
    df_2d.columns = df_2d.columns.str.strip()
    df_mr.columns = df_mr.columns.str.strip()

    # Use only shared analyzable columns
    shared_columns = [c for c in df_2d.columns if c in df_mr.columns]
    columns_to_analyze = [
        col for col in shared_columns[2:-3]
        if col not in columns_to_ignore
    ]

    groups = {
        "All_Participants": list(range(len(df_2d))),
        "Experts_Only": expert_indices
    }

    for group_name, indices in groups.items():
        # Keep only valid indices that exist in both dataframes
        valid_indices = [i for i in indices if i < len(df_2d) and i < len(df_mr)]

        sub_2d = df_2d.iloc[valid_indices].copy()
        sub_mr = df_mr.iloc[valid_indices].copy()

        task_results = []

        for column in columns_to_analyze:
            combined = safe_numeric_pair(sub_2d[column], sub_mr[column])

            if len(combined) < MIN_N:
                continue

            data_2d = combined["2D"]
            data_mr = combined["MR"]

            diff, p_norm, test_name, stat_label, stat, p_val = choose_test(data_2d, data_mr)

            mean_2d = data_2d.mean()
            mean_mr = data_mr.mean()
            mean_diff = diff.mean()

            median_2d = data_2d.median()
            median_mr = data_mr.median()

            iqr_2d = iqr(data_2d)
            iqr_mr = iqr(data_mr)

            #dz = cohen_dz(diff)
            if test_name == "Paired t-test":
                effect_type = "Cohen's dz"
                effect_value = cohen_dz(diff)
                ci_low, ci_high = paired_mean_ci(diff)


            else:
                effect_type = "Rank-biserial r"
                effect_value = rank_biserial_paired(diff)
                ci_low, ci_high = np.nan, np.nan


            #ci_low, ci_high = paired_mean_ci(diff)

            mr_higher = int((data_mr > data_2d).sum())
            d2_higher = int((data_2d > data_mr).sum())
            ties = int((data_2d == data_mr).sum())

            task_results.append({
                "Task": task_name,
                "Column": column,
                "N": len(combined),

                "Test": test_name,
                "Stat_Type": stat_label,
                "Stat_Value": round(stat, 4),
                "Shapiro_p_diff": round(p_norm, 4),

                "p_raw": p_val,

                "Mean_2D": round(mean_2d, 3),
                "Mean_MR": round(mean_mr, 3),
                "Mean_Diff_2D_minus_MR": round(mean_diff, 3),

                "Median_2D": round(median_2d, 3),
                "Median_MR": round(median_mr, 3),
                "IQR_2D": round(iqr_2d, 3),
                "IQR_MR": round(iqr_mr, 3),

                "Cohens_dz": round(effect_value, 3)
                    if effect_type == "Cohen's dz" else np.nan,

                "Rank_Biserial_r": round(effect_value, 3)
                    if effect_type == "Rank-biserial r" else np.nan,

                "Effect_Size_Type": effect_type,
                "Effect_Size": round(effect_value, 3)
                    if not np.isnan(effect_value) else np.nan,

                "CI95_low_diff": round(ci_low, 3),
                "CI95_high_diff": round(ci_high, 3),

                "MR_higher_count": mr_higher,
                "2D_higher_count": d2_higher,
                "Ties": ties
            })

        if not task_results:
            print(f"No valid results for {task_name} - {group_name}")
            continue

        results_df = pd.DataFrame(task_results)

        # Holm correction within each task/group result file
        _, p_adj, _, _ = multipletests(results_df["p_raw"], method="holm")
        results_df["p_corrected"] = p_adj
        results_df["Significant"] = results_df["p_corrected"] < 0.05

        # Sort for easier inspection
        results_df = results_df.sort_values(by=["p_corrected", "p_raw", "Column"]).reset_index(drop=True)

        output_file = f"quali_analysis_results/{task_name}_{group_name}_Rigorous.xlsx"
        results_df.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")

print("All reports generated.")