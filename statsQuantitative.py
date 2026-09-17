import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon, t as t_dist, rankdata
from statsmodels.stats.multitest import multipletests
import os
# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "Tasks_Performance_Data.xlsx"
OUTPUT_PREFIX = "Quantitative"

os.makedirs("quanti_analysis_results", exist_ok=True)

# Hardcoded expert rows using 0-based pandas indexing
expert_indices = [0, 3, 7, 8, 9, 11, 13, 14, 15]

MIN_N = 5
ALPHA = 0.05

metrics = ["Effectiveness", "Time", "Efficiency"]
tasks = ["T1", "T2", "T3", "T4"]

# =========================================================
# HELPERS
# =========================================================

def flatten_tworow_columns(df):
    """
    If the file has 2 header rows like:
        row 1:   PC, MR, PC, MR, ...
        row 2:   Effectiveness_T1, Effectiveness_T1, Time_T1, Time_T1, ...
    then create unique columns like:
        PC_Effectiveness_T1, MR_Effectiveness_T1, ...
    """
    new_cols = []
    seen = {}

    for col in df.columns:
        if isinstance(col, tuple):
            top = str(col[0]).strip()
            bottom = str(col[1]).strip()
            name = f"{top}_{bottom}" if top and top != "nan" else bottom
        else:
            name = str(col).strip()

        # Make sure names are unique
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        new_cols.append(name)

    df.columns = new_cols
    return df


def read_combined_file(path):
    """
    Try reading with a 2-row header first.
    If that fails to produce usable columns, fall back to 1-row header.
    """
    # Try 2-row header
    try:
        df = pd.read_excel(path, header=[0, 1])
        df = flatten_tworow_columns(df)

        # If ParticipantID appears correctly, good
        if any("ParticipantID" in c for c in df.columns):
            return df
    except Exception:
        pass

    # Fall back to 1-row header
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_column(df, condition, metric, task):
    """
    Expected final names:
        PC_Effectiveness_T1
        MR_Effectiveness_T1
    """
    target = f"{condition}_{metric}_{task}"

    # Exact match
    if target in df.columns:
        return target

    # Backup: search more flexibly
    candidates = [
        c for c in df.columns
        if condition in c and metric in c and task in c
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(f"Could not find column for {condition}, {metric}, {task}")


def paired_dz(diff):
    """Cohen's dz for paired samples."""
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return diff.mean() / sd


def mean_diff_ci(diff, alpha=0.05):
    """95% CI for paired mean difference."""
    n = len(diff)
    mean_diff = diff.mean()

    if n < 2:
        return np.nan, np.nan

    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return mean_diff, mean_diff

    se = sd / np.sqrt(n)
    tcrit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    low = mean_diff - tcrit * se
    high = mean_diff + tcrit * se
    return low, high


def wilcoxon_effect_r_from_diff(diff):
    """
    Approximate Wilcoxon effect size r using z / sqrt(n),
    where z is derived from the signed-rank statistic.
    """
    diff = np.asarray(diff)
    diff = diff[diff != 0]
    n = len(diff)

    if n == 0:
        return 0.0, 0.0

    abs_diff = np.abs(diff)
    ranks = rankdata(abs_diff)
    w_pos = np.sum(ranks[diff > 0])

    mean_w = n * (n + 1) / 4
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if sd_w == 0:
        return 0.0, 0.0

    z = (w_pos - mean_w) / sd_w
    r = z / np.sqrt(n)
    return z, r


def safe_numeric_pair(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    combined = pd.concat([a, b], axis=1).dropna()
    combined.columns = ["PC", "MR"]
    return combined


def choose_paired_test(pc, mr):
    """
    Shapiro on paired differences.
    If normal => paired t-test
    Else => Wilcoxon
    """
    diff = pc - mr

    try:
        if len(diff) >= 3 and diff.nunique() > 1:
            _, p_norm = shapiro(diff)
        else:
            p_norm = 0.0
    except Exception:
        p_norm = 0.0

    if p_norm > 0.05:
        stat, p_val = ttest_rel(pc, mr, nan_policy="omit")
        test = "Paired t-test"
        stat_label = f"t({len(diff)-1})"
        effect = paired_dz(diff)
        effect_label = "dz"
        z_val = np.nan
    else:
        try:
            stat, p_val = wilcoxon(pc, mr, zero_method="wilcox", correction=False)
        except ValueError:
            stat, p_val = 0.0, 1.0

        z_val, effect = wilcoxon_effect_r_from_diff(diff.values)
        test = "Wilcoxon signed-rank"
        stat_label = "W"
        effect_label = "r"

    return diff, p_norm, test, stat_label, stat, p_val, effect_label, effect, z_val


# =========================================================
# MAIN
# =========================================================

df = read_combined_file(INPUT_FILE)

# Identify participant column
participant_col = None
for c in df.columns:
    if "ParticipantID" in c or c == "ParticipantID":
        participant_col = c
        break

if participant_col is None:
    participant_col = df.columns[0]  # fallback

groups = {
    "All_Participants": list(range(len(df))),
    "Experts_Only": expert_indices
}

for group_name, indices in groups.items():
    valid_indices = [i for i in indices if i < len(df)]
    sub = df.iloc[valid_indices].copy()

    results = []

    for task in tasks:
        for metric in metrics:
            try:
                col_pc = get_column(sub, "PC", metric, task)
                col_mr = get_column(sub, "MR", metric, task)
            except KeyError as e:
                print(f"[WARN] {e}")
                continue

            combined = safe_numeric_pair(sub[col_pc], sub[col_mr])

            if len(combined) < MIN_N:
                continue

            pc = combined["PC"]
            mr = combined["MR"]

            diff, p_norm, test, stat_label, stat, p_val, effect_label, effect, z_val = choose_paired_test(pc, mr)
            ci_low, ci_high = mean_diff_ci(diff, alpha=ALPHA)

            mean_pc = pc.mean()
            mean_mr = mr.mean()
            sd_pc = pc.std(ddof=1)
            sd_mr = mr.std(ddof=1)
            mean_diff = diff.mean()

            median_pc = pc.median()
            median_mr = mr.median()
            iqr_pc = pc.quantile(0.75) - pc.quantile(0.25)
            iqr_mr = mr.quantile(0.75) - mr.quantile(0.25)

            pc_higher = int((pc > mr).sum())
            mr_higher = int((mr > pc).sum())
            ties = int((pc == mr).sum())

            if test == "Paired t-test":
                stat_string = f"{stat_label} = {stat:.3f}, p = {p_val:.4f}, {effect_label} = {effect:.3f}"
            else:
                stat_string = f"{stat_label} = {stat:.3f}, p = {p_val:.4f}, z = {z_val:.3f}, {effect_label} = {effect:.3f}"

            results.append({
                "Group": group_name,
                "Task": task,
                "Metric": metric,
                "N": len(combined),

                "PC_Column": col_pc,
                "MR_Column": col_mr,

                "Test": test,
                "Shapiro_p_diff": round(p_norm, 4),
                "Stat_Label": stat_label,
                "Stat_Value": round(stat, 4),
                "Z_Value": round(z_val, 4) if not np.isnan(z_val) else np.nan,
                "p_raw": p_val,

                "Effect_Label": effect_label,
                "Effect_Size": round(effect, 4),

                "Mean_PC": round(mean_pc, 4),
                "SD_PC": round(sd_pc, 4),
                "Mean_MR": round(mean_mr, 4),
                "SD_MR": round(sd_mr, 4),

                "Mean_Diff_PC_minus_MR": round(mean_diff, 4),
                "CI95_Low_Diff": round(ci_low, 4),
                "CI95_High_Diff": round(ci_high, 4),

                "Median_PC": round(median_pc, 4),
                "Median_MR": round(median_mr, 4),
                "IQR_PC": round(iqr_pc, 4),
                "IQR_MR": round(iqr_mr, 4),

                "PC_Higher_Count": pc_higher,
                "MR_Higher_Count": mr_higher,
                "Ties": ties,

                "Stat_String": stat_string
            })

    if not results:
        print(f"No valid results for {group_name}")
        continue

    results_df = pd.DataFrame(results)

    # Holm correction across all quantitative comparisons in this group
    reject, p_corr, _, _ = multipletests(results_df["p_raw"], method="holm", alpha=ALPHA)
    results_df["p_corrected"] = p_corr
    results_df["Significant_Holm"] = reject

    # Optional cleaner significance text
    results_df["Sig_Label"] = np.where(results_df["Significant_Holm"], "Yes", "No")

    # Sort for inspection
    results_df = results_df.sort_values(
        by=["Task", "Metric", "p_corrected", "p_raw"]
    ).reset_index(drop=True)

    out_file = f"quanti_analysis_results/{OUTPUT_PREFIX}_{group_name}.xlsx"
    results_df.to_excel(out_file, index=False)
    print(f"Saved: {out_file}")

print("Done.")