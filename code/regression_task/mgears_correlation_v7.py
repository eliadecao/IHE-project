from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

DATASET = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\qc_v7\v4"
    r"\dataset_v4_prefiltered.csv"
)

OUTPUT_DIR = DATASET.parent / "mgears_correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# M-GEARS DOMAINS
# These are the six domains that are actually scored.
# Autonomy and Basic Energy Skills are excluded.
# ============================================================

MGEARS = {
    "Depth perception":
        "mgears_depth_perception",

    "Dexterity":
        "mgears_dexterity_with_multiple_wristed_instruments",

    "Efficiency / Flow":
        "mgears_efficiency_flow_of_operation",

    "Force / Tissue handling":
        "mgears_force_sensitivity_and_tissue_handling",

    "Robotic control":
        "mgears_master_manipulator_workspace_robotic_control",

    "Overall quality":
        "mgears_overall_performance_quality_of_the_final_product",
}


# ============================================================
# INTERPRETABLE KINEMATIC METRICS
#
# USM0 = left instrument
# USM2 = principal right instrument
#
# We deliberately use a compact set rather than all 225
# features so that the heatmaps remain clinically interpretable.
# ============================================================

METRICS = {

    # ----- Overall efficiency -----

    "Task duration":
        "qc_fk_duration_seconds",

    # ----- Instrument path / economy -----

    "Left path length":
        "USM0_pose_global_path_l2",

    "Right path length":
        "USM2_pose_global_path_l2",

    "L/R path imbalance":
        "USM0_USM2_pose_global_path_length_imbalance",

    # ----- Speed -----

    "Left mean speed":
        "USM0_pose_global_mean_speed_l2",

    "Right mean speed":
        "USM2_pose_global_mean_speed_l2",

    # ----- Smoothness -----

    "Left smoothness (LDJ)":
        "USM0_pose_global_log_dimensionless_jerk",

    "Right smoothness (LDJ)":
        "USM2_pose_global_log_dimensionless_jerk",

    # ----- Pauses / hesitation -----

    "Left pause fraction":
        "USM0_pose_global_pause_fraction",

    "Right pause fraction":
        "USM2_pose_global_pause_fraction",

    # ----- Bimanual coordination -----

    "Inter-tool distance":
        "USM0_USM2_pose_global_mean_distance_rel",

    "Relative tool speed":
        "USM0_USM2_pose_global_mean_abs_relative_speed",

    "Tool speed correlation":
        "USM0_USM2_pose_global_speed_correlation",

    # ----- Camera -----

    "Camera path length":
        "camera_path_length_m",

    "Camera move count":
        "camera_move_count",

    "Camera moving fraction":
        "camera_moving_fraction",

    "Camera-tool distance":
        "camera_to_primary_tools_mean_distance_m",

    # ----- Instrument orientation -----

    "Left rotation path":
        "USM0_pose_global_rotation_path_radians",

    "Right rotation path":
        "USM2_pose_global_rotation_path_radians",
}


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(p_values):

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid = np.isfinite(p_values)

    if valid.sum() == 0:
        return adjusted

    p = p_values[valid]

    order = np.argsort(p)
    ranked = p[order]

    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)

    # Ensure monotonic corrected p-values
    q = np.minimum.accumulate(q[::-1])[::-1]

    q = np.clip(q, 0, 1)

    restored = np.empty(n)
    restored[order] = q

    adjusted[np.where(valid)[0]] = restored

    return adjusted


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

print("\n" + "=" * 75)
print("M-GEARS × KINEMATIC METRIC CORRELATION ANALYSIS")
print("=" * 75)

print(f"\nDataset: {DATASET}")
print(f"Shape: {df.shape}")


# ============================================================
# CHECK COLUMNS
# ============================================================

available_mgears = {
    label: col
    for label, col in MGEARS.items()
    if col in df.columns
}

missing_mgears = {
    label: col
    for label, col in MGEARS.items()
    if col not in df.columns
}

available_metrics = {
    label: col
    for label, col in METRICS.items()
    if col in df.columns
}

missing_metrics = {
    label: col
    for label, col in METRICS.items()
    if col not in df.columns
}


print(
    f"\nM-GEARS domains found: "
    f"{len(available_mgears)}/{len(MGEARS)}"
)

if missing_mgears:
    print("\nMISSING M-GEARS COLUMNS:")
    for label, col in missing_mgears.items():
        print(f"  {label:<30} -> {col}")


print(
    f"\nKinematic metrics found: "
    f"{len(available_metrics)}/{len(METRICS)}"
)

if missing_metrics:
    print("\nMISSING KINEMATIC METRICS:")
    for label, col in missing_metrics.items():
        print(f"  {label:<30} -> {col}")


if len(available_mgears) == 0:
    raise ValueError(
        "No M-GEARS domain columns were found."
    )

if len(available_metrics) == 0:
    raise ValueError(
        "No requested kinematic metrics were found."
    )


# ============================================================
# TASK INFORMATION
# ============================================================

tasks = sorted(
    df["task_clean"]
    .dropna()
    .unique()
)

print("\nTasks:")

for task in tasks:
    n = int(df["task_clean"].eq(task).sum())
    print(f"  {task:<25} n = {n}")


# ============================================================
# CALCULATE SPEARMAN CORRELATIONS
# ============================================================

all_results = []


for task in tasks:

    task_df = df[
        df["task_clean"].eq(task)
    ].copy()

    print("\n" + "=" * 75)
    print(f"TASK: {task}")
    print(f"N = {len(task_df)}")
    print("=" * 75)

    rows = []

    for metric_name, metric_col in available_metrics.items():

        x = pd.to_numeric(
            task_df[metric_col],
            errors="coerce"
        )

        for gears_name, gears_col in available_mgears.items():

            y = pd.to_numeric(
                task_df[gears_col],
                errors="coerce"
            )

            valid = x.notna() & y.notna()

            n = int(valid.sum())

            if (
                n < 8
                or x[valid].nunique() < 2
                or y[valid].nunique() < 2
            ):
                rho = np.nan
                p_value = np.nan

            else:

                result = spearmanr(
                    x[valid],
                    y[valid]
                )

                rho = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "task": task,

                    "metric":
                        metric_name,

                    "metric_column":
                        metric_col,

                    "mgears_domain":
                        gears_name,

                    "mgears_column":
                        gears_col,

                    "n":
                        n,

                    "spearman_rho":
                        rho,

                    "abs_rho":
                        (
                            abs(rho)
                            if np.isfinite(rho)
                            else np.nan
                        ),

                    "p_value":
                        p_value,
                }
            )

    task_results = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Multiple-comparison correction within each task
    # --------------------------------------------------------

    task_results["p_fdr"] = benjamini_hochberg(
        task_results["p_value"].to_numpy()
    )

    task_results["significant_fdr_005"] = (
        task_results["p_fdr"] < 0.05
    )

    all_results.append(task_results)


# ============================================================
# COMBINE
# ============================================================

results = pd.concat(
    all_results,
    ignore_index=True
)

results.to_csv(
    OUTPUT_DIR / "all_correlations.csv",
    index=False
)


# ============================================================
# HEATMAP FOR EACH TASK
# ============================================================

for task in tasks:

    task_results = results[
        results["task"].eq(task)
    ].copy()

    heatmap = task_results.pivot(
        index="metric",
        columns="mgears_domain",
        values="spearman_rho"
    )

    # Same ordering for every task
    heatmap = heatmap.reindex(
        index=list(available_metrics.keys()),
        columns=list(available_mgears.keys())
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                8,
                len(heatmap.index) * 0.48
            )
        )
    )

    image = ax.imshow(
        heatmap.to_numpy(dtype=float),
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    ax.set_xticks(
        np.arange(len(heatmap.columns))
    )

    ax.set_xticklabels(
        heatmap.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(
        np.arange(len(heatmap.index))
    )

    ax.set_yticklabels(
        heatmap.index
    )

    # --------------------------------------------------------
    # Write rho inside cells
    # Add * if FDR significant
    # --------------------------------------------------------

    for i, metric in enumerate(heatmap.index):

        for j, domain in enumerate(heatmap.columns):

            value = heatmap.loc[
                metric,
                domain
            ]

            if not np.isfinite(value):
                continue

            row = task_results[
                task_results["metric"].eq(metric)
                &
                task_results["mgears_domain"].eq(domain)
            ]

            significant = (
                not row.empty
                and bool(
                    row.iloc[0][
                        "significant_fdr_005"
                    ]
                )
            )

            text = f"{value:.2f}"

            if significant:
                text += "*"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title(
        f"{task}\n"
        "Kinematic metrics vs M-GEARS "
        "(Spearman ρ)"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax
    )

    colourbar.set_label(
        "Spearman correlation (ρ)"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / f"heatmap_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# TOP 5 METRICS FOR EACH DOMAIN × TASK
# ============================================================

top5_list = []

for (task, domain), subset in results.groupby(
    ["task", "mgears_domain"]
):

    subset = (
        subset
        .dropna(
            subset=["spearman_rho"]
        )
        .sort_values(
            "abs_rho",
            ascending=False
        )
        .head(5)
    )

    top5_list.append(subset)


if top5_list:

    top5 = pd.concat(
        top5_list,
        ignore_index=True
    )

else:

    top5 = pd.DataFrame()


top5.to_csv(
    OUTPUT_DIR
    / "top5_metrics_by_task_and_mgears.csv",
    index=False
)


# ============================================================
# FDR-SIGNIFICANT RESULTS
# ============================================================

significant = results[
    results["significant_fdr_005"]
].copy()

significant = significant.sort_values(
    ["task", "abs_rho"],
    ascending=[True, False]
)

significant.to_csv(
    OUTPUT_DIR
    / "fdr_significant_correlations.csv",
    index=False
)


# ============================================================
# OVERALL CORRELATION WITH TOTAL M-GEARS
#
# Useful as a secondary analysis.
# ============================================================

total_rows = []

if "target_score" in df.columns:

    for task in tasks:

        task_df = df[
            df["task_clean"].eq(task)
        ].copy()

        y = pd.to_numeric(
            task_df["target_score"],
            errors="coerce"
        )

        for metric_name, metric_col in available_metrics.items():

            x = pd.to_numeric(
                task_df[metric_col],
                errors="coerce"
            )

            valid = x.notna() & y.notna()

            n = int(valid.sum())

            if (
                n < 8
                or x[valid].nunique() < 2
                or y[valid].nunique() < 2
            ):
                rho = np.nan
                p = np.nan

            else:

                result = spearmanr(
                    x[valid],
                    y[valid]
                )

                rho = float(result.statistic)
                p = float(result.pvalue)

            total_rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "metric_column": metric_col,
                    "n": n,
                    "spearman_rho": rho,
                    "abs_rho": (
                        abs(rho)
                        if np.isfinite(rho)
                        else np.nan
                    ),
                    "p_value": p,
                }
            )


total_results = pd.DataFrame(total_rows)

if not total_results.empty:

    total_results["p_fdr"] = np.nan

    for task in tasks:

        mask = total_results[
            "task"
        ].eq(task)

        total_results.loc[
            mask,
            "p_fdr"
        ] = benjamini_hochberg(
            total_results.loc[
                mask,
                "p_value"
            ].to_numpy()
        )

    total_results.to_csv(
        OUTPUT_DIR
        / "total_mgears_correlations.csv",
        index=False
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nMetrics analysed: "
    f"{len(available_metrics)}"
)

print(
    f"M-GEARS domains analysed: "
    f"{len(available_mgears)}"
)

print(
    f"Tasks analysed: "
    f"{len(tasks)}"
)

print(
    f"Total domain correlations: "
    f"{len(results)}"
)

print(
    f"FDR-significant correlations: "
    f"{len(significant)}"
)


print("\nTOP 20 DOMAIN CORRELATIONS")
print("-" * 75)

top20 = (
    results
    .dropna(
        subset=["spearman_rho"]
    )
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .head(20)
)

print(
    top20[
        [
            "task",
            "metric",
            "mgears_domain",
            "n",
            "spearman_rho",
            "p_value",
            "p_fdr",
        ]
    ].to_string(index=False)
)


if not total_results.empty:

    print("\nTOP 15 TOTAL M-GEARS CORRELATIONS")
    print("-" * 75)

    print(
        total_results
        .dropna(
            subset=["spearman_rho"]
        )
        .sort_values(
            "abs_rho",
            ascending=False
        )
        [
            [
                "task",
                "metric",
                "n",
                "spearman_rho",
                "p_value",
                "p_fdr",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


print("\nOutputs:")
print(OUTPUT_DIR)
