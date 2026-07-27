from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4"
    r"\grouped_cv\baseline_regression"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse coefficient magnitude, sign consistency and fold "
            "stability for grouped regression models."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing baseline regression outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to input-dir / feature_importance",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to include in plots and summaries.",
    )
    return parser.parse_args()


def classify_feature(feature: str) -> dict[str, str]:
    lower = feature.lower()

    if lower.startswith("usm"):
        source = "USM"
    elif lower.startswith("sus"):
        source = "SUS"
    elif lower.startswith("suj"):
        source = "SUJ"
    elif "endoscope" in lower:
        source = "Endoscope"
    else:
        source = "Other"

    if "jerk" in lower:
        metric_family = "jerk"
    elif "acceleration" in lower:
        metric_family = "acceleration"
    elif "velocity" in lower or "speed" in lower:
        metric_family = "velocity"
    elif "path_length" in lower:
        metric_family = "path_length"
    elif "smoothness" in lower:
        metric_family = "smoothness"
    elif "rotation" in lower or "rotational" in lower:
        metric_family = "rotation"
    elif "workspace" in lower or "range" in lower:
        metric_family = "workspace"
    elif "duration" in lower or "time" in lower:
        metric_family = "time"
    else:
        metric_family = "other"

    arm_match = re.search(r"(usm\d+)", lower)
    arm = arm_match.group(1).upper() if arm_match else "NA"

    return {
        "feature_source": source,
        "metric_family": metric_family,
        "arm": arm,
    }


def add_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = df["feature"].apply(classify_feature).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), metadata], axis=1)


def prepare_coefficient_summary(
    fold_coefficients: pd.DataFrame,
    model_name: str,
    total_folds: int,
) -> pd.DataFrame:
    subset = fold_coefficients[
        fold_coefficients["model"] == model_name
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No coefficients were found for model: {model_name}"
        )

    subset["sign"] = np.sign(subset["coefficient"])
    subset["nonzero"] = ~np.isclose(
        subset["coefficient"],
        0.0,
        atol=1e-12,
    )

    summary = (
        subset.groupby("feature", as_index=False)
        .agg(
            folds_present=("cv_fold", "nunique"),
            nonzero_folds=("nonzero", "sum"),
            mean_coefficient=("coefficient", "mean"),
            median_coefficient=("coefficient", "median"),
            std_coefficient=("coefficient", "std"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            median_abs_coefficient=("abs_coefficient", "median"),
            max_abs_coefficient=("abs_coefficient", "max"),
            positive_folds=(
                "coefficient",
                lambda x: int((x > 0).sum()),
            ),
            negative_folds=(
                "coefficient",
                lambda x: int((x < 0).sum()),
            ),
        )
    )

    summary["fold_coverage"] = (
        summary["folds_present"] / total_folds
    )
    summary["nonzero_fold_fraction"] = (
        summary["nonzero_folds"] / total_folds
    )

    denominator = (
        summary["positive_folds"]
        + summary["negative_folds"]
    ).replace(0, np.nan)

    summary["sign_consistency"] = (
        summary[
            ["positive_folds", "negative_folds"]
        ].max(axis=1)
        / denominator
    ).fillna(0.0)

    summary["stable_all_folds"] = (
        (summary["folds_present"] == total_folds)
        & (summary["nonzero_folds"] == total_folds)
        & (summary["sign_consistency"] == 1.0)
    )

    summary["stability_weighted_importance"] = (
        summary["mean_abs_coefficient"]
        * summary["nonzero_fold_fraction"]
        * summary["sign_consistency"]
    )

    summary["model"] = model_name
    summary = add_feature_metadata(summary)

    return summary.sort_values(
        [
            "stability_weighted_importance",
            "mean_abs_coefficient",
        ],
        ascending=False,
    )


def save_top_feature_plot(
    summary: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    top_n: int,
):
    plot_df = (
        summary.head(top_n)
        .sort_values(
            "stability_weighted_importance",
            ascending=True,
        )
    )

    fig_height = max(6, top_n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(
        plot_df["feature"],
        plot_df["stability_weighted_importance"],
    )
    ax.set_xlabel(
        "Stability-weighted absolute coefficient"
    )
    ax.set_ylabel("Feature")
    ax.set_title(
        f"{model_name}: top stable coefficient features"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{model_name}_top_stable_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_signed_coefficient_plot(
    summary: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    top_n: int,
):
    plot_df = (
        summary.head(top_n)
        .sort_values(
            "mean_coefficient",
            ascending=True,
        )
    )

    fig_height = max(6, top_n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(
        plot_df["feature"],
        plot_df["mean_coefficient"],
    )
    ax.axvline(0, linestyle="--")
    ax.set_xlabel(
        "Mean standardised coefficient across folds"
    )
    ax.set_ylabel("Feature")
    ax.set_title(
        f"{model_name}: direction of top coefficients"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{model_name}_signed_top_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_family_summary_plot(
    family_summary: pd.DataFrame,
    model_name: str,
    output_dir: Path,
):
    plot_df = (
        family_summary.sort_values(
            "total_stability_weighted_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        plot_df["metric_family"],
        plot_df[
            "total_stability_weighted_importance"
        ],
    )
    ax.set_xlabel(
        "Total stability-weighted importance"
    )
    ax.set_ylabel("Feature family")
    ax.set_title(
        f"{model_name}: importance by metric family"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{model_name}_importance_by_metric_family.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or (
        input_dir / "feature_importance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficient_path = (
        input_dir / "fold_coefficients.csv"
    )
    fold_metrics_path = (
        input_dir / "fold_metrics.csv"
    )

    if not coefficient_path.exists():
        raise FileNotFoundError(
            f"Missing file: {coefficient_path}"
        )

    fold_coefficients = pd.read_csv(
        coefficient_path
    )

    if fold_metrics_path.exists():
        fold_metrics = pd.read_csv(
            fold_metrics_path
        )
        total_folds = int(
            fold_metrics["cv_fold"].nunique()
        )
    else:
        total_folds = int(
            fold_coefficients["cv_fold"].nunique()
        )

    models = [
        model
        for model in ["ridge", "elastic_net"]
        if model
        in set(fold_coefficients["model"].unique())
    ]

    if not models:
        raise ValueError(
            "Neither ridge nor elastic_net coefficients were found."
        )

    all_model_summaries = []
    stable_feature_tables = []
    family_tables = []
    source_tables = []
    arm_tables = []

    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    print(f"Input directory:      {input_dir}")
    print(f"Detected CV folds:    {total_folds}")
    print(f"Models:               {models}")

    for model_name in models:
        summary = prepare_coefficient_summary(
            fold_coefficients=fold_coefficients,
            model_name=model_name,
            total_folds=total_folds,
        )

        summary.to_csv(
            output_dir
            / f"{model_name}_feature_stability.csv",
            index=False,
        )

        stable_features = summary[
            summary["stable_all_folds"]
        ].copy()
        stable_features.to_csv(
            output_dir
            / f"{model_name}_stable_all_folds.csv",
            index=False,
        )

        family_summary = (
            summary.groupby(
                "metric_family",
                as_index=False,
            )
            .agg(
                n_features=("feature", "nunique"),
                mean_feature_importance=(
                    "stability_weighted_importance",
                    "mean",
                ),
                total_stability_weighted_importance=(
                    "stability_weighted_importance",
                    "sum",
                ),
                n_stable_features=(
                    "stable_all_folds",
                    "sum",
                ),
            )
            .sort_values(
                "total_stability_weighted_importance",
                ascending=False,
            )
        )
        family_summary["model"] = model_name
        family_summary.to_csv(
            output_dir
            / f"{model_name}_metric_family_summary.csv",
            index=False,
        )

        source_summary = (
            summary.groupby(
                "feature_source",
                as_index=False,
            )
            .agg(
                n_features=("feature", "nunique"),
                total_stability_weighted_importance=(
                    "stability_weighted_importance",
                    "sum",
                ),
                mean_feature_importance=(
                    "stability_weighted_importance",
                    "mean",
                ),
                n_stable_features=(
                    "stable_all_folds",
                    "sum",
                ),
            )
            .sort_values(
                "total_stability_weighted_importance",
                ascending=False,
            )
        )
        source_summary["model"] = model_name
        source_summary.to_csv(
            output_dir
            / f"{model_name}_feature_source_summary.csv",
            index=False,
        )

        arm_summary = (
            summary[
                summary["arm"] != "NA"
            ]
            .groupby(
                "arm",
                as_index=False,
            )
            .agg(
                n_features=("feature", "nunique"),
                total_stability_weighted_importance=(
                    "stability_weighted_importance",
                    "sum",
                ),
                mean_feature_importance=(
                    "stability_weighted_importance",
                    "mean",
                ),
                n_stable_features=(
                    "stable_all_folds",
                    "sum",
                ),
            )
            .sort_values(
                "total_stability_weighted_importance",
                ascending=False,
            )
        )
        arm_summary["model"] = model_name
        arm_summary.to_csv(
            output_dir
            / f"{model_name}_arm_summary.csv",
            index=False,
        )

        save_top_feature_plot(
            summary=summary,
            model_name=model_name,
            output_dir=output_dir,
            top_n=args.top_n,
        )
        save_signed_coefficient_plot(
            summary=summary,
            model_name=model_name,
            output_dir=output_dir,
            top_n=args.top_n,
        )
        save_family_summary_plot(
            family_summary=family_summary,
            model_name=model_name,
            output_dir=output_dir,
        )

        all_model_summaries.append(summary)
        stable_feature_tables.append(stable_features)
        family_tables.append(family_summary)
        source_tables.append(source_summary)
        arm_tables.append(arm_summary)

        print(f"\n--- {model_name} ---")
        print(
            f"Features evaluated:       {len(summary)}"
        )
        print(
            "Stable in all folds:      "
            f"{int(summary['stable_all_folds'].sum())}"
        )
        print("\nTop 15 stable features:")
        display_cols = [
            "feature",
            "mean_coefficient",
            "mean_abs_coefficient",
            "nonzero_folds",
            "sign_consistency",
            "metric_family",
            "feature_source",
        ]
        print(
            summary[
                display_cols
            ]
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    combined_summary = pd.concat(
        all_model_summaries,
        ignore_index=True,
    )
    combined_summary.to_csv(
        output_dir
        / "combined_model_feature_stability.csv",
        index=False,
    )

    # Features that are important in both Ridge and Elastic Net
    pivot = combined_summary.pivot_table(
        index="feature",
        columns="model",
        values="stability_weighted_importance",
        aggfunc="first",
    ).fillna(0.0)

    for model_name in models:
        if model_name not in pivot.columns:
            pivot[model_name] = 0.0

    pivot["mean_across_models"] = pivot[
        models
    ].mean(axis=1)
    pivot["minimum_across_models"] = pivot[
        models
    ].min(axis=1)
    pivot["important_in_all_models"] = (
        pivot[models] > 0
    ).all(axis=1)

    cross_model = (
        pivot.reset_index()
        .sort_values(
            [
                "minimum_across_models",
                "mean_across_models",
            ],
            ascending=False,
        )
    )
    cross_model = add_feature_metadata(
        cross_model
    )
    cross_model.to_csv(
        output_dir
        / "cross_model_consensus_features.csv",
        index=False,
    )

    manifest = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "models_analysed": models,
        "n_cv_folds": total_folds,
        "top_n_plotted": int(args.top_n),
        "interpretation_notes": [
            (
                "Coefficients are comparable because the modelling "
                "pipeline standardised features inside each fold."
            ),
            (
                "Sign consistency indicates whether a feature was "
                "associated with higher or lower scores in the same "
                "direction across folds."
            ),
            (
                "Elastic Net zero coefficients indicate features "
                "excluded by regularisation in that fitted fold."
            ),
            (
                "Coefficient magnitude describes model association, "
                "not causal importance."
            ),
        ],
    }

    with open(
        output_dir / "feature_importance_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Outputs saved to: {output_dir}")
    print(
        "\nMost useful files:\n"
        "  - elastic_net_feature_stability.csv\n"
        "  - ridge_feature_stability.csv\n"
        "  - cross_model_consensus_features.csv\n"
        "  - elastic_net_metric_family_summary.csv\n"
        "  - elastic_net_top_stable_features.png"
    )


if __name__ == "__main__":
    main()