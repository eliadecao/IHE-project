from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_GROUPED_DIR = Path(
    # r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4\grouped_cv"
    r"C:/Users/ROG/IHE-project/data/v7_fk_camera/qc_v7/v4/grouped_cv"
)

MODEL_FILES = {
    "extra_trees": (
        "nonlinear_regression",
        "nonlinear_oof_predictions.csv",
    ),
    "random_forest": (
        "nonlinear_regression",
        "nonlinear_oof_predictions.csv",
    ),
    "hist_gradient_boosting": (
        "nonlinear_regression",
        "nonlinear_oof_predictions.csv",
    ),
    "ridge": (
        "ablation_analysis",
        "ablation_oof_predictions.csv",
    ),
    "elastic_net": (
        "ablation_analysis",
        "ablation_oof_predictions.csv",
    ),
}

LINEAR_EXPERIMENT = "C_kinematics_no_qc_plus_task"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare grouped out-of-fold regression models using "
            "paired participant-level bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--grouped-dir",
        type=Path,
        default=DEFAULT_GROUPED_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser.parse_args()


def load_predictions(
    grouped_dir: Path,
) -> pd.DataFrame:
    frames = []

    nonlinear_path = (
        grouped_dir
        / "nonlinear_regression"
        / "nonlinear_oof_predictions.csv"
    )
    ablation_path = (
        grouped_dir
        / "ablation_analysis"
        / "ablation_oof_predictions.csv"
    )

    if not nonlinear_path.exists():
        raise FileNotFoundError(
            f"Missing nonlinear predictions: {nonlinear_path}"
        )

    if not ablation_path.exists():
        raise FileNotFoundError(
            f"Missing ablation predictions: {ablation_path}"
        )

    nonlinear = pd.read_csv(nonlinear_path)
    nonlinear = nonlinear[
        nonlinear["model"].isin(
            [
                "extra_trees",
                "random_forest",
                "hist_gradient_boosting",
            ]
        )
    ].copy()
    frames.append(nonlinear)

    ablation = pd.read_csv(ablation_path)
    ablation = ablation[
        (ablation["experiment"] == LINEAR_EXPERIMENT)
        & ablation["model"].isin(
            ["ridge", "elastic_net"]
        )
    ].copy()
    frames.append(ablation)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = {
        "model",
        "participant_group",
        "task_clean",
        "cv_fold",
        "y_true",
        "y_pred",
        "absolute_error",
    }
    missing = required - set(combined.columns)

    if missing:
        raise ValueError(
            f"Prediction files are missing columns: {sorted(missing)}"
        )

    return combined


def validate_alignment(
    predictions: pd.DataFrame,
) -> list[str]:
    models = sorted(
        predictions["model"].unique()
    )

    counts = predictions.groupby(
        "model"
    ).size()

    if counts.nunique() != 1:
        raise ValueError(
            "Models do not contain the same number of OOF predictions:\n"
            f"{counts}"
        )

    key_cols = [
        "participant_group",
        "task_clean",
        "cv_fold",
        "y_true",
    ]

    reference = (
        predictions[
            predictions["model"] == models[0]
        ][key_cols]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    for model in models[1:]:
        candidate = (
            predictions[
                predictions["model"] == model
            ][key_cols]
            .sort_values(key_cols)
            .reset_index(drop=True)
        )

        if not reference.equals(candidate):
            raise ValueError(
                f"OOF rows are not aligned for model: {model}"
            )

    return models


def participant_level_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    return (
        predictions.groupby(
            ["model", "participant_group"],
            as_index=False,
        )
        .agg(
            n_operations=("absolute_error", "size"),
            participant_mae=("absolute_error", "mean"),
            participant_rmse=(
                "residual",
                lambda x: float(
                    np.sqrt(
                        np.mean(
                            np.square(x)
                        )
                    )
                ),
            ),
            participant_bias=("residual", "mean"),
        )
    )


def paired_bootstrap_difference(
    participant_metrics: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pivot = participant_metrics.pivot(
        index="participant_group",
        columns="model",
        values=metric,
    )[[model_a, model_b]].dropna()

    groups = pivot.index.to_numpy()
    values_a = pivot[model_a].to_numpy()
    values_b = pivot[model_b].to_numpy()

    observed = float(
        np.mean(values_a - values_b)
    )

    bootstrap_differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(groups),
            size=len(groups),
        )
        bootstrap_differences[i] = np.mean(
            values_a[indices]
            - values_b[indices]
        )

    ci_low, ci_high = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    probability_a_better = float(
        np.mean(
            bootstrap_differences < 0
        )
    )

    differences = values_a - values_b

    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(
                wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = np.nan

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_participants": int(len(pivot)),
        "mean_difference_a_minus_b": observed,
        "ci_2_5": float(ci_low),
        "ci_97_5": float(ci_high),
        "probability_model_a_better": probability_a_better,
        "wilcoxon_p_value": wilcoxon_p,
    }


def operation_level_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, subset in predictions.groupby(
        "model"
    ):
        y_true = subset["y_true"].to_numpy()
        y_pred = subset["y_pred"].to_numpy()
        residual = y_true - y_pred

        mae = float(
            np.mean(
                np.abs(residual)
            )
        )
        rmse = float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        )
        r2 = float(
            1
            - np.sum(residual**2)
            / np.sum(
                (
                    y_true
                    - np.mean(y_true)
                )
                ** 2
            )
        )

        rows.append(
            {
                "model": model,
                "n_operations": len(subset),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mean_residual": float(
                    np.mean(residual)
                ),
                "median_absolute_error": float(
                    np.median(
                        np.abs(residual)
                    )
                ),
                "error_90th_percentile": float(
                    np.percentile(
                        np.abs(residual),
                        90,
                    )
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mae"
    )


def save_model_comparison_plot(
    summary: pd.DataFrame,
    output_dir: Path,
):
    plot_df = summary.sort_values(
        "mae",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    ax.bar(
        plot_df["model"],
        plot_df["mae"],
    )
    ax.set_ylabel(
        "Grouped out-of-fold MAE"
    )
    ax.set_xlabel("Model")
    ax.set_title(
        "Final regression model comparison"
    )
    ax.tick_params(
        axis="x",
        rotation=20,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "final_model_mae_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_extra_trees_scatter(
    predictions: pd.DataFrame,
    output_dir: Path,
):
    subset = predictions[
        predictions["model"] == "extra_trees"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.5, 6)
    )
    ax.scatter(
        subset["y_true"],
        subset["y_pred"],
        alpha=0.7,
    )

    minimum = float(
        min(
            subset["y_true"].min(),
            subset["y_pred"].min(),
        )
    )
    maximum = float(
        max(
            subset["y_true"].max(),
            subset["y_pred"].max(),
        )
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )
    ax.set_xlabel(
        "Observed M-GEARS score"
    )
    ax.set_ylabel(
        "Out-of-fold predicted score"
    )
    ax.set_title(
        "Extra Trees: observed vs predicted"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "extra_trees_observed_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_extra_trees_residual_plot(
    predictions: pd.DataFrame,
    output_dir: Path,
):
    subset = predictions[
        predictions["model"] == "extra_trees"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )
    ax.scatter(
        subset["y_pred"],
        subset["residual"],
        alpha=0.7,
    )
    ax.axhline(
        0,
        linestyle="--",
    )
    ax.set_xlabel(
        "Out-of-fold predicted score"
    )
    ax.set_ylabel(
        "Residual: observed − predicted"
    )
    ax.set_title(
        "Extra Trees residual diagnostics"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "extra_trees_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    output_dir = args.output_dir or (
        args.grouped_dir
        / "final_model_comparison"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions(
        args.grouped_dir
    )
    models = validate_alignment(
        predictions
    )

    participant_metrics = participant_level_table(
        predictions
    )
    participant_metrics.to_csv(
        output_dir
        / "participant_level_model_metrics.csv",
        index=False,
    )

    operation_summary = operation_level_summary(
        predictions
    )
    operation_summary.to_csv(
        output_dir
        / "final_operation_level_metrics.csv",
        index=False,
    )

    rng = np.random.default_rng(
        args.random_state
    )

    comparison_rows = []
    reference_model = "extra_trees"

    for competing_model in [
        model
        for model in models
        if model != reference_model
    ]:
        for metric in [
            "participant_mae",
            "participant_rmse",
        ]:
            comparison_rows.append(
                paired_bootstrap_difference(
                    participant_metrics=participant_metrics,
                    model_a=reference_model,
                    model_b=competing_model,
                    metric=metric,
                    n_bootstrap=args.bootstrap_iterations,
                    rng=rng,
                )
            )

    comparisons = pd.DataFrame(
        comparison_rows
    )
    comparisons.to_csv(
        output_dir
        / "extra_trees_paired_bootstrap_comparisons.csv",
        index=False,
    )

    task_metrics = (
        predictions.groupby(
            ["model", "task_clean"],
            as_index=False,
        )
        .agg(
            n=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
        )
    )
    task_metrics.to_csv(
        output_dir
        / "final_task_level_metrics.csv",
        index=False,
    )

    role_metrics = pd.DataFrame()
    if "role" in predictions.columns:
        role_metrics = (
            predictions.groupby(
                ["model", "role"],
                as_index=False,
            )
            .agg(
                n=("absolute_error", "size"),
                mae=("absolute_error", "mean"),
                mean_residual=("residual", "mean"),
                median_absolute_error=(
                    "absolute_error",
                    "median",
                ),
            )
        )
        role_metrics.to_csv(
            output_dir
            / "final_role_level_metrics.csv",
            index=False,
        )

    save_model_comparison_plot(
        summary=operation_summary,
        output_dir=output_dir,
    )
    save_extra_trees_scatter(
        predictions=predictions,
        output_dir=output_dir,
    )
    save_extra_trees_residual_plot(
        predictions=predictions,
        output_dir=output_dir,
    )

    manifest = {
        "grouped_directory": str(
            args.grouped_dir
        ),
        "output_directory": str(
            output_dir
        ),
        "models_compared": models,
        "reference_model": reference_model,
        "bootstrap_iterations": int(
            args.bootstrap_iterations
        ),
        "bootstrap_unit": (
            "participant_group"
        ),
        "interpretation": (
            "A negative Extra Trees minus comparator difference "
            "means Extra Trees has lower participant-level error."
        ),
        "operation_level_results": (
            operation_summary.to_dict(
                orient="records"
            )
        ),
    }

    with open(
        output_dir
        / "final_model_comparison_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== FINAL MODEL COMPARISON ==="
    )
    print(
        operation_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\n=== EXTRA TREES PAIRED PARTICIPANT-LEVEL COMPARISONS ==="
    )
    print(
        comparisons.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nOutputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()