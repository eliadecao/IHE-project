from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_GROUPED_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4\grouped_cv"
)

TARGET_COLUMN = "target_score"

BASE_EXCLUSIONS = {
    "file_name",
    "role",
    "participant_id",
    "participant_group",
    "task",
    "task_clean",
    "trial",
    "licence",
    "date",
    "session_licence",
    "session_datetime",
    "target_score",
    "percentage_score",
    "target_fraction",
    "target_score_std",
    "percentage_score_std",
    "n_unique_target_scores",
    "label_disagreement_flag",
    "n_label_records",
    "n_applicable_mgears_domains",
    "mgears_domain_score_sum",
    "target_minus_domain_sum",
    "inferred_max_score",
    "cv_fold",
}

LEAKAGE_PREFIXES = ("mgears_", "label_")

LEAKAGE_FRAGMENTS = (
    "target_score",
    "percentage_score",
    "domain_score",
    "mgears",
)

QC_FRAGMENTS = (
    "orthogonality_error",
    "determinant_error",
    "rotation_valid",
    "invalid_rotation",
    "nan_fraction",
    "missing_fraction",
    "interpolation_fraction",
    "reconstruction_error",
    "timestamp_error",
)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)

        corr = X_df.corr(
            method="pearson",
            min_periods=5,
        ).abs()

        upper = corr.where(
            np.triu(
                np.ones(corr.shape, dtype=bool),
                k=1,
            )
        )

        self.columns_to_drop_ = [
            col
            for col in upper.columns
            if (upper[col] >= self.threshold).any()
        ]

        self.columns_to_keep_ = [
            col
            for col in X_df.columns
            if col not in self.columns_to_drop_
        ]

        if not self.columns_to_keep_:
            raise ValueError(
                "CorrelationFilter removed every feature."
            )

        return self

    def transform(self, X):
        X_df = self._as_dataframe(X)
        return X_df.loc[
            :,
            self.columns_to_keep_,
        ].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.columns_to_keep_,
            dtype=object,
        )

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        columns = [
            f"feature_{i}"
            for i in range(X.shape[1])
        ]
        return pd.DataFrame(
            X,
            columns=columns,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate outer-test-fold permutation importance "
            "for the final Extra Trees model."
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
        "--target",
        type=str,
        default=TARGET_COLUMN,
        choices=[
            "target_score",
            "percentage_score",
        ],
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
    )
    return parser.parse_args()


def select_clean_features(
    df: pd.DataFrame,
) -> list[str]:
    features = []

    for col in df.columns:
        lower = col.lower()

        if col in BASE_EXCLUSIONS:
            continue

        if lower.startswith(LEAKAGE_PREFIXES):
            continue

        if any(
            fragment in lower
            for fragment in LEAKAGE_FRAGMENTS
        ):
            continue

        if lower.startswith("qc_"):
            continue

        if any(
            fragment in lower
            for fragment in QC_FRAGMENTS
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        features.append(col)

    if not features:
        raise ValueError(
            "No clean numeric features found."
        )

    return features


def build_pipeline(
    numeric_features: list[str],
    model: ExtraTreesRegressor,
    corr_threshold: float,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "correlation_filter",
                CorrelationFilter(
                    threshold=corr_threshold
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    task_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "task",
                task_pipeline,
                ["task_clean"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def load_best_parameters(
    grouped_dir: Path,
) -> dict[int, dict]:
    path = (
        grouped_dir
        / "nonlinear_regression"
        / "nonlinear_hyperparameters.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing hyperparameter file: {path}"
        )

    tuning = pd.read_csv(path)
    tuning = tuning[
        tuning["model"] == "extra_trees"
    ].copy()

    if tuning.empty:
        raise ValueError(
            "No Extra Trees rows found in hyperparameter file."
        )

    result = {}

    for _, row in tuning.iterrows():
        fold = int(row["cv_fold"])
        params = json.loads(
            row["best_parameters"]
        )

        cleaned = {
            key.replace("model__", ""): value
            for key, value in params.items()
        }

        result[fold] = cleaned

    return result


def classify_feature(
    feature: str,
) -> dict[str, str]:
    lower = feature.lower()

    if "task_clean_" in lower:
        source = "Task"
        family = "task"
        arm = "NA"
    else:
        if "usm0" in lower:
            arm = "USM0"
        elif "usm1" in lower:
            arm = "USM1"
        elif "usm2" in lower:
            arm = "USM2"
        elif "usm3" in lower:
            arm = "USM3"
        else:
            arm = "NA"

        if "usm" in lower:
            source = "USM"
        elif "sus" in lower:
            source = "SUS"
        elif "suj" in lower:
            source = "SUJ"
        elif "endoscope" in lower:
            source = "Endoscope"
        else:
            source = "Other"

        if "jerk" in lower:
            family = "jerk"
        elif "acceleration" in lower:
            family = "acceleration"
        elif "velocity" in lower or "speed" in lower:
            family = "velocity"
        elif "path_length" in lower:
            family = "path_length"
        elif "smoothness" in lower:
            family = "smoothness"
        elif "rotation" in lower or "angular" in lower:
            family = "rotation"
        elif "range" in lower or "workspace" in lower:
            family = "workspace"
        elif "pause" in lower:
            family = "pause"
        elif "idle" in lower:
            family = "idle"
        elif "duration" in lower or "time" in lower:
            family = "time"
        elif "distance" in lower:
            family = "distance"
        else:
            family = "other"

    return {
        "feature_source": source,
        "metric_family": family,
        "arm": arm,
    }


def clean_feature_name(
    name: str,
) -> str:
    prefixes = (
        "numeric__",
        "task__",
    )

    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


def main():
    args = parse_args()

    data_path = (
        args.grouped_dir
        / "dataset_v4_grouped_folds.csv"
    )
    output_dir = args.output_dir or (
        args.grouped_dir
        / "extra_trees_explainability"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing grouped dataset: {data_path}"
        )

    df = pd.read_csv(data_path)

    required = {
        args.target,
        "participant_group",
        "cv_fold",
        "task_clean",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    feature_cols = select_clean_features(df)
    best_parameters = load_best_parameters(
        args.grouped_dir
    )

    X = df[
        feature_cols + ["task_clean"]
    ].copy()
    y = pd.to_numeric(
        df[args.target],
        errors="raise",
    )

    fold_rows = []
    impurity_rows = []

    folds = sorted(
        df["cv_fold"].astype(int).unique()
    )

    print(
        "\n=== EXTRA TREES TEST-FOLD PERMUTATION IMPORTANCE ==="
    )
    print(f"Dataset shape:       {df.shape}")
    print(f"Numeric features:    {len(feature_cols)}")
    print(f"Permutation repeats: {args.n_repeats}")
    print(f"Outer folds:         {folds}")

    for fold in folds:
        if fold not in best_parameters:
            raise ValueError(
                f"No saved Extra Trees parameters for fold {fold}"
            )

        train_mask = (
            df["cv_fold"].astype(int)
            != fold
        )
        test_mask = ~train_mask

        model_params = {
            **best_parameters[fold],
            "random_state": (
                args.random_state + fold
            ),
            "n_jobs": 1,
        }

        model = ExtraTreesRegressor(
            **model_params
        )

        pipeline = build_pipeline(
            numeric_features=feature_cols,
            model=model,
            corr_threshold=args.corr_threshold,
        )

        X_train = X.loc[
            train_mask
        ].copy()
        X_test = X.loc[
            test_mask
        ].copy()
        y_train = y.loc[
            train_mask
        ].copy()
        y_test = y.loc[
            test_mask
        ].copy()

        pipeline.fit(
            X_train,
            y_train,
        )

        transformed_test = pipeline.named_steps[
            "preprocess"
        ].transform(X_test)

        fitted_model = pipeline.named_steps[
            "model"
        ]

        feature_names = [
            clean_feature_name(name)
            for name
            in pipeline.named_steps[
                "preprocess"
            ].get_feature_names_out()
        ]

        perm = permutation_importance(
            estimator=fitted_model,
            X=transformed_test,
            y=y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=args.n_repeats,
            random_state=(
                args.random_state + fold
            ),
            n_jobs=args.n_jobs,
        )

        fold_df = pd.DataFrame(
            {
                "cv_fold": fold,
                "feature": feature_names,
                "permutation_importance_mean": (
                    perm.importances_mean
                ),
                "permutation_importance_std": (
                    perm.importances_std
                ),
                "impurity_importance": (
                    fitted_model.feature_importances_
                ),
            }
        )

        metadata = (
            fold_df["feature"]
            .apply(classify_feature)
            .apply(pd.Series)
        )

        fold_df = pd.concat(
            [
                fold_df.reset_index(drop=True),
                metadata,
            ],
            axis=1,
        )

        fold_rows.append(fold_df)

        print(
            f"Fold {fold}: "
            f"{len(feature_names)} transformed features, "
            f"top permutation importance="
            f"{fold_df['permutation_importance_mean'].max():.4f}"
        )

    all_folds = pd.concat(
        fold_rows,
        ignore_index=True,
    )
    all_folds.to_csv(
        output_dir
        / "extra_trees_fold_permutation_importance.csv",
        index=False,
    )

    summary = (
        all_folds.groupby(
            [
                "feature",
                "feature_source",
                "metric_family",
                "arm",
            ],
            as_index=False,
        )
        .agg(
            folds_present=(
                "cv_fold",
                "nunique",
            ),
            mean_permutation_importance=(
                "permutation_importance_mean",
                "mean",
            ),
            median_permutation_importance=(
                "permutation_importance_mean",
                "median",
            ),
            std_permutation_importance=(
                "permutation_importance_mean",
                "std",
            ),
            positive_folds=(
                "permutation_importance_mean",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            mean_impurity_importance=(
                "impurity_importance",
                "mean",
            ),
        )
    )

    summary["positive_fold_fraction"] = (
        summary["positive_folds"]
        / summary["folds_present"]
    )

    summary["stable_permutation_score"] = (
        summary[
            "mean_permutation_importance"
        ].clip(lower=0)
        * summary[
            "positive_fold_fraction"
        ]
    )

    summary = summary.sort_values(
        [
            "stable_permutation_score",
            "mean_permutation_importance",
        ],
        ascending=False,
    )

    summary.to_csv(
        output_dir
        / "extra_trees_permutation_importance_summary.csv",
        index=False,
    )

    stable = summary[
        (summary["positive_folds"] >= 4)
        & (
            summary[
                "mean_permutation_importance"
            ] > 0
        )
    ].copy()

    stable.to_csv(
        output_dir
        / "extra_trees_stable_permutation_features.csv",
        index=False,
    )

    family_summary = (
        summary.groupby(
            "metric_family",
            as_index=False,
        )
        .agg(
            n_features=(
                "feature",
                "nunique",
            ),
            total_stable_importance=(
                "stable_permutation_score",
                "sum",
            ),
            mean_stable_importance=(
                "stable_permutation_score",
                "mean",
            ),
            n_stable_features=(
                "positive_folds",
                lambda values: int(
                    (values >= 4).sum()
                ),
            ),
        )
        .sort_values(
            "total_stable_importance",
            ascending=False,
        )
    )

    family_summary.to_csv(
        output_dir
        / "extra_trees_metric_family_summary.csv",
        index=False,
    )

    source_summary = (
        summary.groupby(
            "feature_source",
            as_index=False,
        )
        .agg(
            n_features=(
                "feature",
                "nunique",
            ),
            total_stable_importance=(
                "stable_permutation_score",
                "sum",
            ),
            mean_stable_importance=(
                "stable_permutation_score",
                "mean",
            ),
        )
        .sort_values(
            "total_stable_importance",
            ascending=False,
        )
    )

    source_summary.to_csv(
        output_dir
        / "extra_trees_feature_source_summary.csv",
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
            n_features=(
                "feature",
                "nunique",
            ),
            total_stable_importance=(
                "stable_permutation_score",
                "sum",
            ),
            mean_stable_importance=(
                "stable_permutation_score",
                "mean",
            ),
        )
        .sort_values(
            "total_stable_importance",
            ascending=False,
        )
    )

    arm_summary.to_csv(
        output_dir
        / "extra_trees_arm_summary.csv",
        index=False,
    )

    top_plot = (
        summary.head(args.top_n)
        .sort_values(
            "stable_permutation_score",
            ascending=True,
        )
    )

    fig_height = max(
        7,
        args.top_n * 0.3,
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )
    ax.barh(
        top_plot["feature"],
        top_plot[
            "stable_permutation_score"
        ],
    )
    ax.set_xlabel(
        "Stable test-fold permutation importance "
        "(increase in MAE)"
    )
    ax.set_ylabel("Feature")
    ax.set_title(
        "Extra Trees: stable outer-test-fold "
        "permutation importance"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "extra_trees_top_permutation_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    family_plot = family_summary.sort_values(
        "total_stable_importance",
        ascending=True,
    )
    ax.barh(
        family_plot["metric_family"],
        family_plot[
            "total_stable_importance"
        ],
    )
    ax.set_xlabel(
        "Total stable permutation importance"
    )
    ax.set_ylabel(
        "Metric family"
    )
    ax.set_title(
        "Extra Trees importance by metric family"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "extra_trees_metric_family_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    manifest = {
        "dataset": str(data_path),
        "output_directory": str(output_dir),
        "target": args.target,
        "model": "ExtraTreesRegressor",
        "n_outer_folds": len(folds),
        "n_repeats": args.n_repeats,
        "scoring": "neg_mean_absolute_error",
        "interpretation": [
            (
                "Permutation importance was calculated only on "
                "each held-out outer test fold."
            ),
            (
                "A positive value means shuffling the feature "
                "increased MAE and therefore harmed prediction."
            ),
            (
                "Correlated features can share or dilute "
                "permutation importance."
            ),
            (
                "Importance describes predictive contribution, "
                "not causality."
            ),
        ],
    }

    with open(
        output_dir
        / "extra_trees_explainability_manifest.json",
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
        "\n=== TOP 20 STABLE PERMUTATION FEATURES ==="
    )

    display_columns = [
        "feature",
        "mean_permutation_importance",
        "positive_folds",
        "stable_permutation_score",
        "metric_family",
        "arm",
    ]

    print(
        summary[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nOutputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()