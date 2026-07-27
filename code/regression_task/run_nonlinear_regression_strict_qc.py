from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4"
    r"\grouped_cv\dataset_v4_grouped_folds.csv"
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
    # Rotation-matrix numerical quality / validity proxies
    "orthogonality",
    "determinant",
    "rotation_valid",
    "invalid_rotation",
    "rotation_quality",
    "rotation_error",

    # Missingness / reconstruction / acquisition-quality proxies
    "nan_fraction",
    "missing_fraction",
    "interpolation_fraction",
    "reconstruction_error",
    "timestamp_error",
)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Fit correlation pruning on the current training data only."""

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
            "Evaluate nonlinear tree-based M-GEARS regressors "
            "using frozen participant-grouped outer folds after "
            "strictly excluding QC and rotation-validity proxies."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
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
        "--inner-splits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help=(
            "Randomized-search candidates per model and outer fold."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
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
            "No leakage-safe kinematic features found."
        )

    return features


def build_pipeline(
    numeric_features: list[str],
    model,
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


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if (
        np.unique(y_true).size < 2
        or np.unique(y_pred).size < 2
    ):
        return np.nan

    return float(
        spearmanr(
            y_true,
            y_pred,
            nan_policy="omit",
        ).statistic
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "spearman_rho": safe_spearman(
            y_true,
            y_pred,
        ),
    }


def get_model_specs(
    random_state: int,
):
    return {
        "random_forest": {
            "model": RandomForestRegressor(
                random_state=random_state,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": randint(
                    250,
                    801,
                ),
                "model__max_depth": [
                    None,
                    3,
                    5,
                    8,
                    12,
                ],
                "model__min_samples_split": randint(
                    2,
                    11,
                ),
                "model__min_samples_leaf": randint(
                    1,
                    7,
                ),
                "model__max_features": [
                    0.3,
                    0.5,
                    0.7,
                    "sqrt",
                ],
            },
        },
        "extra_trees": {
            "model": ExtraTreesRegressor(
                random_state=random_state,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": randint(
                    250,
                    801,
                ),
                "model__max_depth": [
                    None,
                    3,
                    5,
                    8,
                    12,
                ],
                "model__min_samples_split": randint(
                    2,
                    11,
                ),
                "model__min_samples_leaf": randint(
                    1,
                    7,
                ),
                "model__max_features": [
                    0.3,
                    0.5,
                    0.7,
                    "sqrt",
                ],
            },
        },
        "hist_gradient_boosting": {
            "model": HistGradientBoostingRegressor(
                random_state=random_state,
                early_stopping=True,
            ),
            "params": {
                "model__learning_rate": loguniform(
                    0.01,
                    0.3,
                ),
                "model__max_iter": randint(
                    100,
                    501,
                ),
                "model__max_leaf_nodes": randint(
                    5,
                    32,
                ),
                "model__max_depth": [
                    None,
                    2,
                    3,
                    5,
                ],
                "model__min_samples_leaf": randint(
                    5,
                    31,
                ),
                "model__l2_regularization": loguniform(
                    1e-3,
                    100.0,
                ),
            },
        },
    }


def extract_feature_importance(
    fitted_pipeline: Pipeline,
    model_name: str,
    fold: int,
) -> pd.DataFrame:
    model = fitted_pipeline.named_steps[
        "model"
    ]

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return pd.DataFrame()

    preprocess = fitted_pipeline.named_steps[
        "preprocess"
    ]

    feature_names = preprocess.get_feature_names_out()
    importances = np.asarray(
        model.feature_importances_
    )

    if len(feature_names) != len(importances):
        return pd.DataFrame()

    return (
        pd.DataFrame(
            {
                "model": model_name,
                "cv_fold": fold,
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )


def main():
    args = parse_args()

    output_dir = args.output_dir or (
        args.input.parent
        / "nonlinear_regression_strict_qc"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}"
        )

    df = pd.read_csv(args.input)

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

    numeric_features = select_clean_features(df)

    excluded_qc_columns = [
        col
        for col in df.columns
        if any(
            fragment in col.lower()
            for fragment in QC_FRAGMENTS
        )
    ]

    print("\nStrict QC proxy columns excluded:")
    if excluded_qc_columns:
        for col in sorted(excluded_qc_columns):
            print(f"  - {col}")
    else:
        print("  (none matched)")

    X = df[
        numeric_features + ["task_clean"]
    ].copy()
    y = pd.to_numeric(
        df[args.target],
        errors="raise",
    )

    folds = sorted(
        df["cv_fold"]
        .astype(int)
        .unique()
    )

    model_specs = get_model_specs(
        args.random_state
    )

    prediction_rows = []
    fold_metric_rows = []
    tuning_rows = []
    importance_tables = []

    print(
        "\n=== NONLINEAR GROUPED REGRESSION ==="
    )
    print(f"Input shape:          {df.shape}")
    print(f"Target:               {args.target}")
    print(
        f"Clean numeric features: {len(numeric_features)}"
    )
    print(
        f"Participant groups:   "
        f"{df['participant_group'].nunique()}"
    )
    print(
        f"Outer folds:          {folds}"
    )
    print(
        f"Random search n_iter: {args.n_iter}"
    )

    for model_name, spec in model_specs.items():
        print(f"\n=== {model_name} ===")

        for fold in folds:
            test_mask = (
                df["cv_fold"].astype(int)
                == fold
            )
            train_mask = ~test_mask

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

            train_groups = df.loc[
                train_mask,
                "participant_group",
            ]

            inner_cv = GroupKFold(
                n_splits=min(
                    args.inner_splits,
                    train_groups.nunique(),
                )
            )

            pipeline = build_pipeline(
                numeric_features=numeric_features,
                model=spec["model"],
                corr_threshold=args.corr_threshold,
            )

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=spec["params"],
                n_iter=args.n_iter,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                random_state=(
                    args.random_state + fold
                ),
                n_jobs=args.n_jobs,
                refit=True,
                return_train_score=False,
                error_score="raise",
            )

            search.fit(
                X_train,
                y_train,
                groups=train_groups,
            )

            fitted = search.best_estimator_
            y_pred = fitted.predict(X_test)

            fold_result = compute_metrics(
                y_test.to_numpy(),
                y_pred,
            )

            fold_metric_rows.append(
                {
                    "model": model_name,
                    "cv_fold": fold,
                    "n_train": int(
                        train_mask.sum()
                    ),
                    "n_test": int(
                        test_mask.sum()
                    ),
                    "inner_best_mae": float(
                        -search.best_score_
                    ),
                    **fold_result,
                }
            )

            tuning_rows.append(
                {
                    "model": model_name,
                    "cv_fold": fold,
                    "best_parameters": json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
                    "inner_best_mae": float(
                        -search.best_score_
                    ),
                }
            )

            metadata_cols = [
                col
                for col in [
                    "file_name",
                    "role",
                    "participant_id",
                    "participant_group",
                    "task_clean",
                    "trial",
                ]
                if col in df.columns
            ]

            fold_predictions = df.loc[
                test_mask,
                metadata_cols,
            ].copy()
            fold_predictions["model"] = (
                model_name
            )
            fold_predictions["cv_fold"] = fold
            fold_predictions["y_true"] = (
                y_test.to_numpy()
            )
            fold_predictions["y_pred"] = y_pred
            fold_predictions["residual"] = (
                fold_predictions["y_true"]
                - fold_predictions["y_pred"]
            )
            fold_predictions[
                "absolute_error"
            ] = fold_predictions[
                "residual"
            ].abs()

            prediction_rows.extend(
                fold_predictions.to_dict(
                    orient="records"
                )
            )

            importance = extract_feature_importance(
                fitted_pipeline=fitted,
                model_name=model_name,
                fold=fold,
            )
            if not importance.empty:
                importance_tables.append(
                    importance
                )

            print(
                f"Fold {fold}: "
                f"MAE={fold_result['mae']:.3f}, "
                f"RMSE={fold_result['rmse']:.3f}, "
                f"R2={fold_result['r2']:.3f}, "
                f"rho={fold_result['spearman_rho']:.3f}"
            )

    predictions = pd.DataFrame(
        prediction_rows
    )
    fold_metrics = pd.DataFrame(
        fold_metric_rows
    )
    tuning = pd.DataFrame(
        tuning_rows
    )

    predictions.to_csv(
        output_dir
        / "nonlinear_oof_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir
        / "nonlinear_fold_metrics.csv",
        index=False,
    )
    tuning.to_csv(
        output_dir
        / "nonlinear_hyperparameters.csv",
        index=False,
    )

    overall_rows = []
    for model_name, subset in predictions.groupby(
        "model"
    ):
        result = compute_metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        overall_rows.append(
            {
                "model": model_name,
                "n_oof_predictions": len(
                    subset
                ),
                **result,
            }
        )

    overall = (
        pd.DataFrame(overall_rows)
        .sort_values(
            ["mae", "rmse"]
        )
    )
    overall.to_csv(
        output_dir
        / "nonlinear_overall_metrics.csv",
        index=False,
    )

    task_rows = []
    for (
        model_name,
        task,
    ), subset in predictions.groupby(
        ["model", "task_clean"]
    ):
        result = compute_metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        task_rows.append(
            {
                "model": model_name,
                "task_clean": task,
                "n": len(subset),
                **result,
            }
        )

    pd.DataFrame(
        task_rows
    ).to_csv(
        output_dir
        / "nonlinear_task_metrics.csv",
        index=False,
    )

    if importance_tables:
        all_importance = pd.concat(
            importance_tables,
            ignore_index=True,
        )
        all_importance.to_csv(
            output_dir
            / "tree_fold_feature_importance.csv",
            index=False,
        )

        importance_summary = (
            all_importance.groupby(
                ["model", "feature"],
                as_index=False,
            )
            .agg(
                folds_present=(
                    "cv_fold",
                    "nunique",
                ),
                mean_importance=(
                    "importance",
                    "mean",
                ),
                std_importance=(
                    "importance",
                    "std",
                ),
                max_importance=(
                    "importance",
                    "max",
                ),
            )
            .sort_values(
                [
                    "model",
                    "mean_importance",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )
        importance_summary.to_csv(
            output_dir
            / "tree_feature_importance_summary.csv",
            index=False,
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    plot_df = overall.sort_values(
        "mae"
    )
    ax.bar(
        plot_df["model"],
        plot_df["mae"],
    )
    ax.set_ylabel(
        "Out-of-fold MAE"
    )
    ax.set_xlabel("Model")
    ax.set_title(
        "Nonlinear grouped regression comparison"
    )
    ax.tick_params(
        axis="x",
        rotation=20,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "nonlinear_model_comparison_mae.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    manifest = {
        "input_path": str(
            args.input
        ),
        "output_directory": str(
            output_dir
        ),
        "target": args.target,
        "n_rows": int(len(df)),
        "n_participant_groups": int(
            df[
                "participant_group"
            ].nunique()
        ),
        "n_clean_numeric_features": int(
            len(numeric_features)
        ),
        "task_included": True,
        "n_random_search_iterations": int(
            args.n_iter
        ),
        "models": list(
            model_specs.keys()
        ),
        "methodology": [
            (
                "The previously frozen participant-grouped "
                "outer folds were used unchanged."
            ),
            (
                "Correlation filtering, imputation and task "
                "encoding were fitted on training data only."
            ),
            (
                "Hyperparameters were selected by grouped "
                "inner cross-validation using MAE."
            ),
            (
                "Overall results were calculated from "
                "out-of-fold predictions."
            ),
        ],
        "overall_results": overall.to_dict(
            orient="records"
        ),
    }

    with open(
        output_dir
        / "nonlinear_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== OVERALL NONLINEAR RESULTS ==="
    )
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(
        f"\nOutputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()