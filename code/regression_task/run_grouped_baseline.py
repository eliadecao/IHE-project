from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4\grouped_cv"
    r"\dataset_v4_grouped_folds.csv"
)

TARGET_COLUMN = "target_score"

NON_FEATURE_COLUMNS = {
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
    "cv_fold",
}


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove redundant numeric features using training data only.

    This transformer is fitted separately inside each cross-validation
    training fold, preventing test-fold information leakage.
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)

        corr = X_df.corr(method="pearson", min_periods=5).abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape, dtype=bool), k=1)
        )

        self.columns_to_drop_ = [
            col
            for col in upper.columns
            if (upper[col] >= self.threshold).any()
        ]
        self.columns_to_keep_ = [
            col for col in X_df.columns
            if col not in self.columns_to_drop_
        ]

        if not self.columns_to_keep_:
            raise ValueError(
                "CorrelationFilter removed every feature."
            )

        return self

    def transform(self, X):
        X_df = self._as_dataframe(X)

        missing = [
            col for col in self.columns_to_keep_
            if col not in X_df.columns
        ]
        if missing:
            raise ValueError(
                "Missing columns during transform: "
                + ", ".join(missing[:10])
            )

        return X_df.loc[:, self.columns_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_to_keep_, dtype=object)

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if hasattr(X, "shape"):
            columns = [
                f"feature_{i}" for i in range(X.shape[1])
            ]
            return pd.DataFrame(X, columns=columns)

        return pd.DataFrame(X)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run participant-grouped out-of-fold regression baselines "
            "for M-GEARS prediction."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to dataset_v4_grouped_folds.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to input parent / baseline_regression",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COLUMN,
        choices=["target_score", "percentage_score"],
        help="Regression target.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.99,
        help="Absolute correlation threshold inside each fold.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=4,
        help="Grouped inner-CV folds for Ridge and Elastic Net tuning.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by GridSearchCV.",
    )
    return parser.parse_args()


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    explicit_exclusions = {
        # Identifiers and metadata
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
        "cv_fold",

        # Regression targets
        "target_score",
        "percentage_score",
        "target_fraction",

        # Label aggregation and quality information
        "target_score_std",
        "percentage_score_std",
        "n_unique_target_scores",
        "label_disagreement_flag",
        "n_label_records",
        "n_applicable_mgears_domains",
        "mgears_domain_score_sum",
        "target_minus_domain_sum",
        "inferred_max_score",
    }

    forbidden_prefixes = (
        "mgears_",
        "label_",
    )

    forbidden_name_fragments = (
        "target_score",
        "percentage_score",
        "domain_score",
        "mgears",
    )

    feature_cols = []

    for col in df.columns:
        lower = col.lower()

        if col in explicit_exclusions:
            continue

        if lower.startswith(forbidden_prefixes):
            continue

        if any(fragment in lower for fragment in forbidden_name_fragments):
            continue

        if lower.startswith("qc_"):
            # QC/setup variables are excluded from the primary model.
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No valid kinematic feature columns were found.")

    print("\nExcluded possible label/leakage columns:")
    excluded = [
        col for col in df.columns
        if col not in feature_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in excluded:
        print("  -", col)

    return feature_cols


def build_pipeline(model, corr_threshold: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "correlation_filter",
                CorrelationFilter(
                    threshold=corr_threshold
                ),
            ),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            ("model", model),
        ]
    )


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) < 3:
        return np.nan

    if np.unique(y_true).size < 2:
        return np.nan

    if np.unique(y_pred).size < 2:
        return np.nan

    result = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(result.statistic)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            math.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": safe_spearman(y_true, y_pred),
    }


def make_inner_cv(
    train_df: pd.DataFrame,
    requested_splits: int,
) -> GroupKFold:
    n_groups = int(
        train_df["participant_group"].nunique()
    )
    n_splits = min(requested_splits, n_groups)

    if n_splits < 2:
        raise ValueError(
            "Not enough participant groups for inner CV."
        )

    return GroupKFold(n_splits=n_splits)


def tune_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    inner_cv: GroupKFold,
    n_jobs: int,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
        groups=train_groups,
    )
    return search


def extract_coefficients(
    fitted_pipeline: Pipeline,
    model_name: str,
    fold: int,
) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    corr_filter = fitted_pipeline.named_steps[
        "correlation_filter"
    ]
    feature_names = list(
        corr_filter.get_feature_names_out()
    )

    coefficients = np.asarray(model.coef_).reshape(-1)

    if len(feature_names) != len(coefficients):
        warnings.warn(
            f"Could not align coefficients for {model_name}, "
            f"fold {fold}."
        )
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "model": model_name,
            "cv_fold": fold,
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )

    return result.sort_values(
        "abs_coefficient",
        ascending=False,
    )


def save_prediction_plot(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    target: str,
):
    subset = predictions[
        predictions["model"] == model_name
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        subset["y_true"],
        subset["y_pred"],
        alpha=0.75,
    )

    minimum = min(
        subset["y_true"].min(),
        subset["y_pred"].min(),
    )
    maximum = max(
        subset["y_true"].max(),
        subset["y_pred"].max(),
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )
    ax.set_xlabel(f"Observed {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"{model_name}: observed vs predicted"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"{model_name}_observed_vs_predicted.png",
        dpi=300,
    )
    plt.close(fig)


def save_residual_plot(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: Path,
):
    subset = predictions[
        predictions["model"] == model_name
    ].copy()

    residual = subset["y_true"] - subset["y_pred"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        subset["y_pred"],
        residual,
        alpha=0.75,
    )
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (observed - predicted)")
    ax.set_title(f"{model_name}: residual plot")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{model_name}_residuals.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir or (
        input_path.parent / "baseline_regression"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        args.target,
        "participant_group",
        "cv_fold",
        "task_clean",
        "file_name",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[args.target].isna().any():
        raise ValueError(
            f"{args.target} contains missing values."
        )

    if df["cv_fold"].isna().any():
        raise ValueError("cv_fold contains missing values.")

    feature_cols = choose_feature_columns(df)
    X = df[feature_cols].copy()
    y = pd.to_numeric(
        df[args.target],
        errors="raise",
    )

    outer_folds = sorted(
        df["cv_fold"].astype(int).unique()
    )

    model_specs = {
        "dummy_mean": {
            "model": DummyRegressor(
                strategy="mean"
            ),
            "grid": None,
        },
        "linear_regression": {
            "model": LinearRegression(),
            "grid": None,
        },
        "ridge": {
            "model": Ridge(
                max_iter=20000
            ),
            "grid": {
                "model__alpha": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                    1000.0,
                ]
            },
        },
        "elastic_net": {
            "model": ElasticNet(
                max_iter=50000,
                tol=1e-4,
                random_state=42,
            ),
            "grid": {
                "model__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
                "model__l1_ratio": [
                    0.1,
                    0.5,
                    0.9,
                ],
            },
        },
    }

    prediction_rows = []
    fold_metric_rows = []
    tuning_rows = []
    coefficient_tables = []
    selected_feature_rows = []

    print("\n=== GROUPED REGRESSION BASELINE ===")
    print(f"Input shape:          {df.shape}")
    print(f"Target:               {args.target}")
    print(f"Numeric features:     {len(feature_cols)}")
    print(f"Outer folds:          {outer_folds}")
    print(
        "Unique participants:  "
        f"{df['participant_group'].nunique()}"
    )

    for model_name, spec in model_specs.items():
        print(f"\n--- {model_name} ---")

        for fold in outer_folds:
            test_mask = (
                df["cv_fold"].astype(int) == fold
            )
            train_mask = ~test_mask

            X_train = X.loc[train_mask].copy()
            X_test = X.loc[test_mask].copy()
            y_train = y.loc[train_mask].copy()
            y_test = y.loc[test_mask].copy()

            train_groups = df.loc[
                train_mask,
                "participant_group",
            ].copy()

            pipeline = build_pipeline(
                clone(spec["model"]),
                corr_threshold=args.corr_threshold,
            )

            if spec["grid"] is None:
                fitted = pipeline.fit(
                    X_train,
                    y_train,
                )
                best_params = {}
                inner_best_mae = np.nan
            else:
                inner_cv = make_inner_cv(
                    df.loc[train_mask],
                    requested_splits=args.inner_splits,
                )

                search = tune_model(
                    pipeline=pipeline,
                    param_grid=spec["grid"],
                    X_train=X_train,
                    y_train=y_train,
                    train_groups=train_groups,
                    inner_cv=inner_cv,
                    n_jobs=args.n_jobs,
                )

                fitted = search.best_estimator_
                best_params = search.best_params_
                inner_best_mae = float(
                    -search.best_score_
                )

            y_pred = fitted.predict(X_test)

            fold_metrics = compute_metrics(
                y_test.to_numpy(),
                y_pred,
            )

            fold_metric_rows.append(
                {
                    "model": model_name,
                    "cv_fold": fold,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "n_train_participants": int(
                        df.loc[
                            train_mask,
                            "participant_group",
                        ].nunique()
                    ),
                    "n_test_participants": int(
                        df.loc[
                            test_mask,
                            "participant_group",
                        ].nunique()
                    ),
                    **fold_metrics,
                }
            )

            tuning_rows.append(
                {
                    "model": model_name,
                    "cv_fold": fold,
                    "inner_best_mae": inner_best_mae,
                    "best_parameters": json.dumps(
                        best_params,
                        sort_keys=True,
                    ),
                }
            )

            correlation_filter = fitted.named_steps[
                "correlation_filter"
            ]
            kept_features = (
                correlation_filter.columns_to_keep_
            )
            dropped_features = (
                correlation_filter.columns_to_drop_
            )

            selected_feature_rows.append(
                {
                    "model": model_name,
                    "cv_fold": fold,
                    "n_features_input": len(feature_cols),
                    "n_features_kept": len(kept_features),
                    "n_features_dropped_correlation": len(
                        dropped_features
                    ),
                    "kept_features": json.dumps(
                        kept_features
                    ),
                    "dropped_features": json.dumps(
                        dropped_features
                    ),
                }
            )

            coefficients = extract_coefficients(
                fitted,
                model_name=model_name,
                fold=fold,
            )
            if not coefficients.empty:
                coefficient_tables.append(coefficients)

            metadata_cols = [
                col for col in [
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
            fold_predictions["model"] = model_name
            fold_predictions["cv_fold"] = fold
            fold_predictions["y_true"] = (
                y_test.to_numpy()
            )
            fold_predictions["y_pred"] = y_pred
            fold_predictions["residual"] = (
                fold_predictions["y_true"]
                - fold_predictions["y_pred"]
            )
            fold_predictions["absolute_error"] = (
                fold_predictions["residual"].abs()
            )

            prediction_rows.extend(
                fold_predictions.to_dict(
                    orient="records"
                )
            )

            print(
                f"Fold {fold}: "
                f"MAE={fold_metrics['mae']:.3f}, "
                f"RMSE={fold_metrics['rmse']:.3f}, "
                f"R2={fold_metrics['r2']:.3f}, "
                f"rho={fold_metrics['spearman_rho']:.3f}, "
                f"features={len(kept_features)}"
            )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    selected_features_df = pd.DataFrame(
        selected_feature_rows
    )

    predictions.to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )
    fold_metrics_df.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
    )
    tuning_df.to_csv(
        output_dir / "hyperparameter_selection.csv",
        index=False,
    )
    selected_features_df.to_csv(
        output_dir / "fold_feature_selection.csv",
        index=False,
    )

    if coefficient_tables:
        all_coefficients = pd.concat(
            coefficient_tables,
            ignore_index=True,
        )
        all_coefficients.to_csv(
            output_dir / "fold_coefficients.csv",
            index=False,
        )

        coefficient_summary = (
            all_coefficients.groupby(
                ["model", "feature"],
                as_index=False,
            )
            .agg(
                folds_present=("cv_fold", "nunique"),
                mean_coefficient=(
                    "coefficient",
                    "mean",
                ),
                std_coefficient=(
                    "coefficient",
                    "std",
                ),
                mean_abs_coefficient=(
                    "abs_coefficient",
                    "mean",
                ),
            )
            .sort_values(
                [
                    "model",
                    "mean_abs_coefficient",
                ],
                ascending=[True, False],
            )
        )
        coefficient_summary.to_csv(
            output_dir
            / "coefficient_summary_across_folds.csv",
            index=False,
        )

    # Overall OOF metrics
    overall_rows = []
    for model_name in model_specs:
        subset = predictions[
            predictions["model"] == model_name
        ]
        metrics = compute_metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        overall_rows.append(
            {
                "model": model_name,
                "n_oof_predictions": len(subset),
                **metrics,
            }
        )

    overall_metrics = pd.DataFrame(
        overall_rows
    ).sort_values("mae")

    overall_metrics.to_csv(
        output_dir / "overall_oof_metrics.csv",
        index=False,
    )

    # Per-task OOF metrics
    task_rows = []
    for (model_name, task), subset in (
        predictions.groupby(
            ["model", "task_clean"]
        )
    ):
        metrics = compute_metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        task_rows.append(
            {
                "model": model_name,
                "task_clean": task,
                "n": len(subset),
                **metrics,
            }
        )

    task_metrics = pd.DataFrame(task_rows)
    task_metrics.to_csv(
        output_dir / "task_oof_metrics.csv",
        index=False,
    )

    # Per-role OOF metrics
    if "role" in predictions.columns:
        role_rows = []
        for (model_name, role), subset in (
            predictions.groupby(
                ["model", "role"]
            )
        ):
            metrics = compute_metrics(
                subset["y_true"].to_numpy(),
                subset["y_pred"].to_numpy(),
            )
            role_rows.append(
                {
                    "model": model_name,
                    "role": role,
                    "n": len(subset),
                    **metrics,
                }
            )

        pd.DataFrame(role_rows).to_csv(
            output_dir / "role_oof_metrics.csv",
            index=False,
        )

    # Plots
    for model_name in model_specs:
        save_prediction_plot(
            predictions,
            model_name,
            output_dir,
            target=args.target,
        )
        save_residual_plot(
            predictions,
            model_name,
            output_dir,
        )

    # Compact model-comparison plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = overall_metrics.sort_values("mae")
    ax.bar(
        ordered["model"],
        ordered["mae"],
    )
    ax.set_ylabel("Out-of-fold MAE")
    ax.set_xlabel("Model")
    ax.set_title(
        f"Grouped CV model comparison: {args.target}"
    )
    ax.tick_params(
        axis="x",
        rotation=25,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "model_comparison_mae.png",
        dpi=300,
    )
    plt.close(fig)

    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "target": args.target,
        "n_rows": int(len(df)),
        "n_participant_groups": int(
            df["participant_group"].nunique()
        ),
        "n_outer_folds": int(len(outer_folds)),
        "n_numeric_features_input": int(
            len(feature_cols)
        ),
        "correlation_threshold": float(
            args.corr_threshold
        ),
        "inner_grouped_cv_splits_requested": int(
            args.inner_splits
        ),
        "models": list(model_specs.keys()),
        "methodology": [
            "The outer test folds were frozen before modelling.",
            "All preprocessing was fitted using training-fold data only.",
            "Correlation filtering was performed inside each outer fold.",
            "Ridge and Elastic Net hyperparameters were selected using grouped inner cross-validation.",
            "All reported overall metrics use out-of-fold predictions.",
        ],
        "overall_metrics": overall_metrics.to_dict(
            orient="records"
        ),
    }

    with open(
        output_dir / "baseline_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== OVERALL OUT-OF-FOLD RESULTS ===")
    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()