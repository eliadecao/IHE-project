from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\dataset_v5\qc_v5\v4"
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

LEAKAGE_PREFIXES = (
    "mgears_",
    "label_",
)

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
            raise ValueError("CorrelationFilter removed every feature.")
        return self

    def transform(self, X):
        X_df = self._as_dataframe(X)
        return X_df.loc[:, self.columns_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_to_keep_, dtype=object)

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        columns = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped ablation experiments for M-GEARS regression."
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
        choices=["target_score", "percentage_score"],
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
        "--n-jobs",
        type=int,
        default=-1,
    )
    return parser.parse_args()


def is_numeric_feature_candidate(
    df: pd.DataFrame,
    col: str,
) -> bool:
    lower = col.lower()

    if col in BASE_EXCLUSIONS:
        return False
    if lower.startswith(LEAKAGE_PREFIXES):
        return False
    if any(fragment in lower for fragment in LEAKAGE_FRAGMENTS):
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False

    return True


def select_kinematic_features(
    df: pd.DataFrame,
    exclude_qc: bool,
) -> list[str]:
    features = []

    for col in df.columns:
        if not is_numeric_feature_candidate(df, col):
            continue

        lower = col.lower()

        if lower.startswith("qc_"):
            continue

        if exclude_qc and any(
            fragment in lower for fragment in QC_FRAGMENTS
        ):
            continue

        features.append(col)

    if not features:
        raise ValueError("No kinematic features were selected.")

    return features


def safe_spearman(y_true, y_pred) -> float:
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    return float(
        spearmanr(y_true, y_pred, nan_policy="omit").statistic
    )


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            math.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": safe_spearman(y_true, y_pred),
    }


def build_numeric_pipeline(
    model,
    corr_threshold: float,
) -> Pipeline:
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


def build_task_adjusted_pipeline(
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
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "task",
                categorical_pipeline,
                ["task_clean"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", model),
        ]
    )


def tune_and_fit(
    pipeline,
    param_grid,
    X_train,
    y_train,
    train_groups,
    inner_splits,
    n_jobs,
):
    n_groups = pd.Series(train_groups).nunique()
    inner_cv = GroupKFold(
        n_splits=min(inner_splits, n_groups)
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(
        X_train,
        y_train,
        groups=train_groups,
    )
    return search.best_estimator_, search.best_params_, -search.best_score_


def main():
    args = parse_args()

    output_dir = args.output_dir or (
        args.input.parent / "ablation_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
            f"Missing required columns: {sorted(missing)}"
        )

    y = pd.to_numeric(df[args.target], errors="raise")
    folds = sorted(df["cv_fold"].astype(int).unique())

    all_kinematic = select_kinematic_features(
        df,
        exclude_qc=False,
    )
    qc_excluded = select_kinematic_features(
        df,
        exclude_qc=True,
    )

    removed_qc = sorted(
        set(all_kinematic) - set(qc_excluded)
    )

    experiments = {
        "A_kinematics_all": {
            "features": all_kinematic,
            "include_task": False,
        },
        "B_kinematics_no_qc": {
            "features": qc_excluded,
            "include_task": False,
        },
        "C_kinematics_no_qc_plus_task": {
            "features": qc_excluded,
            "include_task": True,
        },
    }

    model_specs = {
        "ridge": {
            "model": Ridge(max_iter=30000),
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
                max_iter=200000,
                tol=1e-3,
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

    print("\n=== GROUPED ABLATION ANALYSIS ===")
    print(f"Input shape:              {df.shape}")
    print(f"Target:                   {args.target}")
    print(f"All kinematic features:   {len(all_kinematic)}")
    print(f"No-QC kinematic features: {len(qc_excluded)}")
    print(f"QC-like features removed: {len(removed_qc)}")

    if removed_qc:
        print("\nRemoved QC-like features:")
        for col in removed_qc:
            print(f"  - {col}")

    for experiment_name, experiment in experiments.items():
        feature_cols = experiment["features"]
        include_task = experiment["include_task"]

        print(f"\n=== {experiment_name} ===")
        print(f"Numeric features: {len(feature_cols)}")
        print(f"Task included:    {include_task}")

        if include_task:
            X_all = df[feature_cols + ["task_clean"]].copy()
        else:
            X_all = df[feature_cols].copy()

        for model_name, spec in model_specs.items():
            print(f"\n--- {model_name} ---")

            for fold in folds:
                test_mask = df["cv_fold"].astype(int) == fold
                train_mask = ~test_mask

                X_train = X_all.loc[train_mask].copy()
                X_test = X_all.loc[test_mask].copy()
                y_train = y.loc[train_mask].copy()
                y_test = y.loc[test_mask].copy()
                train_groups = df.loc[
                    train_mask,
                    "participant_group",
                ]

                if include_task:
                    pipeline = build_task_adjusted_pipeline(
                        numeric_features=feature_cols,
                        model=clone(spec["model"]),
                        corr_threshold=args.corr_threshold,
                    )
                else:
                    pipeline = build_numeric_pipeline(
                        model=clone(spec["model"]),
                        corr_threshold=args.corr_threshold,
                    )

                fitted, best_params, inner_mae = tune_and_fit(
                    pipeline=pipeline,
                    param_grid=spec["grid"],
                    X_train=X_train,
                    y_train=y_train,
                    train_groups=train_groups,
                    inner_splits=args.inner_splits,
                    n_jobs=args.n_jobs,
                )

                y_pred = fitted.predict(X_test)
                fold_result = metrics(
                    y_test.to_numpy(),
                    y_pred,
                )

                fold_metric_rows.append(
                    {
                        "experiment": experiment_name,
                        "model": model_name,
                        "cv_fold": fold,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "n_numeric_features": len(feature_cols),
                        "task_included": include_task,
                        **fold_result,
                    }
                )

                tuning_rows.append(
                    {
                        "experiment": experiment_name,
                        "model": model_name,
                        "cv_fold": fold,
                        "inner_best_mae": float(inner_mae),
                        "best_parameters": json.dumps(
                            best_params,
                            sort_keys=True,
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
                fold_predictions["experiment"] = experiment_name
                fold_predictions["model"] = model_name
                fold_predictions["cv_fold"] = fold
                fold_predictions["y_true"] = y_test.to_numpy()
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
                    f"MAE={fold_result['mae']:.3f}, "
                    f"RMSE={fold_result['rmse']:.3f}, "
                    f"R2={fold_result['r2']:.3f}, "
                    f"rho={fold_result['spearman_rho']:.3f}"
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    tuning_df = pd.DataFrame(tuning_rows)

    predictions.to_csv(
        output_dir / "ablation_oof_predictions.csv",
        index=False,
    )
    fold_metrics_df.to_csv(
        output_dir / "ablation_fold_metrics.csv",
        index=False,
    )
    tuning_df.to_csv(
        output_dir / "ablation_hyperparameters.csv",
        index=False,
    )

    overall_rows = []
    for (experiment, model), subset in predictions.groupby(
        ["experiment", "model"]
    ):
        result = metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        overall_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "n_oof_predictions": len(subset),
                **result,
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values(
        ["mae", "rmse"]
    )
    overall.to_csv(
        output_dir / "ablation_overall_metrics.csv",
        index=False,
    )

    task_rows = []
    for (experiment, model, task), subset in predictions.groupby(
        ["experiment", "model", "task_clean"]
    ):
        result = metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        task_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "task_clean": task,
                "n": len(subset),
                **result,
            }
        )

    pd.DataFrame(task_rows).to_csv(
        output_dir / "ablation_task_metrics.csv",
        index=False,
    )

    comparison = overall.copy()
    best_baseline = comparison[
        comparison["experiment"] == "A_kinematics_all"
    ][
        ["model", "mae", "rmse", "r2", "spearman_rho"]
    ].rename(
        columns={
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "r2": "baseline_r2",
            "spearman_rho": "baseline_spearman_rho",
        }
    )

    comparison = comparison.merge(
        best_baseline,
        on="model",
        how="left",
    )
    comparison["mae_change_vs_A"] = (
        comparison["mae"] - comparison["baseline_mae"]
    )
    comparison["r2_change_vs_A"] = (
        comparison["r2"] - comparison["baseline_r2"]
    )
    comparison.to_csv(
        output_dir / "ablation_comparison_vs_A.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = overall.copy()
    labels = (
        plot_df["experiment"]
        + "\n"
        + plot_df["model"]
    )
    ax.bar(labels, plot_df["mae"])
    ax.set_ylabel("Out-of-fold MAE")
    ax.set_xlabel("Experiment and model")
    ax.set_title("Grouped ablation comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(
        output_dir / "ablation_mae_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    manifest = {
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "target": args.target,
        "n_rows": int(len(df)),
        "n_participant_groups": int(
            df["participant_group"].nunique()
        ),
        "n_all_kinematic_features": len(all_kinematic),
        "n_no_qc_kinematic_features": len(qc_excluded),
        "removed_qc_like_features": removed_qc,
        "experiments": {
            "A_kinematics_all": (
                "All leakage-safe kinematic features, excluding qc_ columns."
            ),
            "B_kinematics_no_qc": (
                "As A, additionally excluding rotation/data-quality proxy features."
            ),
            "C_kinematics_no_qc_plus_task": (
                "As B, with task_clean one-hot encoded inside each fold."
            ),
        },
        "methodology": [
            "Outer folds were fixed before modelling.",
            "All preprocessing was learned using training-fold data only.",
            "Hyperparameters were tuned with grouped inner cross-validation.",
            "Task one-hot encoding was performed inside the pipeline.",
            "Overall metrics were calculated from out-of-fold predictions.",
        ],
        "overall_results": overall.to_dict(
            orient="records"
        ),
    }

    with open(
        output_dir / "ablation_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== OVERALL ABLATION RESULTS ===")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()