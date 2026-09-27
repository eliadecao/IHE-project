from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\qc_v7\v4"
    r"\grouped_cv\dataset_v4_grouped_folds.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\qc_v7\v4"
    r"\grouped_cv\mgears_domain_regression"
)

DIRECT_TOTAL_BENCHMARK_MAE = 2.5041055670190966
DIRECT_TOTAL_BENCHMARK_RMSE = 3.072861274544127
DIRECT_TOTAL_BENCHMARK_R2 = 0.5377908072879796

RANDOM_STATE = 42
INNER_CV_SPLITS = 4
RANDOM_SEARCH_ITER = 12
CORRELATION_THRESHOLD = 0.98

DOMAIN_MIN_SCORE = 1
DOMAIN_MAX_SCORE = 5

DOMAIN_COLUMNS = [
    "mgears_depth_perception",
    "mgears_dexterity_with_multiple_wristed_instruments",
    "mgears_efficiency_flow_of_operation",
    "mgears_force_sensitivity_and_tissue_handling",
    "mgears_master_manipulator_workspace_robotic_control",
    "mgears_overall_performance_quality_of_the_final_product",
]

TARGET_COLUMN = "target_score"
FOLD_COLUMN = "cv_fold"


# ============================================================
# LEAKAGE-SAFE CORRELATION FILTER
# ============================================================

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2-D matrix.")

        if X.shape[1] <= 1:
            self.keep_indices_ = np.arange(X.shape[1])
            return self

        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        drop = set()

        for j in range(1, corr.shape[1]):
            for i in range(j):
                if i in drop:
                    continue

                if abs(corr[i, j]) >= self.threshold:
                    drop.add(j)
                    break

        self.keep_indices_ = np.array(
            [i for i in range(X.shape[1]) if i not in drop],
            dtype=int,
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


# ============================================================
# METRICS / HELPERS
# ============================================================

@dataclass
class MetricResult:
    n: int
    mae: float
    rmse: float
    r2: float
    spearman_rho: float


def safe_spearman(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan

    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan

    result = spearmanr(
        y_true,
        y_pred,
        nan_policy="omit",
    )

    return float(result.statistic)


def regression_metrics(y_true, y_pred) -> MetricResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return MetricResult(
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    if len(y_true) >= 2 and np.std(y_true) > 0:
        r2 = r2_score(
            y_true,
            y_pred,
        )
    else:
        r2 = np.nan

    rho = safe_spearman(
        y_true,
        y_pred,
    )

    return MetricResult(
        n=int(len(y_true)),
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        spearman_rho=float(rho)
        if np.isfinite(rho)
        else np.nan,
    )


def clip_round(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.rint(values),
        DOMAIN_MIN_SCORE,
        DOMAIN_MAX_SCORE,
    )


def make_participant_group(df: pd.DataFrame) -> pd.Series:
    if "participant_group" in df.columns:
        return df["participant_group"].astype(str)

    required = {
        "role",
        "participant_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Need either participant_group or role + participant_id. "
            f"Missing: {sorted(missing)}"
        )

    return (
        df["role"].astype(str)
        + "__participant_"
        + df["participant_id"].astype(str)
    )


def choose_numeric_features(df: pd.DataFrame) -> list[str]:
    excluded_exact = {
        TARGET_COLUMN,
        "percentage_score",
        "target_fraction",
        "mgears_domain_score_sum",
        "n_applicable_mgears_domains",
        "inferred_max_score",

        FOLD_COLUMN,
        "participant_id",
        "trial",
        "file_name",
        "fk_file_name",
        "role",
        "task",
        "task_clean",
        "licence",
        "date",
        "session_licence",
        "session_datetime",
        "participant_group",

        "n_label_records",
        "target_score_std",
        "percentage_score_std",
        "domain_disagreement_max_std",
        "label_disagreement_flag",
    }

    features = []

    for col in df.columns:
        if col in excluded_exact:
            continue

        if col.startswith("mgears_"):
            continue

        lower = col.lower()

        if any(
            token in lower
            for token in [
                "target_score",
                "percentage_score",
                "label_disagreement",
                "domain_score_sum",
                "applicable_mgears",
                "inferred_max_score",
            ]
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            features.append(col)

    if not features:
        raise ValueError(
            "No numeric modelling features remain "
            "after leakage exclusions."
        )

    return features


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "variance",
            VarianceThreshold(
                threshold=0.0
            ),
        ),
        (
            "correlation",
            CorrelationFilter(
                threshold=CORRELATION_THRESHOLD
            ),
        ),
        (
            "model",
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [
        300,
        500,
        800,
        1200,
    ],
    "model__max_depth": [
        None,
        6,
        10,
        15,
        20,
    ],
    "model__min_samples_split": [
        2,
        3,
        4,
        6,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        3,
        4,
    ],
    "model__max_features": [
        1.0,
        "sqrt",
        0.5,
        0.75,
    ],
    "model__bootstrap": [
        False,
        True,
    ],
}


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            INPUT_CSV
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        "\n=== SIX-DOMAIN M-GEARS REGRESSION ==="
    )
    print(
        f"Input shape:         {df.shape}"
    )

    required = set(
        DOMAIN_COLUMNS
        + [
            TARGET_COLUMN,
            FOLD_COLUMN,
        ]
    )

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["participant_group"] = (
        make_participant_group(df)
    )

    features = choose_numeric_features(
        df
    )

    print(
        f"Numeric X features:  {len(features)}"
    )
    print(
        f"Participants:        "
        f"{df['participant_group'].nunique()}"
    )
    print(
        f"Outer folds:         "
        f"{sorted(df[FOLD_COLUMN].dropna().unique())}"
    )

    pd.DataFrame({
        "feature": features
    }).to_csv(
        OUTPUT_DIR
        / "domain_model_input_features.csv",
        index=False,
    )

    keep_cols = [
        "participant_group",
        FOLD_COLUMN,
        TARGET_COLUMN,
    ]

    for col in [
        "file_name",
        "task_clean",
        "role",
        "participant_id",
        "trial",
    ]:
        if col in df.columns:
            keep_cols.append(col)

    keep_cols += DOMAIN_COLUMNS

    oof = df[
        keep_cols
    ].copy()

    domain_metric_rows = []
    fold_metric_rows = []
    hyperparameter_rows = []

    for domain_index, domain in enumerate(
        DOMAIN_COLUMNS,
        start=1,
    ):
        print(
            "\n"
            + "=" * 78
        )
        print(
            f"DOMAIN {domain_index}/6: {domain}"
        )
        print(
            "=" * 78
        )

        raw_col = (
            f"pred_raw__{domain}"
        )

        rounded_col = (
            f"pred_rounded__{domain}"
        )

        oof[raw_col] = np.nan
        oof[rounded_col] = np.nan

        available_mask = (
            pd.to_numeric(
                df[domain],
                errors="coerce",
            ).notna()
        )

        print(
            f"Available labels: "
            f"{int(available_mask.sum())}/{len(df)}"
        )

        for fold in sorted(
            df[FOLD_COLUMN]
            .dropna()
            .unique()
        ):
            train_mask = (
                (df[FOLD_COLUMN] != fold)
                & available_mask
            )

            test_mask = (
                (df[FOLD_COLUMN] == fold)
                & available_mask
            )

            train_df = df.loc[
                train_mask
            ]

            test_df = df.loc[
                test_mask
            ]

            if train_df.empty or test_df.empty:
                print(
                    f"Fold {fold}: skipped "
                    f"(train={len(train_df)}, "
                    f"test={len(test_df)})"
                )
                continue

            X_train = train_df[
                features
            ]

            y_train = pd.to_numeric(
                train_df[domain],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            X_test = test_df[
                features
            ]

            y_test = pd.to_numeric(
                test_df[domain],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            train_groups = (
                train_df[
                    "participant_group"
                ].to_numpy()
            )

            n_unique_train_groups = (
                pd.Series(
                    train_groups
                ).nunique()
            )

            inner_splits = min(
                INNER_CV_SPLITS,
                n_unique_train_groups,
            )

            if inner_splits < 2:
                raise ValueError(
                    f"Not enough training groups "
                    f"for {domain}, fold {fold}"
                )

            inner_cv = GroupKFold(
                n_splits=inner_splits
            )

            search = RandomizedSearchCV(
                estimator=build_pipeline(),
                param_distributions=
                    PARAM_DISTRIBUTIONS,
                n_iter=RANDOM_SEARCH_ITER,
                scoring=
                    "neg_mean_absolute_error",
                cv=inner_cv,
                random_state=
                    RANDOM_STATE
                    + int(fold)
                    + domain_index * 100,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )

            search.fit(
                X_train,
                y_train,
                groups=train_groups,
            )

            pred_raw = (
                search.predict(
                    X_test
                )
            )

            pred_rounded = (
                clip_round(
                    pred_raw
                )
            )

            oof.loc[
                test_df.index,
                raw_col,
            ] = pred_raw

            oof.loc[
                test_df.index,
                rounded_col,
            ] = pred_rounded

            raw_metrics = (
                regression_metrics(
                    y_test,
                    pred_raw,
                )
            )

            rounded_metrics = (
                regression_metrics(
                    y_test,
                    pred_rounded,
                )
            )

            true_integer = (
                clip_round(
                    y_test
                )
            )

            exact_accuracy = (
                accuracy_score(
                    true_integer,
                    pred_rounded,
                )
            )

            within_one_accuracy = float(
                np.mean(
                    np.abs(
                        true_integer
                        - pred_rounded
                    ) <= 1
                )
            )

            try:
                qwk = (
                    cohen_kappa_score(
                        true_integer.astype(int),
                        pred_rounded.astype(int),
                        weights="quadratic",
                        labels=list(
                            range(
                                DOMAIN_MIN_SCORE,
                                DOMAIN_MAX_SCORE + 1,
                            )
                        ),
                    )
                )
            except Exception:
                qwk = np.nan

            print(
                f"Fold {int(fold)}: "
                f"raw MAE="
                f"{raw_metrics.mae:.3f}, "
                f"rounded MAE="
                f"{rounded_metrics.mae:.3f}, "
                f"exact="
                f"{exact_accuracy:.3f}, "
                f"within±1="
                f"{within_one_accuracy:.3f}"
            )

            fold_metric_rows.append({
                "domain": domain,
                "fold": int(fold),
                "n_test":
                    raw_metrics.n,
                "raw_mae":
                    raw_metrics.mae,
                "raw_rmse":
                    raw_metrics.rmse,
                "raw_r2":
                    raw_metrics.r2,
                "raw_spearman_rho":
                    raw_metrics.spearman_rho,
                "rounded_mae":
                    rounded_metrics.mae,
                "rounded_rmse":
                    rounded_metrics.rmse,
                "rounded_r2":
                    rounded_metrics.r2,
                "rounded_spearman_rho":
                    rounded_metrics.spearman_rho,
                "exact_accuracy":
                    float(exact_accuracy),
                "within_one_accuracy":
                    float(within_one_accuracy),
                "quadratic_weighted_kappa":
                    float(qwk)
                    if np.isfinite(qwk)
                    else np.nan,
            })

            hyperparameter_rows.append({
                "domain": domain,
                "fold": int(fold),
                "best_inner_cv_mae":
                    float(
                        -search.best_score_
                    ),
                "best_params":
                    json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
            })

        valid_domain_oof = (
            pd.to_numeric(
                oof[domain],
                errors="coerce",
            ).notna()
            &
            pd.to_numeric(
                oof[raw_col],
                errors="coerce",
            ).notna()
        )

        y_true = pd.to_numeric(
            oof.loc[
                valid_domain_oof,
                domain,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        pred_raw = pd.to_numeric(
            oof.loc[
                valid_domain_oof,
                raw_col,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        pred_rounded = pd.to_numeric(
            oof.loc[
                valid_domain_oof,
                rounded_col,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        raw_metrics = (
            regression_metrics(
                y_true,
                pred_raw,
            )
        )

        rounded_metrics = (
            regression_metrics(
                y_true,
                pred_rounded,
            )
        )

        true_integer = (
            clip_round(
                y_true
            )
        )

        exact_accuracy = (
            accuracy_score(
                true_integer,
                pred_rounded,
            )
        )

        within_one_accuracy = float(
            np.mean(
                np.abs(
                    true_integer
                    - pred_rounded
                ) <= 1
            )
        )

        try:
            qwk = (
                cohen_kappa_score(
                    true_integer.astype(int),
                    pred_rounded.astype(int),
                    weights="quadratic",
                    labels=list(
                        range(
                            DOMAIN_MIN_SCORE,
                            DOMAIN_MAX_SCORE + 1,
                        )
                    ),
                )
            )
        except Exception:
            qwk = np.nan

        domain_metric_rows.append({
            "domain": domain,
            "n_oof":
                raw_metrics.n,
            "raw_mae":
                raw_metrics.mae,
            "raw_rmse":
                raw_metrics.rmse,
            "raw_r2":
                raw_metrics.r2,
            "raw_spearman_rho":
                raw_metrics.spearman_rho,
            "rounded_mae":
                rounded_metrics.mae,
            "rounded_rmse":
                rounded_metrics.rmse,
            "rounded_r2":
                rounded_metrics.r2,
            "rounded_spearman_rho":
                rounded_metrics.spearman_rho,
            "exact_accuracy":
                float(exact_accuracy),
            "within_one_accuracy":
                float(within_one_accuracy),
            "quadratic_weighted_kappa":
                float(qwk)
                if np.isfinite(qwk)
                else np.nan,
        })

        print(
            "\nOverall OOF:"
        )
        print(
            f"  raw:     "
            f"MAE={raw_metrics.mae:.4f}, "
            f"RMSE={raw_metrics.rmse:.4f}, "
            f"R2={raw_metrics.r2:.4f}, "
            f"rho={raw_metrics.spearman_rho:.4f}"
        )
        print(
            f"  rounded: "
            f"MAE={rounded_metrics.mae:.4f}, "
            f"exact={exact_accuracy:.4f}, "
            f"within±1="
            f"{within_one_accuracy:.4f}, "
            f"QWK={qwk:.4f}"
        )

    # ========================================================
    # RECONSTRUCT TOTAL FROM THE SIX DOMAIN PREDICTIONS
    # ========================================================

    raw_pred_columns = [
        f"pred_raw__{domain}"
        for domain in DOMAIN_COLUMNS
    ]

    rounded_pred_columns = [
        f"pred_rounded__{domain}"
        for domain in DOMAIN_COLUMNS
    ]

    applicability = (
        oof[DOMAIN_COLUMNS]
        .notna()
    )

    raw_matrix = (
        oof[raw_pred_columns]
        .to_numpy(dtype=float)
    )

    rounded_matrix = (
        oof[rounded_pred_columns]
        .to_numpy(dtype=float)
    )

    applicability_matrix = (
        applicability
        .to_numpy(dtype=bool)
    )

    raw_masked = np.where(
        applicability_matrix,
        raw_matrix,
        np.nan,
    )

    rounded_masked = np.where(
        applicability_matrix,
        rounded_matrix,
        np.nan,
    )

    oof[
        "pred_total_domain_raw_sum"
    ] = np.nansum(
        raw_masked,
        axis=1,
    )

    oof[
        "pred_total_domain_rounded_sum"
    ] = np.nansum(
        rounded_masked,
        axis=1,
    )

    oof[
        "n_domains_used_for_total"
    ] = applicability.sum(
        axis=1
    ).astype(int)

    total_valid = (
        pd.to_numeric(
            oof[TARGET_COLUMN],
            errors="coerce",
        ).notna()
        &
        np.isfinite(
            oof[
                "pred_total_domain_raw_sum"
            ]
        )
    )

    total_true = pd.to_numeric(
        oof.loc[
            total_valid,
            TARGET_COLUMN,
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    total_pred_raw = pd.to_numeric(
        oof.loc[
            total_valid,
            "pred_total_domain_raw_sum",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    total_pred_rounded = pd.to_numeric(
        oof.loc[
            total_valid,
            "pred_total_domain_rounded_sum",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    total_raw_metrics = (
        regression_metrics(
            total_true,
            total_pred_raw,
        )
    )

    total_rounded_metrics = (
        regression_metrics(
            total_true,
            total_pred_rounded,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    domain_metrics_df = pd.DataFrame(
        domain_metric_rows
    )

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    hyperparameters_df = pd.DataFrame(
        hyperparameter_rows
    )

    total_comparison = pd.DataFrame([
        {
            "approach":
                "direct_total_extra_trees_benchmark",
            "n": len(df),
            "mae":
                DIRECT_TOTAL_BENCHMARK_MAE,
            "rmse":
                DIRECT_TOTAL_BENCHMARK_RMSE,
            "r2":
                DIRECT_TOTAL_BENCHMARK_R2,
            "spearman_rho":
                np.nan,
        },
        {
            "approach":
                "six_domain_raw_sum",
            "n":
                total_raw_metrics.n,
            "mae":
                total_raw_metrics.mae,
            "rmse":
                total_raw_metrics.rmse,
            "r2":
                total_raw_metrics.r2,
            "spearman_rho":
                total_raw_metrics.spearman_rho,
        },
        {
            "approach":
                "six_domain_rounded_sum",
            "n":
                total_rounded_metrics.n,
            "mae":
                total_rounded_metrics.mae,
            "rmse":
                total_rounded_metrics.rmse,
            "r2":
                total_rounded_metrics.r2,
            "spearman_rho":
                total_rounded_metrics.spearman_rho,
        },
    ])

    oof.to_csv(
        OUTPUT_DIR
        / "domain_oof_predictions.csv",
        index=False,
    )

    domain_metrics_df.to_csv(
        OUTPUT_DIR
        / "domain_overall_metrics.csv",
        index=False,
    )

    fold_metrics_df.to_csv(
        OUTPUT_DIR
        / "domain_fold_metrics.csv",
        index=False,
    )

    hyperparameters_df.to_csv(
        OUTPUT_DIR
        / "domain_best_hyperparameters.csv",
        index=False,
    )

    total_comparison.to_csv(
        OUTPUT_DIR
        / "domain_total_comparison.csv",
        index=False,
    )

    manifest = {
        "input": str(INPUT_CSV),
        "n_operations": int(len(df)),
        "n_participants": int(
            df[
                "participant_group"
            ].nunique()
        ),
        "n_numeric_features":
            int(len(features)),
        "outer_folds": [
            int(x)
            for x in sorted(
                df[FOLD_COLUMN]
                .dropna()
                .unique()
            )
        ],
        "domain_columns":
            DOMAIN_COLUMNS,
        "domain_min_score":
            DOMAIN_MIN_SCORE,
        "domain_max_score":
            DOMAIN_MAX_SCORE,
        "correlation_threshold":
            CORRELATION_THRESHOLD,
        "random_search_iter":
            RANDOM_SEARCH_ITER,
        "direct_total_benchmark": {
            "mae":
                DIRECT_TOTAL_BENCHMARK_MAE,
            "rmse":
                DIRECT_TOTAL_BENCHMARK_RMSE,
            "r2":
                DIRECT_TOTAL_BENCHMARK_R2,
        },
    }

    with open(
        OUTPUT_DIR
        / "domain_regression_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL TERMINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 78
    )
    print(
        "FINAL SIX-DOMAIN TOTAL-SCORE COMPARISON"
    )
    print(
        "=" * 78
    )

    print(
        "\nDIRECT TOTAL EXTRA TREES BENCHMARK"
    )
    print(
        f"MAE  = "
        f"{DIRECT_TOTAL_BENCHMARK_MAE:.4f}\n"
        f"RMSE = "
        f"{DIRECT_TOTAL_BENCHMARK_RMSE:.4f}\n"
        f"R2   = "
        f"{DIRECT_TOTAL_BENCHMARK_R2:.4f}"
    )

    print(
        "\nSIX-DOMAIN RAW SUM"
    )
    print(
        f"N    = "
        f"{total_raw_metrics.n}\n"
        f"MAE  = "
        f"{total_raw_metrics.mae:.4f}\n"
        f"RMSE = "
        f"{total_raw_metrics.rmse:.4f}\n"
        f"R2   = "
        f"{total_raw_metrics.r2:.4f}\n"
        f"rho  = "
        f"{total_raw_metrics.spearman_rho:.4f}"
    )

    print(
        "\nSIX-DOMAIN ROUNDED SUM"
    )
    print(
        f"N    = "
        f"{total_rounded_metrics.n}\n"
        f"MAE  = "
        f"{total_rounded_metrics.mae:.4f}\n"
        f"RMSE = "
        f"{total_rounded_metrics.rmse:.4f}\n"
        f"R2   = "
        f"{total_rounded_metrics.r2:.4f}\n"
        f"rho  = "
        f"{total_rounded_metrics.spearman_rho:.4f}"
    )

    mae_delta_raw = (
        total_raw_metrics.mae
        - DIRECT_TOTAL_BENCHMARK_MAE
    )

    mae_delta_rounded = (
        total_rounded_metrics.mae
        - DIRECT_TOTAL_BENCHMARK_MAE
    )

    print(
        "\nCHANGE RELATIVE TO DIRECT TOTAL MODEL"
    )
    print(
        f"Raw-sum MAE difference:     "
        f"{mae_delta_raw:+.4f}"
    )
    print(
        f"Rounded-sum MAE difference: "
        f"{mae_delta_rounded:+.4f}"
    )

    if mae_delta_raw < 0:
        print(
            "\nRESULT: Raw six-domain "
            "decomposition improved MAE "
            "over the direct total model."
        )
    else:
        print(
            "\nRESULT: Direct total prediction "
            "remains better than the raw "
            "six-domain sum."
        )

    print(
        f"\nOutputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )
        main()