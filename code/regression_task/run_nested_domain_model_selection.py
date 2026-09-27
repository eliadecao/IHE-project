from __future__ import annotations

"""
Nested participant-grouped model selection for six M-GEARS domains.

For each OUTER fold and each M-GEARS domain:
    1. Use only the outer-training participants.
    2. Run INNER GroupKFold model/hyperparameter selection separately for:
       - Extra Trees
       - Random Forest
       - HistGradientBoosting
       - XGBoost
       - CatBoost
    3. Select the algorithm with the lowest inner-CV MAE.
    4. Refit that selected model on the whole outer-training set.
    5. Predict the untouched outer-test participants.

Then:
    - combine the six unbiased OOF domain predictions,
    - sum applicable raw domain predictions,
    - optionally sum rounded/clipped domain predictions,
    - compare against:
        direct-total Extra Trees MAE = 2.4426
        six-domain Extra Trees raw-sum MAE = 2.4085

This script deliberately performs model selection INSIDE each outer training fold.
Do not replace this with choosing the globally best algorithm from outer OOF results,
because that would make the final performance estimate optimistically biased.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.feature_selection import VarianceThreshold
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

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    r"C:\Users\ROG\IHE-project\data\v5_fk_updated\qc_v5\v4"
    r"\grouped_cv\dataset_v4_grouped_folds.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\v5_fk_updated\qc_v5\v4"
    r"\grouped_cv\mgears_domain_model_selection"
)

# Existing unbiased benchmarks from your current pipeline.
DIRECT_TOTAL_ET_MAE = 2.4426
DIRECT_TOTAL_ET_RMSE = 3.0523
DIRECT_TOTAL_ET_R2 = 0.5360

SIX_DOMAIN_ET_RAW_SUM_MAE = 2.4085
SIX_DOMAIN_ET_RAW_SUM_RMSE = 3.0046
SIX_DOMAIN_ET_RAW_SUM_R2 = 0.5504
SIX_DOMAIN_ET_RAW_SUM_RHO = 0.6985

RANDOM_STATE = 42
INNER_CV_SPLITS = 4

# With 6 domains x 5 outer folds x 5 algorithms this is already substantial.
# Increase later only if the first run suggests it is worthwhile.
SEARCH_ITERATIONS = {
    "extra_trees": 10,
    "random_forest": 10,
    "hist_gradient_boosting": 10,
    "xgboost": 12,
    "catboost": 12,
}

CORRELATION_THRESHOLD = 0.98

DOMAIN_MIN_SCORE = 1
DOMAIN_MAX_SCORE = 5

TARGET_COLUMN = "target_score"
FOLD_COLUMN = "cv_fold"

DOMAIN_COLUMNS = [
    "mgears_depth_perception",
    "mgears_dexterity_with_multiple_wristed_instruments",
    "mgears_efficiency_flow_of_operation",
    "mgears_force_sensitivity_and_tissue_handling",
    "mgears_master_manipulator_workspace_robotic_control",
    "mgears_overall_performance_quality_of_the_final_product",
]


# ============================================================
# PREPROCESSING
# ============================================================

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Fit correlation pruning using training data only."""

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2-D array.")

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

        self.keep_indices_ = np.asarray(
            [
                idx
                for idx in range(X.shape[1])
                if idx not in drop
            ],
            dtype=int,
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


def make_pipeline(model) -> Pipeline:
    """
    Common leakage-safe preprocessing.

    We impute and filter features inside each inner/outer training process,
    never globally using the held-out outer fold.
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "variance",
            VarianceThreshold(threshold=0.0),
        ),
        (
            "correlation",
            CorrelationFilter(
                threshold=CORRELATION_THRESHOLD
            ),
        ),
        (
            "model",
            model,
        ),
    ])


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def model_spaces():
    """
    Return candidate pipelines and parameter distributions.

    Estimator-level threading is kept at 1 where possible because
    RandomizedSearchCV itself uses parallel workers.
    """

    models = {}

    # ---------------- Extra Trees ----------------
    et = ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    models["extra_trees"] = {
        "pipeline": make_pipeline(et),
        "params": {
            "model__n_estimators": [
                300, 500, 800, 1200
            ],
            "model__max_depth": [
                None, 6, 10, 15, 20
            ],
            "model__min_samples_split": [
                2, 3, 4, 6
            ],
            "model__min_samples_leaf": [
                1, 2, 3, 4
            ],
            "model__max_features": [
                1.0, "sqrt", 0.5, 0.75
            ],
            "model__bootstrap": [
                False, True
            ],
        },
    }

    # ---------------- Random Forest ----------------
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    models["random_forest"] = {
        "pipeline": make_pipeline(rf),
        "params": {
            "model__n_estimators": [
                300, 500, 800, 1200
            ],
            "model__max_depth": [
                None, 6, 10, 15, 20
            ],
            "model__min_samples_split": [
                2, 3, 4, 6
            ],
            "model__min_samples_leaf": [
                1, 2, 3, 4
            ],
            "model__max_features": [
                1.0, "sqrt", 0.5, 0.75
            ],
            "model__bootstrap": [
                True
            ],
        },
    }

    # ---------------- HistGradientBoosting ----------------
    hgb = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
    )

    models["hist_gradient_boosting"] = {
        "pipeline": make_pipeline(hgb),
        "params": {
            "model__learning_rate": [
                0.02, 0.03, 0.05, 0.08, 0.1
            ],
            "model__max_iter": [
                100, 200, 300, 500
            ],
            "model__max_leaf_nodes": [
                7, 15, 31, 63
            ],
            "model__max_depth": [
                None, 3, 5, 8
            ],
            "model__min_samples_leaf": [
                5, 10, 15, 20
            ],
            "model__l2_regularization": [
                0.0, 0.1, 0.5, 1.0, 5.0
            ],
        },
    }

    # ---------------- XGBoost ----------------
    # XGBRegressor exposes the sklearn estimator interface, so it can
    # participate directly in Pipeline + RandomizedSearchCV.
    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )

    models["xgboost"] = {
        "pipeline": make_pipeline(xgb),
        "params": {
            "model__n_estimators": [
                100, 200, 300, 500, 800
            ],
            "model__learning_rate": [
                0.01, 0.02, 0.03, 0.05, 0.08, 0.1
            ],
            "model__max_depth": [
                2, 3, 4, 5, 6
            ],
            "model__min_child_weight": [
                1, 2, 3, 5, 8
            ],
            "model__subsample": [
                0.6, 0.75, 0.9, 1.0
            ],
            "model__colsample_bytree": [
                0.5, 0.7, 0.85, 1.0
            ],
            "model__reg_alpha": [
                0.0, 0.01, 0.1, 0.5, 1.0
            ],
            "model__reg_lambda": [
                0.1, 0.5, 1.0, 2.0, 5.0, 10.0
            ],
        },
    }

    # ---------------- CatBoost ----------------
    # CatBoostRegressor implements a sklearn-compatible estimator API.
    cb = CatBoostRegressor(
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    models["catboost"] = {
        "pipeline": make_pipeline(cb),
        "params": {
            "model__iterations": [
                200, 400, 600, 800, 1200
            ],
            "model__learning_rate": [
                0.01, 0.02, 0.03, 0.05, 0.08
            ],
            "model__depth": [
                3, 4, 5, 6, 7, 8
            ],
            "model__l2_leaf_reg": [
                1, 3, 5, 10, 20
            ],
            "model__random_strength": [
                0.0, 0.1, 0.5, 1.0, 2.0
            ],
            "model__bagging_temperature": [
                0.0, 0.2, 0.5, 1.0
            ],
        },
    }

    return models


# ============================================================
# HELPERS
# ============================================================

@dataclass
class Metrics:
    n: int
    mae: float
    rmse: float
    r2: float
    rho: float


def safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(
        spearmanr(
            y_true,
            y_pred,
            nan_policy="omit",
        ).statistic
    )


def regression_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return Metrics(
            0, np.nan, np.nan, np.nan, np.nan
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

    return Metrics(
        n=int(len(y_true)),
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        rho=float(rho)
        if np.isfinite(rho)
        else np.nan,
    )


def clip_round(values):
    return np.clip(
        np.rint(values),
        DOMAIN_MIN_SCORE,
        DOMAIN_MAX_SCORE,
    )


def make_participant_group(df):
    if "participant_group" in df.columns:
        return df[
            "participant_group"
        ].astype(str)

    required = {
        "role",
        "participant_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Need participant_group or "
            f"role + participant_id. Missing: {missing}"
        )

    return (
        df["role"].astype(str)
        + "__participant_"
        + df["participant_id"].astype(str)
    )


def choose_numeric_features(df):
    """
    Defensively remove all targets, score-derived columns and IDs.
    """

    excluded_exact = {
        TARGET_COLUMN,
        FOLD_COLUMN,

        "percentage_score",
        "target_fraction",
        "mgears_domain_score_sum",
        "n_applicable_mgears_domains",
        "inferred_max_score",

        "participant_id",
        "trial",
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

        # All M-GEARS fields are targets/audit fields.
        if col.startswith("mgears_"):
            continue

        lower = col.lower()

        # Defensive exclusions.
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
            "No numeric input features remain."
        )

    return features


# ============================================================
# MAIN NESTED CV
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
    ).copy()

    df["participant_group"] = (
        make_participant_group(df)
    )

    required = set(
        DOMAIN_COLUMNS
        + [
            TARGET_COLUMN,
            FOLD_COLUMN,
        ]
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    features = (
        choose_numeric_features(df)
    )

    candidates = (
        model_spaces()
    )

    print(
        "\n=== NESTED SIX-DOMAIN MODEL SELECTION ==="
    )
    print(
        f"Input shape:        {df.shape}"
    )
    print(
        f"Participants:       "
        f"{df['participant_group'].nunique()}"
    )
    print(
        f"Numeric features:   {len(features)}"
    )
    print(
        f"Outer folds:        "
        f"{sorted(df[FOLD_COLUMN].unique())}"
    )
    print(
        f"Candidate models:   "
        f"{list(candidates.keys())}"
    )
    print(
        "\nThis may take a while: "
        "model selection is nested inside the participant-grouped outer CV."
    )

    pd.DataFrame({
        "feature": features
    }).to_csv(
        OUTPUT_DIR
        / "selected_model_input_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # OOF prediction table
    # --------------------------------------------------------

    metadata_cols = [
        c
        for c in [
            "file_name",
            "task_clean",
            "role",
            "participant_id",
            "trial",
            "participant_group",
            FOLD_COLUMN,
            TARGET_COLUMN,
        ]
        if c in df.columns
    ]

    oof = df[
        metadata_cols
        + DOMAIN_COLUMNS
    ].copy()

    for domain in DOMAIN_COLUMNS:
        oof[
            f"selected_pred_raw__{domain}"
        ] = np.nan

        oof[
            f"selected_pred_rounded__{domain}"
        ] = np.nan

        oof[
            f"selected_model__{domain}"
        ] = ""

    selection_rows = []
    candidate_search_rows = []
    outer_prediction_rows = []

    outer_folds = sorted(
        df[FOLD_COLUMN]
        .dropna()
        .unique()
    )

    # ========================================================
    # DOMAIN LOOP
    # ========================================================

    for domain_index, domain in enumerate(
        DOMAIN_COLUMNS,
        start=1,
    ):

        print(
            "\n"
            + "#" * 86
        )
        print(
            f"DOMAIN {domain_index}/6: {domain}"
        )
        print(
            "#" * 86
        )

        available = (
            pd.to_numeric(
                df[domain],
                errors="coerce",
            ).notna()
        )

        print(
            f"Available labels: "
            f"{int(available.sum())}/{len(df)}"
        )

        # ====================================================
        # OUTER FOLD LOOP
        # ====================================================

        for fold in outer_folds:

            outer_start = time.time()

            train_mask = (
                (df[FOLD_COLUMN] != fold)
                & available
            )

            test_mask = (
                (df[FOLD_COLUMN] == fold)
                & available
            )

            train_df = df.loc[
                train_mask
            ]

            test_df = df.loc[
                test_mask
            ]

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

            train_groups = train_df[
                "participant_group"
            ].to_numpy()

            n_train_groups = (
                train_df[
                    "participant_group"
                ].nunique()
            )

            inner_splits = min(
                INNER_CV_SPLITS,
                n_train_groups,
            )

            if inner_splits < 2:
                raise ValueError(
                    f"Too few groups for inner CV: "
                    f"{domain}, fold {fold}"
                )

            inner_cv = GroupKFold(
                n_splits=inner_splits
            )

            print(
                f"\nOuter fold {int(fold)} "
                f"| train={len(train_df)} "
                f"| test={len(test_df)} "
                f"| train participants={n_train_groups}"
            )

            best_algorithm = None
            best_inner_mae = np.inf
            best_search = None

            # ================================================
            # CANDIDATE MODEL LOOP
            # ================================================

            for model_index, (
                model_name,
                spec,
            ) in enumerate(
                candidates.items(),
                start=1,
            ):

                search_start = time.time()

                n_iter = (
                    SEARCH_ITERATIONS[
                        model_name
                    ]
                )

                search = RandomizedSearchCV(
                    estimator=clone(
                        spec["pipeline"]
                    ),
                    param_distributions=
                        spec["params"],
                    n_iter=n_iter,
                    scoring=
                        "neg_mean_absolute_error",
                    cv=inner_cv,
                    random_state=(
                        RANDOM_STATE
                        + domain_index * 1000
                        + int(fold) * 100
                        + model_index
                    ),
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                    error_score="raise",
                )

                search.fit(
                    X_train,
                    y_train,
                    groups=train_groups,
                )

                inner_mae = float(
                    -search.best_score_
                )

                elapsed = (
                    time.time()
                    - search_start
                )

                print(
                    f"  {model_name:<24} "
                    f"best inner MAE="
                    f"{inner_mae:.4f} "
                    f"({elapsed:.1f}s)"
                )

                candidate_search_rows.append({
                    "domain": domain,
                    "outer_fold": int(fold),
                    "model": model_name,
                    "best_inner_mae":
                        inner_mae,
                    "search_seconds":
                        elapsed,
                    "best_params":
                        json.dumps(
                            search.best_params_,
                            sort_keys=True,
                        ),
                })

                if (
                    inner_mae
                    < best_inner_mae
                ):
                    best_inner_mae = (
                        inner_mae
                    )
                    best_algorithm = (
                        model_name
                    )
                    best_search = (
                        search
                    )

            # ================================================
            # OUTER TEST PREDICTION
            # ================================================

            selected_pipeline = (
                best_search.best_estimator_
            )

            pred_raw = (
                selected_pipeline.predict(
                    X_test
                )
            )

            pred_rounded = (
                clip_round(
                    pred_raw
                )
            )

            metrics = (
                regression_metrics(
                    y_test,
                    pred_raw,
                )
            )

            print(
                f"  --> SELECTED: "
                f"{best_algorithm} "
                f"| inner MAE="
                f"{best_inner_mae:.4f} "
                f"| OUTER MAE="
                f"{metrics.mae:.4f}"
            )

            oof.loc[
                test_df.index,
                f"selected_pred_raw__{domain}",
            ] = pred_raw

            oof.loc[
                test_df.index,
                f"selected_pred_rounded__{domain}",
            ] = pred_rounded

            oof.loc[
                test_df.index,
                f"selected_model__{domain}",
            ] = best_algorithm

            selection_rows.append({
                "domain": domain,
                "outer_fold": int(fold),
                "selected_model":
                    best_algorithm,
                "selected_inner_mae":
                    best_inner_mae,
                "outer_n":
                    metrics.n,
                "outer_mae":
                    metrics.mae,
                "outer_rmse":
                    metrics.rmse,
                "outer_r2":
                    metrics.r2,
                "outer_spearman_rho":
                    metrics.rho,
                "outer_fold_seconds":
                    time.time()
                    - outer_start,
            })

            for row_idx, truth, prediction in zip(
                test_df.index,
                y_test,
                pred_raw,
            ):
                outer_prediction_rows.append({
                    "row_index":
                        int(row_idx),
                    "domain":
                        domain,
                    "outer_fold":
                        int(fold),
                    "selected_model":
                        best_algorithm,
                    "y_true":
                        float(truth),
                    "y_pred":
                        float(prediction),
                })

    # ========================================================
    # DOMAIN-LEVEL OOF RESULTS
    # ========================================================

    domain_summary_rows = []

    for domain in DOMAIN_COLUMNS:

        pred_col = (
            f"selected_pred_raw__{domain}"
        )

        valid = (
            pd.to_numeric(
                oof[domain],
                errors="coerce",
            ).notna()
            &
            pd.to_numeric(
                oof[pred_col],
                errors="coerce",
            ).notna()
        )

        y_true = pd.to_numeric(
            oof.loc[
                valid,
                domain,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        y_pred = pd.to_numeric(
            oof.loc[
                valid,
                pred_col,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        metrics = (
            regression_metrics(
                y_true,
                y_pred,
            )
        )

        domain_summary_rows.append({
            "domain":
                domain,
            "n_oof":
                metrics.n,
            "selected_model_oof_mae":
                metrics.mae,
            "selected_model_oof_rmse":
                metrics.rmse,
            "selected_model_oof_r2":
                metrics.r2,
            "selected_model_oof_spearman_rho":
                metrics.rho,
        })

    domain_summary = pd.DataFrame(
        domain_summary_rows
    )

    # ========================================================
    # RECONSTRUCT TOTAL SCORE
    # ========================================================

    applicability = (
        oof[
            DOMAIN_COLUMNS
        ].notna()
    )

    raw_pred_columns = [
        f"selected_pred_raw__{domain}"
        for domain in DOMAIN_COLUMNS
    ]

    rounded_pred_columns = [
        f"selected_pred_rounded__{domain}"
        for domain in DOMAIN_COLUMNS
    ]

    raw = (
        oof[
            raw_pred_columns
        ].to_numpy(
            dtype=float
        )
    )

    rounded = (
        oof[
            rounded_pred_columns
        ].to_numpy(
            dtype=float
        )
    )

    applicable = (
        applicability
        .to_numpy(
            dtype=bool
        )
    )

    raw_masked = np.where(
        applicable,
        raw,
        np.nan,
    )

    rounded_masked = np.where(
        applicable,
        rounded,
        np.nan,
    )

    # Require a prediction for every applicable domain.
    expected_count = (
        applicable.sum(axis=1)
    )

    available_raw_count = (
        np.isfinite(
            raw_masked
        ).sum(axis=1)
    )

    complete_prediction = (
        available_raw_count
        == expected_count
    )

    oof[
        "selected_total_raw_sum"
    ] = np.where(
        complete_prediction,
        np.nansum(
            raw_masked,
            axis=1,
        ),
        np.nan,
    )

    oof[
        "selected_total_rounded_sum"
    ] = np.where(
        complete_prediction,
        np.nansum(
            rounded_masked,
            axis=1,
        ),
        np.nan,
    )

    oof[
        "n_applicable_domains_reconstructed"
    ] = expected_count

    valid_total = (
        pd.to_numeric(
            oof[TARGET_COLUMN],
            errors="coerce",
        ).notna()
        &
        pd.to_numeric(
            oof[
                "selected_total_raw_sum"
            ],
            errors="coerce",
        ).notna()
    )

    y_total = pd.to_numeric(
        oof.loc[
            valid_total,
            TARGET_COLUMN,
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    pred_total_raw = pd.to_numeric(
        oof.loc[
            valid_total,
            "selected_total_raw_sum",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    pred_total_rounded = pd.to_numeric(
        oof.loc[
            valid_total,
            "selected_total_rounded_sum",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    total_raw_metrics = (
        regression_metrics(
            y_total,
            pred_total_raw,
        )
    )

    total_rounded_metrics = (
        regression_metrics(
            y_total,
            pred_total_rounded,
        )
    )

    # ========================================================
    # MODEL SELECTION COUNTS
    # ========================================================

    selection_df = pd.DataFrame(
        selection_rows
    )

    selection_counts = (
        selection_df
        .groupby(
            [
                "domain",
                "selected_model",
            ]
        )
        .size()
        .reset_index(
            name="n_outer_folds_selected"
        )
        .sort_values(
            [
                "domain",
                "n_outer_folds_selected",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    candidate_df = pd.DataFrame(
        candidate_search_rows
    )

    outer_prediction_df = pd.DataFrame(
        outer_prediction_rows
    )

    total_comparison = pd.DataFrame([
        {
            "approach":
                "direct_total_extra_trees",
            "n":
                len(df),
            "mae":
                DIRECT_TOTAL_ET_MAE,
            "rmse":
                DIRECT_TOTAL_ET_RMSE,
            "r2":
                DIRECT_TOTAL_ET_R2,
            "spearman_rho":
                np.nan,
        },
        {
            "approach":
                "six_domain_extra_trees_raw_sum",
            "n":
                len(df),
            "mae":
                SIX_DOMAIN_ET_RAW_SUM_MAE,
            "rmse":
                SIX_DOMAIN_ET_RAW_SUM_RMSE,
            "r2":
                SIX_DOMAIN_ET_RAW_SUM_R2,
            "spearman_rho":
                SIX_DOMAIN_ET_RAW_SUM_RHO,
        },
        {
            "approach":
                "six_domain_nested_selected_raw_sum",
            "n":
                total_raw_metrics.n,
            "mae":
                total_raw_metrics.mae,
            "rmse":
                total_raw_metrics.rmse,
            "r2":
                total_raw_metrics.r2,
            "spearman_rho":
                total_raw_metrics.rho,
        },
        {
            "approach":
                "six_domain_nested_selected_rounded_sum",
            "n":
                total_rounded_metrics.n,
            "mae":
                total_rounded_metrics.mae,
            "rmse":
                total_rounded_metrics.rmse,
            "r2":
                total_rounded_metrics.r2,
            "spearman_rho":
                total_rounded_metrics.rho,
        },
    ])

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    oof.to_csv(
        OUTPUT_DIR
        / "nested_selected_domain_oof_predictions.csv",
        index=False,
    )

    domain_summary.to_csv(
        OUTPUT_DIR
        / "nested_selected_domain_metrics.csv",
        index=False,
    )

    selection_df.to_csv(
        OUTPUT_DIR
        / "outer_fold_selected_models.csv",
        index=False,
    )

    selection_counts.to_csv(
        OUTPUT_DIR
        / "selected_model_counts_by_domain.csv",
        index=False,
    )

    candidate_df.to_csv(
        OUTPUT_DIR
        / "all_inner_model_search_results.csv",
        index=False,
    )

    outer_prediction_df.to_csv(
        OUTPUT_DIR
        / "long_format_outer_predictions.csv",
        index=False,
    )

    total_comparison.to_csv(
        OUTPUT_DIR
        / "nested_model_total_comparison.csv",
        index=False,
    )

    manifest = {
        "input":
            str(INPUT_CSV),
        "n_operations":
            int(len(df)),
        "n_participants":
            int(
                df[
                    "participant_group"
                ].nunique()
            ),
        "n_features":
            int(len(features)),
        "outer_folds": [
            int(f)
            for f in outer_folds
        ],
        "candidate_models":
            list(
                candidates.keys()
            ),
        "search_iterations":
            SEARCH_ITERATIONS,
        "correlation_threshold":
            CORRELATION_THRESHOLD,
        "selection_rule":
            (
                "Within each outer fold/domain, choose the "
                "candidate algorithm with the lowest grouped "
                "inner-CV MAE."
            ),
        "benchmarks": {
            "direct_total_extra_trees_mae":
                DIRECT_TOTAL_ET_MAE,
            "six_domain_extra_trees_raw_sum_mae":
                SIX_DOMAIN_ET_RAW_SUM_MAE,
        },
    }

    with open(
        OUTPUT_DIR
        / "nested_model_selection_manifest.json",
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
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 88
    )
    print(
        "FINAL NESTED SIX-DOMAIN MODEL-SELECTION RESULTS"
    )
    print(
        "=" * 88
    )

    print(
        "\nMODEL SELECTION COUNTS BY DOMAIN"
    )
    print(
        selection_counts.to_string(
            index=False
        )
    )

    print(
        "\nDOMAIN OOF PERFORMANCE"
    )
    print(
        domain_summary.to_string(
            index=False
        )
    )

    print(
        "\nTOTAL-SCORE COMPARISON"
    )
    print(
        total_comparison.to_string(
            index=False
        )
    )

    delta_vs_et_domains = (
        total_raw_metrics.mae
        - SIX_DOMAIN_ET_RAW_SUM_MAE
    )

    delta_vs_direct = (
        total_raw_metrics.mae
        - DIRECT_TOTAL_ET_MAE
    )

    print(
        "\nKEY COMPARISON"
    )
    print(
        f"Direct total Extra Trees MAE:       "
        f"{DIRECT_TOTAL_ET_MAE:.4f}"
    )
    print(
        f"Six-domain Extra Trees MAE:         "
        f"{SIX_DOMAIN_ET_RAW_SUM_MAE:.4f}"
    )
    print(
        f"Nested selected-model raw-sum MAE:  "
        f"{total_raw_metrics.mae:.4f}"
    )
    print(
        f"Change vs six-domain ET:            "
        f"{delta_vs_et_domains:+.4f}"
    )
    print(
        f"Change vs direct total ET:          "
        f"{delta_vs_direct:+.4f}"
    )

    if delta_vs_et_domains < 0:
        print(
            "\nRESULT: Nested per-domain model selection "
            "improved on the six-domain Extra Trees benchmark."
        )
    else:
        print(
            "\nRESULT: The simpler six-domain Extra Trees "
            "benchmark remains better."
        )

    print(
        f"\nOutputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
