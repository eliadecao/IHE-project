from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATASET = Path(
    r"C:\Users\ROG\IHE-project\data\v5_fk_corrected\qc_v5\v4"
    r"\grouped_cv\dataset_v4_grouped_folds.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\v5_fk_corrected\qc_v5\v4"
    r"\grouped_cv\mlp_regression"
)

TARGET = "target_score"

RANDOM_STATE = 42

CORRELATION_THRESHOLD = 0.95

N_ITER = 20


# ============================================================
# CORRELATION FILTER
# ============================================================

class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.keep_indices_ = None

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=float)

        df = pd.DataFrame(X)

        corr = df.corr().abs()

        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        )

        drop_cols = [
            column
            for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        self.keep_indices_ = [
            i for i in range(X.shape[1])
            if i not in drop_cols
        ]

        return self

    def transform(self, X):

        X = np.asarray(X, dtype=float)

        return X[:, self.keep_indices_]


# ============================================================
# HELPERS
# ============================================================

def make_group(df):

    if "participant_group" in df.columns:
        return df["participant_group"].astype(str)

    if "role" in df.columns and "participant_id" in df.columns:
        return (
            df["role"].astype(str)
            + "__"
            + df["participant_id"].astype(str)
        )

    raise ValueError(
        "Could not find participant grouping columns."
    )


def get_feature_columns(df):

    # --------------------------------------------------------
    # Columns that must NEVER enter the model
    # --------------------------------------------------------

    exact_exclude = {
        TARGET,
        "cv_fold",
        "participant_group",
        "participant_id",
        "role",
        "task",
        "file_name",
        "filename",
        "clean_id",
        "clean_task",

        # score-derived / audit columns
        "percentage_score",
        "percentage_score_std",
        "target_fraction",
        "n_applicable_mgears_domains",
        "mgears_domain_score_sum",
        "target_minus_domain_sum",
        "inferred_max_score",
    }

    # Any M-GEARS domain column must be excluded because these
    # are labels, not kinematic predictors.
    forbidden_prefixes = (
        "mgears_",
    )

    feature_cols = []

    for col in df.columns:

        if col in exact_exclude:
            continue

        if col.lower().startswith(forbidden_prefixes):
            continue

        # only numeric features
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        feature_cols.append(col)

    return feature_cols


def build_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "variance",
                VarianceThreshold(threshold=0.0)
            ),

            (
                "correlation",
                CorrelationFilter(
                    threshold=CORRELATION_THRESHOLD
                )
            ),

            # IMPORTANT FOR MLP
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                MLPRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                    early_stopping=False,
                )
            ),
        ]
    )


PARAM_DISTRIBUTIONS = {
    "model__hidden_layer_sizes": [
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
    ],

    "model__activation": [
        "relu",
        "tanh",
    ],

    "model__alpha": [
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ],

    "model__learning_rate_init": [
        1e-4,
        5e-4,
        1e-3,
    ],
}


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print("MLP DIRECT TOTAL REGRESSION")
    print("========================================\n")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATASET)

    print("Dataset shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    if "cv_fold" not in df.columns:
        raise ValueError(
            "cv_fold column not found. "
            "Use dataset_v4_grouped_folds.csv."
        )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    df = df[
        pd.to_numeric(
            df[TARGET],
            errors="coerce"
        ).notna()
    ].copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    print("Rows with valid target:", len(df))

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = make_group(df)

    print(
        "Unique participant groups:",
        groups.nunique()
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_cols = get_feature_columns(df)

    print(
        "Candidate numeric features:",
        len(feature_cols)
    )

    X = df[feature_cols].copy()

    y = df[TARGET].to_numpy(dtype=float)

    # --------------------------------------------------------
    # OOF containers
    # --------------------------------------------------------

    oof_pred = np.full(
        len(df),
        np.nan
    )

    fold_results = []

    best_params_records = []

    folds = sorted(
        df["cv_fold"].dropna().unique()
    )

    print(
        "Outer folds:",
        folds
    )

    # ========================================================
    # OUTER CV
    # ========================================================

    for fold in folds:

        print("\n")
        print("=" * 60)
        print(f"OUTER FOLD {fold}")
        print("=" * 60)

        train_mask = (
            df["cv_fold"] != fold
        )

        test_mask = (
            df["cv_fold"] == fold
        )

        X_train = X.loc[
            train_mask
        ].reset_index(drop=True)

        X_test = X.loc[
            test_mask
        ].reset_index(drop=True)

        y_train = y[train_mask.to_numpy()]

        y_test = y[test_mask.to_numpy()]

        groups_train = (
            groups.loc[train_mask]
            .reset_index(drop=True)
        )

        print(
            "Train:",
            len(X_train)
        )

        print(
            "Test:",
            len(X_test)
        )

        print(
            "Train participants:",
            groups_train.nunique()
        )

        # ----------------------------------------------------
        # Inner grouped CV
        # ----------------------------------------------------

        n_inner_splits = min(
            4,
            groups_train.nunique()
        )

        inner_cv = GroupKFold(
            n_splits=n_inner_splits
        )

        pipeline = build_pipeline()

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        # ----------------------------------------------------
        # Fit ONLY on training fold
        # ----------------------------------------------------

        search.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        best_model = search.best_estimator_

        print("\nBest parameters:")

        for key, value in search.best_params_.items():
            print(
                f"  {key}: {value}"
            )

        print(
            "Inner CV MAE:",
            -search.best_score_
        )

        # ----------------------------------------------------
        # Predict outer test fold
        # ----------------------------------------------------

        pred = best_model.predict(
            X_test
        )

        test_indices = np.where(
            test_mask.to_numpy()
        )[0]

        oof_pred[test_indices] = pred

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred
        )

        rho = spearmanr(
            y_test,
            pred
        ).statistic

        print("\nFold result")

        print(
            f"MAE  = {mae:.4f}"
        )

        print(
            f"RMSE = {rmse:.4f}"
        )

        print(
            f"R2   = {r2:.4f}"
        )

        print(
            f"rho  = {rho:.4f}"
        )

        fold_results.append(
            {
                "fold": fold,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "spearman_rho": rho,
                "inner_best_mae":
                    -search.best_score_,
            }
        )

        best_params_records.append(
            {
                "fold": fold,
                **search.best_params_,
            }
        )

    # ========================================================
    # FINAL OOF METRICS
    # ========================================================

    valid = np.isfinite(oof_pred)

    y_valid = y[valid]

    pred_valid = oof_pred[valid]

    overall_mae = mean_absolute_error(
        y_valid,
        pred_valid
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            pred_valid
        )
    )

    overall_r2 = r2_score(
        y_valid,
        pred_valid
    )

    overall_rho = spearmanr(
        y_valid,
        pred_valid
    ).statistic

    print("\n")
    print("=" * 60)
    print("FINAL MLP OOF RESULT")
    print("=" * 60)

    print(
        f"N    = {len(y_valid)}"
    )

    print(
        f"MAE  = {overall_mae:.4f}"
    )

    print(
        f"RMSE = {overall_rmse:.4f}"
    )

    print(
        f"R2   = {overall_r2:.4f}"
    )

    print(
        f"rho  = {overall_rho:.4f}"
    )

    # ========================================================
    # EXTRA TREES BENCHMARK
    # ========================================================

    EXTRA_TREES_MAE = 2.5722
    EXTRA_TREES_RMSE = 3.2003
    EXTRA_TREES_R2 = 0.4892

    print("\n")
    print("=" * 60)
    print("COMPARISON WITH EXTRA TREES")
    print("=" * 60)

    print("\nExtra Trees:")
    print(
        f"MAE  = {EXTRA_TREES_MAE:.4f}"
    )
    print(
        f"RMSE = {EXTRA_TREES_RMSE:.4f}"
    )
    print(
        f"R2   = {EXTRA_TREES_R2:.4f}"
    )

    print("\nMLP:")
    print(
        f"MAE  = {overall_mae:.4f}"
    )
    print(
        f"RMSE = {overall_rmse:.4f}"
    )
    print(
        f"R2   = {overall_r2:.4f}"
    )

    mae_difference = (
        overall_mae
        - EXTRA_TREES_MAE
    )

    print(
        "\nMLP - Extra Trees MAE difference:"
    )

    print(
        f"{mae_difference:+.4f}"
    )

    if overall_mae < EXTRA_TREES_MAE:

        print(
            "\nRESULT: MLP improved over "
            "the direct-total Extra Trees model."
        )

    else:

        print(
            "\nRESULT: Extra Trees remains "
            "better than MLP."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    oof_df = df.copy()

    oof_df[
        "mlp_oof_prediction"
    ] = oof_pred

    oof_df[
        "mlp_residual"
    ] = (
        oof_df[TARGET]
        - oof_df["mlp_oof_prediction"]
    )

    oof_df[
        "mlp_absolute_error"
    ] = (
        oof_df["mlp_residual"]
        .abs()
    )

    oof_df.to_csv(
        OUTPUT_DIR
        / "mlp_oof_predictions.csv",
        index=False
    )

    pd.DataFrame(
        fold_results
    ).to_csv(
        OUTPUT_DIR
        / "mlp_fold_metrics.csv",
        index=False
    )

    pd.DataFrame(
        best_params_records
    ).to_csv(
        OUTPUT_DIR
        / "mlp_best_parameters.csv",
        index=False
    )

    summary = {
        "model": "MLPRegressor",
        "target": TARGET,
        "n_operations": int(
            len(y_valid)
        ),
        "n_candidate_features": int(
            len(feature_cols)
        ),
        "mae": float(
            overall_mae
        ),
        "rmse": float(
            overall_rmse
        ),
        "r2": float(
            overall_r2
        ),
        "spearman_rho": float(
            overall_rho
        ),
        "extra_trees_mae": (
            EXTRA_TREES_MAE
        ),
        "mae_difference_mlp_minus_et":
            float(mae_difference),
    }

    with open(
        OUTPUT_DIR / "mlp_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
