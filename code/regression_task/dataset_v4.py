from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\qc_v7\dataset_v7_model.csv"
)


METADATA_COLUMNS = {
    "file_name",
    "role",
    "participant_id",
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
}


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible correlation filter.

    IMPORTANT:
    Use this transformer INSIDE a cross-validation Pipeline so that correlated
    features are selected using training-fold data only.

    Parameters
    ----------
    threshold:
        Remove one feature from each pair with absolute Pearson correlation
        greater than or equal to this value.
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)

        numeric = X_df.select_dtypes(include=[np.number]).copy()
        non_numeric_cols = [c for c in X_df.columns if c not in numeric.columns]

        if numeric.shape[1] < 2:
            self.columns_to_drop_ = []
            self.columns_to_keep_ = list(X_df.columns)
            return self

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        columns_to_drop = [
            column
            for column in upper.columns
            if (upper[column] >= self.threshold).any()
        ]

        self.columns_to_drop_ = columns_to_drop
        self.columns_to_keep_ = [
            c for c in X_df.columns
            if c not in columns_to_drop
        ]

        # Keep any non-numeric columns, although modelling pipelines should
        # normally encode or remove them separately.
        for col in non_numeric_cols:
            if col not in self.columns_to_keep_:
                self.columns_to_keep_.append(col)

        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        missing = [c for c in self.columns_to_keep_ if c not in X_df.columns]
        if missing:
            raise ValueError(
                "Input data are missing columns seen during fit: "
                + ", ".join(missing[:10])
            )
        return X_df.loc[:, self.columns_to_keep_].copy()

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset_v4 by safely removing unusable features and "
            "creating a correlation audit."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to qc_v5/dataset_v5_model.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to input parent / v4",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.99,
        help="Absolute Pearson correlation threshold.",
    )
    parser.add_argument(
        "--near-constant-threshold",
        type=float,
        default=0.995,
        help="Most-common-value fraction used to flag near-constant features.",
    )
    parser.add_argument(
        "--drop-near-constant",
        action="store_true",
        help=(
            "Also remove near-constant features globally. By default they are "
            "reported but retained."
        ),
    )
    parser.add_argument(
        "--write-global-correlation-version",
        action="store_true",
        help=(
            "Write an exploratory globally correlation-pruned dataset. "
            "Do not use it for final cross-validated performance estimates."
        ),
    )
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_score_columns = {
        "target_score",
        "percentage_score",
        "target_fraction",
        "target_score_std",
        "percentage_score_std",
        "n_label_records",
        "n_applicable_mgears_domains",
        "mgears_domain_score_sum",
        "target_minus_domain_sum",
        "inferred_max_score",
    }

    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and c not in excluded_score_columns
        and not c.lower().startswith("mgears_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def feature_status_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    near_constant_threshold: float,
) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        nonmissing = s.dropna()
        n_nonmissing = int(nonmissing.shape[0])
        n_unique = int(nonmissing.nunique())

        if n_nonmissing == 0:
            most_common_fraction = np.nan
            status = "all_missing"
        else:
            most_common_fraction = float(
                nonmissing.value_counts(dropna=True).iloc[0] / n_nonmissing
            )
            if n_unique <= 1:
                status = "constant"
            elif most_common_fraction >= near_constant_threshold:
                status = "near_constant"
            else:
                status = "variable"

        rows.append(
            {
                "column": col,
                "n_nonmissing": n_nonmissing,
                "n_missing": int(s.isna().sum()),
                "missing_fraction": float(s.isna().mean()),
                "n_unique_nonmissing": n_unique,
                "most_common_fraction": most_common_fraction,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def correlation_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    usable = [
        c for c in feature_cols
        if df[c].notna().sum() >= 5
        and df[c].nunique(dropna=True) > 1
    ]

    if len(usable) < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "abs_correlation",
            ]
        )

    corr = df[usable].corr(method="pearson", min_periods=5)
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    pairs = (
        corr.where(upper_mask)
        .stack()
        .reset_index()
        .rename(
            columns={
                "level_0": "feature_1",
                "level_1": "feature_2",
                0: "correlation",
            }
        )
    )

    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs = pairs[pairs["abs_correlation"] >= threshold].copy()

    return pairs.sort_values(
        ["abs_correlation", "feature_1", "feature_2"],
        ascending=[False, True, True],
    )


def greedy_global_correlation_drop(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy deterministic global correlation pruning.

    This is useful for exploration and reporting only.
    For final model evaluation, use CorrelationFilter inside each CV fold.
    """

    usable = [
        c for c in feature_cols
        if df[c].notna().sum() >= 5
        and df[c].nunique(dropna=True) > 1
    ]

    if len(usable) < 2:
        return [], pd.DataFrame(
            columns=["dropped_feature", "kept_feature", "abs_correlation"]
        )

    corr = df[usable].corr(method="pearson").abs()
    dropped: set[str] = set()
    decisions = []

    # Preserve earlier columns and drop later columns deterministically.
    for j, col_j in enumerate(usable):
        if col_j in dropped:
            continue

        for i in range(j):
            col_i = usable[i]

            if col_i in dropped:
                continue

            value = corr.loc[col_i, col_j]

            if pd.notna(value) and value >= threshold:
                dropped.add(col_j)
                decisions.append(
                    {
                        "dropped_feature": col_j,
                        "kept_feature": col_i,
                        "abs_correlation": float(value),
                    }
                )
                break

    return sorted(dropped), pd.DataFrame(decisions)


def main():
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir or (input_path.parent / "v4")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    if "target_score" not in df.columns:
        raise ValueError("target_score is missing from the input dataset.")

    if df["target_score"].isna().any():
        raise ValueError(
            "Input still contains missing target_score values. "
            "Use qc_v5/dataset_v5_model.csv."
        )

    feature_cols = numeric_feature_columns(df)

    status = feature_status_table(
        df,
        feature_cols,
        near_constant_threshold=args.near_constant_threshold,
    )
    status.to_csv(output_dir / "v4_feature_status.csv", index=False)

    all_missing_cols = status.loc[
        status["status"] == "all_missing", "column"
    ].tolist()

    constant_cols = status.loc[
        status["status"] == "constant", "column"
    ].tolist()

    near_constant_cols = status.loc[
        status["status"] == "near_constant", "column"
    ].tolist()

    mandatory_drop = sorted(set(all_missing_cols + constant_cols))

    if args.drop_near_constant:
        mandatory_drop = sorted(
            set(mandatory_drop + near_constant_cols)
        )

    # Safe global filtering:
    # all-missing and constant columns contain no information in any fold.
    prefiltered = df.drop(columns=mandatory_drop, errors="ignore").copy()
    prefiltered.to_csv(
        output_dir / "dataset_v4_prefiltered.csv",
        index=False,
    )

    remaining_feature_cols = numeric_feature_columns(prefiltered)

    corr_pairs = correlation_pairs(
        prefiltered,
        remaining_feature_cols,
        threshold=args.corr_threshold,
    )
    corr_pairs.to_csv(
        output_dir / "v4_high_correlation_pairs.csv",
        index=False,
    )

    global_corr_drop, global_decisions = greedy_global_correlation_drop(
        prefiltered,
        remaining_feature_cols,
        threshold=args.corr_threshold,
    )
    global_decisions.to_csv(
        output_dir / "v4_global_correlation_drop_decisions.csv",
        index=False,
    )

    if args.write_global_correlation_version:
        global_pruned = prefiltered.drop(
            columns=global_corr_drop,
            errors="ignore",
        )
        global_pruned.to_csv(
            output_dir / "dataset_v6_global_corr_exploratory.csv",
            index=False,
        )

    removed_rows = []
    removed_rows.extend(
        {"column": c, "reason": "all_missing"}
        for c in all_missing_cols
    )
    removed_rows.extend(
        {"column": c, "reason": "constant"}
        for c in constant_cols
    )

    if args.drop_near_constant:
        removed_rows.extend(
            {"column": c, "reason": "near_constant"}
            for c in near_constant_cols
        )

    pd.DataFrame(removed_rows).to_csv(
        output_dir / "v4_removed_features.csv",
        index=False,
    )

    manifest = {
        "input_path": str(input_path),
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "n_numeric_features_input": int(len(feature_cols)),
        "n_all_missing_removed": int(len(all_missing_cols)),
        "n_constant_removed": int(len(constant_cols)),
        "n_near_constant_flagged": int(len(near_constant_cols)),
        "drop_near_constant": bool(args.drop_near_constant),
        "prefiltered_shape": [
            int(prefiltered.shape[0]),
            int(prefiltered.shape[1]),
        ],
        "n_numeric_features_prefiltered": int(
            len(remaining_feature_cols)
        ),
        "correlation_threshold": float(args.corr_threshold),
        "n_high_correlation_pairs": int(len(corr_pairs)),
        "n_features_global_corr_would_drop": int(
            len(global_corr_drop)
        ),
        "global_correlation_dataset_written": bool(
            args.write_global_correlation_version
        ),
        "methodology_note": (
            "Use dataset_v4_prefiltered.csv for final modelling. "
            "Apply CorrelationFilter inside the cross-validation pipeline. "
            "Do not use globally correlation-pruned data for final unbiased "
            "performance estimation."
        ),
    }

    with open(
        output_dir / "v4_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== DATASET V4 PREPARATION COMPLETE ===")
    print(f"Input shape:                    {df.shape}")
    print(f"Numeric features input:         {len(feature_cols)}")
    print(f"All-missing features removed:   {len(all_missing_cols)}")
    print(f"Constant features removed:      {len(constant_cols)}")
    print(f"Near-constant features flagged: {len(near_constant_cols)}")
    print(f"Prefiltered shape:              {prefiltered.shape}")
    print(
        "Numeric features remaining:     "
        f"{len(remaining_feature_cols)}"
    )
    print(
        "High-correlation pairs:         "
        f"{len(corr_pairs)}"
    )
    print(
        "Global pruning would remove:    "
        f"{len(global_corr_drop)} features"
    )
    print(f"Outputs saved to:               {output_dir}")

    print(
        "\nIMPORTANT: For final regression, use "
        "dataset_v4_prefiltered.csv and put CorrelationFilter "
        "inside grouped cross-validation."
    )


if __name__ == "__main__":
    main()