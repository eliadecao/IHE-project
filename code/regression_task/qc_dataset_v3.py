from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"C:\Users\ROG\IHE-project\data\processed\dataset_v3.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="QC audit for dataset_v3.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.99)
    parser.add_argument("--missing-threshold", type=float, default=0.40)
    parser.add_argument("--near-constant-threshold", type=float, default=0.995)
    return parser.parse_args()


def get_numeric_features(df, excluded):
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir or (input_path.parent / "qc_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = {"target_score", "percentage_score", "file_name", "participant_id", "task_clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Confirmed from the original teacher-provided M-GEARS files:
    # target=0 and percentage=0 means unscored because domain ratings are NaN.
    unscored_mask = (
        pd.to_numeric(df["target_score"], errors="coerce").eq(0)
        & pd.to_numeric(df["percentage_score"], errors="coerce").eq(0)
    )

    audit_cols = [
        c for c in [
            "file_name", "role", "participant_id", "task_clean", "trial",
            "target_score", "percentage_score"
        ] if c in df.columns
    ]
    df.loc[unscored_mask, audit_cols].to_csv(
        output_dir / "unscored_zero_labels.csv", index=False
    )

    clean = df.copy()
    clean.loc[unscored_mask, ["target_score", "percentage_score"]] = np.nan
    model_df = clean.dropna(subset=["target_score"]).copy()
    model_df.to_csv(output_dir / "dataset_v3_model.csv", index=False)

    metadata_cols = {
        "file_name", "role", "participant_id", "task", "task_clean", "trial",
        "licence", "date", "session_licence", "session_datetime",
        "target_score", "percentage_score", "target_fraction"
    }
    numeric_cols = get_numeric_features(model_df, metadata_cols)

    missingness = pd.DataFrame({
        "column": model_df.columns,
        "dtype": [str(model_df[c].dtype) for c in model_df.columns],
        "n_missing": [int(model_df[c].isna().sum()) for c in model_df.columns],
        "missing_fraction": [float(model_df[c].isna().mean()) for c in model_df.columns],
        "n_unique_nonmissing": [int(model_df[c].nunique(dropna=True)) for c in model_df.columns],
    }).sort_values(["missing_fraction", "column"], ascending=[False, True])
    missingness["flag_high_missingness"] = (
        missingness["missing_fraction"] > args.missing_threshold
    )
    missingness.to_csv(output_dir / "column_missingness.csv", index=False)

    status_rows = []
    for col in numeric_cols:
        s = pd.to_numeric(model_df[col], errors="coerce").dropna()
        if len(s) == 0:
            status = "all_missing"
            frac = np.nan
            n_unique = 0
        else:
            n_unique = int(s.nunique())
            frac = float(s.value_counts().iloc[0] / len(s))
            if n_unique <= 1:
                status = "constant"
            elif frac >= args.near_constant_threshold:
                status = "near_constant"
            else:
                status = "variable"
        status_rows.append({
            "column": col,
            "n_nonmissing": len(s),
            "n_unique_nonmissing": n_unique,
            "most_common_fraction": frac,
            "status": status,
        })
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(output_dir / "constant_near_constant_features.csv", index=False)

    if numeric_cols:
        summary = (
            model_df[numeric_cols]
            .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            .T.reset_index().rename(columns={"index": "column"})
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_dir / "numeric_feature_summary.csv", index=False)

    # Robust 3*IQR outlier audit
    outlier_rows = []
    for col in numeric_cols:
        s = pd.to_numeric(model_df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        outlier_rows.append({
            "column": col, "n_nonmissing": len(s), "q1": q1, "q3": q3,
            "iqr": iqr, "lower_3iqr": lower, "upper_3iqr": upper,
            "n_outliers_3iqr": n_out, "outlier_fraction": n_out / len(s),
            "min": s.min(), "median": s.median(), "max": s.max(),
        })
    outliers = pd.DataFrame(outlier_rows)
    if not outliers.empty:
        outliers = outliers.sort_values("outlier_fraction", ascending=False)
    outliers.to_csv(output_dir / "outlier_summary_3iqr.csv", index=False)

    # Highly correlated numeric feature pairs
    usable = [
        c for c in numeric_cols
        if model_df[c].notna().sum() >= 5 and model_df[c].nunique(dropna=True) > 1
    ]
    if len(usable) >= 2:
        corr = model_df[usable].corr(min_periods=5)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = (
            corr.where(mask).stack().reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
        )
        pairs["abs_correlation"] = pairs["correlation"].abs()
        pairs = pairs[pairs["abs_correlation"] >= args.corr_threshold]
        pairs = pairs.sort_values("abs_correlation", ascending=False)
    else:
        pairs = pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
    pairs.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    by_task = (
        model_df.groupby("task_clean")["target_score"]
        .agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    )
    by_task.to_csv(output_dir / "target_summary_by_task.csv", index=False)

    by_participant = (
        model_df.groupby("participant_id")["target_score"]
        .agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    )
    by_participant.to_csv(output_dir / "target_summary_by_participant.csv", index=False)

    if "role" in model_df.columns:
        by_role = (
            model_df.groupby("role")["target_score"]
            .agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
        )
        by_role.to_csv(output_dir / "target_summary_by_role.csv", index=False)

    key_cols = [
        c for c in [
            "session_licence", "session_datetime", "licence", "date",
            "role", "participant_id", "task_clean", "trial"
        ] if c in model_df.columns
    ]
    duplicates = pd.DataFrame()
    if key_cols:
        dup_mask = model_df.duplicated(key_cols, keep=False)
        duplicates = model_df.loc[
            dup_mask, key_cols + ["file_name", "target_score"]
        ].sort_values(key_cols)
    duplicates.to_csv(output_dir / "duplicate_operation_keys.csv", index=False)

    manifest = {
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "n_unscored_zero_labels": int(unscored_mask.sum()),
        "model_shape": [int(model_df.shape[0]), int(model_df.shape[1])],
        "n_numeric_feature_columns": int(len(numeric_cols)),
        "n_high_missingness_columns": int(missingness["flag_high_missingness"].sum()),
        "feature_status_counts": {
            str(k): int(v) for k, v in status_df["status"].value_counts().to_dict().items()
        },
        "n_high_correlation_pairs": int(len(pairs)),
        "n_duplicate_operation_rows": int(len(duplicates)),
        "target_summary": {
            str(k): (None if pd.isna(v) else float(v))
            for k, v in model_df["target_score"].describe().to_dict().items()
        },
    }
    with open(output_dir / "qc_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== DATASET V3 QC COMPLETE ===")
    print(f"Input shape:              {df.shape}")
    print(f"Confirmed unscored rows:  {int(unscored_mask.sum())}")
    print(f"Clean modelling shape:    {model_df.shape}")
    print(f"Numeric feature columns:  {len(numeric_cols)}")
    print(f"High-missing columns:     {int(missingness['flag_high_missingness'].sum())}")
    print(f"High-correlation pairs:   {len(pairs)}")
    print(f"Duplicate operation rows: {len(duplicates)}")
    print(f"Outputs saved to:         {output_dir}")

    print("\nTarget summary:")
    print(model_df["target_score"].describe())

    print("\nFeature status counts:")
    print(status_df["status"].value_counts())


if __name__ == "__main__":
    main()
