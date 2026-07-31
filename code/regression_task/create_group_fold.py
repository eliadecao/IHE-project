from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


DEFAULT_INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\dataset_v5\qc_v5\v4\dataset_v4_prefiltered.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create participant-grouped cross-validation folds for the "
            "da Vinci Xi M-GEARS regression dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to dataset_v4_prefiltered.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to input parent / grouped_cv",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Requested number of grouped cross-validation folds.",
    )
    return parser.parse_args()


def normalise_id(series: pd.Series) -> pd.Series:
    """
    Convert IDs such as 1, 1.0 and '1' into a stable string representation.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    result = series.astype(str).str.strip()

    integer_mask = numeric.notna() & np.isclose(
        numeric,
        np.round(numeric),
        equal_nan=False,
    )
    result.loc[integer_mask] = (
        numeric.loc[integer_mask]
        .round()
        .astype("Int64")
        .astype(str)
    )

    return result.replace(
        {
            "nan": "missing",
            "None": "missing",
            "<NA>": "missing",
            "": "missing",
        }
    )


def build_group_id(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Use role + participant_id when participant numbers are reused across roles.

    This protects against accidentally treating trainee 1 and expert 1 as the
    same person. If no role column exists, participant_id alone is used.
    """
    participant = normalise_id(df["participant_id"])

    if "role" not in df.columns:
        return participant.rename("participant_group"), "participant_id"

    role = (
        df["role"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "nan": "unknown_role",
                "none": "unknown_role",
                "": "unknown_role",
            }
        )
    )

    role_counts_per_id = (
        pd.DataFrame(
            {
                "participant": participant,
                "role": role,
            }
        )
        .drop_duplicates()
        .groupby("participant")["role"]
        .nunique()
    )

    ids_reused_across_roles = role_counts_per_id[
        role_counts_per_id > 1
    ].index.tolist()

    if ids_reused_across_roles:
        group = role + "__participant_" + participant
        method = "role + participant_id"
    else:
        group = participant
        method = "participant_id"

    return group.rename("participant_group"), method


def make_grouped_folds(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
) -> tuple[pd.DataFrame, int]:
    n_groups = int(df[group_col].nunique())

    if n_groups < 2:
        raise ValueError(
            "At least two unique participant groups are required."
        )

    actual_splits = min(n_splits, n_groups)

    if actual_splits < 3:
        raise ValueError(
            f"Only {n_groups} participant groups were found. "
            "At least 3 groups are recommended."
        )

    splitter = GroupKFold(n_splits=actual_splits)

    assignments = pd.DataFrame(index=df.index)
    assignments["cv_fold"] = pd.NA

    dummy_x = np.zeros((len(df), 1))
    dummy_y = pd.to_numeric(
        df["target_score"],
        errors="coerce",
    ).to_numpy()
    groups = df[group_col].to_numpy()

    for fold_number, (_, test_index) in enumerate(
        splitter.split(dummy_x, dummy_y, groups=groups),
        start=1,
    ):
        assignments.loc[test_index, "cv_fold"] = fold_number

    assignments["cv_fold"] = assignments["cv_fold"].astype(int)

    return assignments, actual_splits


def main():
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir or (
        input_path.parent / "grouped_cv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        "participant_id",
        "task_clean",
        "target_score",
        "file_name",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["target_score"].isna().any():
        raise ValueError(
            "target_score still contains missing values."
        )

    participant_group, grouping_method = build_group_id(df)
    df = df.copy()
    df["participant_group"] = participant_group

    assignments, actual_splits = make_grouped_folds(
        df,
        group_col="participant_group",
        n_splits=args.n_splits,
    )
    df["cv_fold"] = assignments["cv_fold"]

    # --------------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------------
    group_fold_counts = (
        df.groupby("participant_group")["cv_fold"]
        .nunique()
    )
    leaking_groups = group_fold_counts[
        group_fold_counts > 1
    ]

    if not leaking_groups.empty:
        raise RuntimeError(
            "Participant leakage detected. These groups appear in more "
            f"than one fold: {leaking_groups.index.tolist()}"
        )

    if df["cv_fold"].isna().any():
        raise RuntimeError(
            "Some rows did not receive a CV fold."
        )

    # --------------------------------------------------------------
    # Save row-level fold assignments
    # --------------------------------------------------------------
    assignment_cols = [
        c for c in [
            "file_name",
            "role",
            "participant_id",
            "participant_group",
            "task_clean",
            "trial",
            "target_score",
            "percentage_score",
            "cv_fold",
        ]
        if c in df.columns
    ]

    df[assignment_cols].to_csv(
        output_dir / "grouped_cv_fold_assignments.csv",
        index=False,
    )

    # Dataset with frozen fold assignment
    df.to_csv(
        output_dir / "dataset_v4_grouped_folds.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Participant-level summary
    # --------------------------------------------------------------
    participant_agg = {
        "n_operations": ("file_name", "size"),
        "mean_target": ("target_score", "mean"),
        "std_target": ("target_score", "std"),
        "min_target": ("target_score", "min"),
        "max_target": ("target_score", "max"),
        "n_tasks": ("task_clean", "nunique"),
        "cv_fold": ("cv_fold", "first"),
    }

    if "role" in df.columns:
        participant_agg["role"] = ("role", "first")

    participant_summary = (
        df.groupby("participant_group", as_index=False)
        .agg(**participant_agg)
        .sort_values(
            ["cv_fold", "participant_group"]
        )
    )

    participant_summary.to_csv(
        output_dir / "participant_group_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Fold-level summary
    # --------------------------------------------------------------
    fold_summary = (
        df.groupby("cv_fold")
        .agg(
            n_operations=("file_name", "size"),
            n_participants=("participant_group", "nunique"),
            mean_target=("target_score", "mean"),
            std_target=("target_score", "std"),
            min_target=("target_score", "min"),
            max_target=("target_score", "max"),
            n_tasks=("task_clean", "nunique"),
        )
        .reset_index()
        .sort_values("cv_fold")
    )

    fold_summary.to_csv(
        output_dir / "grouped_cv_fold_summary.csv",
        index=False,
    )

    # Task distribution by fold
    task_distribution = pd.crosstab(
        df["cv_fold"],
        df["task_clean"],
        margins=True,
    )
    task_distribution.to_csv(
        output_dir / "task_distribution_by_fold.csv"
    )

    # Role distribution by fold
    if "role" in df.columns:
        role_distribution = pd.crosstab(
            df["cv_fold"],
            df["role"],
            margins=True,
        )
        role_distribution.to_csv(
            output_dir / "role_distribution_by_fold.csv"
        )

    # Participant-to-fold mapping for reproducibility
    participant_to_fold = (
        df[
            [
                "participant_group",
                "cv_fold",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "cv_fold",
                "participant_group",
            ]
        )
    )
    participant_to_fold.to_csv(
        output_dir / "participant_to_fold_mapping.csv",
        index=False,
    )

    manifest = {
        "input_path": str(input_path),
        "input_shape": [
            int(df.shape[0]),
            int(df.shape[1] - 2),
        ],
        "grouping_method": grouping_method,
        "n_unique_participant_groups": int(
            df["participant_group"].nunique()
        ),
        "requested_n_splits": int(args.n_splits),
        "actual_n_splits": int(actual_splits),
        "n_rows_assigned": int(len(df)),
        "participant_leakage_detected": False,
        "fold_operation_counts": {
            str(int(row.cv_fold)): int(row.n_operations)
            for row in fold_summary.itertuples()
        },
        "fold_participant_counts": {
            str(int(row.cv_fold)): int(row.n_participants)
            for row in fold_summary.itertuples()
        },
        "methodology_note": (
            "All operations from one participant_group are assigned "
            "to exactly one test fold. The same frozen folds should be "
            "used for every model comparison."
        ),
    }

    with open(
        output_dir / "grouped_cv_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== PARTICIPANT GROUPING COMPLETE ===")
    print(f"Input rows:                  {len(df)}")
    print(
        "Grouping method:             "
        f"{grouping_method}"
    )
    print(
        "Unique participant groups:   "
        f"{df['participant_group'].nunique()}"
    )
    print(f"Number of CV folds:          {actual_splits}")
    print("Participant leakage:         NONE")
    print(f"Outputs saved to:            {output_dir}")

    print("\nFold summary:")
    print(fold_summary.to_string(index=False))

    print("\nParticipant-to-fold mapping:")
    print(participant_to_fold.to_string(index=False))


if __name__ == "__main__":
    main()