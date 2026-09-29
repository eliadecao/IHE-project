from __future__ import annotations

"""Build an FK-only operation-level dataset and merge it with M-GEARS labels.

This script does not read the original raw DVST CSV files. It uses only the
supervisor FK outputs plus the existing dataset_v3.py helper functions.

Example (PowerShell):
    python build_dataset_v5_fk_only.py `
      --fk-dir "C:/Users/ROG/IHE-project/data/fk_outputs" `
      --label-dir "C:/Users/ROG/IHE-project/Griffin_Training_Dataset/M-GEARS" `
      --output-dir "C:/Users/ROG/IHE-project/data/dataset_v5_fk_only" `
      --v3-script "C:/Users/ROG/IHE-project/code/regression_task/dataset_v3.py"
"""

# python build_dataset_v6.py --fk-dir "C:/Users/ROG/IHE-project/data/updated_fk" --label-dir "C:/Users/ROG/IHE-project/Kinematics/corrected/Griffin_Training_Dataset/M-GEARS" --output-dir "C:/Users/ROG/IHE-project/data/v5_fk_corrected" --v3-script "C:/Users/ROG/IHE-project/code/regression_task/dataset_v3.py"

import argparse
import importlib.util
import json
import math
import sys
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


FK_REQUIRED_COLUMNS = {
    "TimeStamp",
    "USM",
    "Role",
    "True_Tip_X",
    "True_Tip_Y",
    "True_Tip_Z",
    "True_Tip_Matrix",
}


def load_v3_module(script_path: Path):
    """Dynamically import dataset_v3.py, including Python 3.14 dataclass support."""
    module_name = "dataset_v3_helpers"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import v3 script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def normalise_fk_stem(path: Path) -> str:
    """operation_FK.csv -> operation"""
    stem = path.stem
    return stem[:-3] if stem.endswith("_FK") else stem


def metadata_path_from_fk(fk_path: Path) -> Path:
    """Create a synthetic non-FK filename for the existing metadata parser."""
    return fk_path.with_name(normalise_fk_stem(fk_path) + ".csv")


def parse_matrix(value: object) -> np.ndarray | None:
    """Parse a flattened 4x4 transformation matrix."""
    if pd.isna(value):
        return None
    arr = np.fromstring(str(value).replace(",", " "), sep=" ", dtype=float)
    if arr.size != 16 or not np.all(np.isfinite(arr)):
        return None
    return arr.reshape(4, 4)


def load_fk_telemetry(
    fk_path: Path,
    v3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load, clean, time-normalise and QC one FK file."""
    fk = pd.read_csv(fk_path)
    missing = FK_REQUIRED_COLUMNS.difference(fk.columns)
    if missing:
        raise ValueError(f"{fk_path.name} is missing FK columns: {sorted(missing)}")

    fk = fk.copy()
    rows_original = len(fk)
    fk["timestamp_raw"] = pd.to_numeric(fk["TimeStamp"], errors="coerce")
    fk["usm"] = pd.to_numeric(fk["USM"], errors="coerce")
    fk["true_x"] = pd.to_numeric(fk["True_Tip_X"], errors="coerce")
    fk["true_y"] = pd.to_numeric(fk["True_Tip_Y"], errors="coerce")
    fk["true_z"] = pd.to_numeric(fk["True_Tip_Z"], errors="coerce")

    fk = fk.dropna(subset=["timestamp_raw", "usm"])
    removed_missing_key = rows_original - len(fk)
    fk["usm"] = fk["usm"].astype(int)
    fk["Role"] = fk["Role"].astype(str).str.strip()
    fk = fk.sort_values(["timestamp_raw", "usm"], kind="stable")

    duplicate_count = int(fk.duplicated(["timestamp_raw", "usm"]).sum())
    fk = fk.drop_duplicates(["timestamp_raw", "usm"], keep="first")

    if fk.empty:
        raise ValueError(f"{fk_path.name} contains no valid timestamp/USM rows")

    scale, unit = v3.infer_timestamp_scale(fk["timestamp_raw"].to_numpy())
    origin = float(fk["timestamp_raw"].min())
    fk["time_seconds"] = (fk["timestamp_raw"] - origin) * scale
    fk["_matrix"] = [parse_matrix(v) for v in fk["True_Tip_Matrix"].tolist()]

    valid_xyz = np.isfinite(fk[["true_x", "true_y", "true_z"]]).all(axis=1)
    duration = (
        float(fk["time_seconds"].max() - fk["time_seconds"].min())
        if len(fk) >= 2
        else np.nan
    )

    qc = {
        "qc_fk_rows_original": int(rows_original),
        "qc_fk_rows": int(len(fk)),
        "qc_fk_rows_removed_missing_timestamp_or_usm": int(removed_missing_key),
        "qc_fk_duplicate_timestamp_usm_rows_removed": duplicate_count,
        "qc_fk_timestamp_unit_inferred": unit,
        "qc_fk_duration_seconds": duration,
        "qc_fk_valid_xyz_fraction": float(valid_xyz.mean()),
        "qc_fk_usms_present": ",".join(map(str, sorted(fk["usm"].unique()))),
        "qc_fk_roles_present": ",".join(sorted(fk["Role"].dropna().unique())),
    }
    return fk, qc


def compute_fk_pose_features(
    frame: pd.DataFrame,
    usm: int,
    v3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute global Cartesian and rotational motion features for one PSM."""
    prefix = f"USM{usm}_pose_global"
    valid = np.isfinite(frame[["true_x", "true_y", "true_z"]]).all(axis=1)
    frame = frame.loc[valid].sort_values("time_seconds", kind="stable").copy()

    if frame.empty:
        return {f"{prefix}_n_samples": 0}, {}

    t = frame["time_seconds"].to_numpy(dtype=float)
    xyz = frame[["true_x", "true_y", "true_z"]].to_numpy(dtype=float)

    features, arrays = v3.compute_motion_summary(
        t,
        xyz,
        prefix,
        calculate_dimensionless_jerk=True,
    )
    if not arrays:
        return features, arrays

    smooth_xyz = arrays["values"]
    x_range, y_range, z_range = map(float, np.ptp(smooth_xyz, axis=0))
    path = features.get(f"{prefix}_path_l2", np.nan)

    features.update({
        f"{prefix}_x_range_m": x_range,
        f"{prefix}_y_range_m": y_range,
        f"{prefix}_z_range_m": z_range,
        f"{prefix}_workspace_box_volume_m3": x_range * y_range * z_range,
        f"{prefix}_economy_area_xy": v3.safe_divide(math.sqrt(x_range * y_range), path),
        f"{prefix}_economy_area_xz": v3.safe_divide(math.sqrt(x_range * z_range), path),
        f"{prefix}_economy_area_yz": v3.safe_divide(math.sqrt(y_range * z_range), path),
        f"{prefix}_economy_volume": v3.safe_divide(
            np.cbrt(x_range * y_range * z_range), path
        ),
    })

    matrix_rows = frame["_matrix"].tolist()
    matrix_valid = np.array([m is not None for m in matrix_rows], dtype=bool)
    features[f"{prefix}_matrix_valid_fraction"] = float(matrix_valid.mean())

    valid_t = frame.loc[matrix_valid, "time_seconds"].to_numpy(dtype=float)
    matrices = [m for m in matrix_rows if m is not None]
    if len(matrices) < 2:
        return features, arrays

    raw_rot = np.stack([m[:3, :3] for m in matrices])
    orth = np.array([
        np.linalg.norm(r.T @ r - np.eye(3), ord="fro")
        for r in raw_rot
    ])
    det = np.linalg.det(raw_rot)
    rotations = np.stack([v3.project_to_rotation_matrix(r) for r in raw_rot])

    angles = np.array([
        v3.rotation_angle_between(rotations[i], rotations[i + 1])
        for i in range(len(rotations) - 1)
    ])
    dt = np.diff(valid_t)
    keep = dt > 0
    angles = angles[keep]
    dt = dt[keep]
    angular_speed = angles / dt if len(dt) else np.array([])

    if len(angular_speed) >= 2:
        mid_t = (valid_t[:-1][keep] + valid_t[1:][keep]) / 2
        angular_acc = np.gradient(
            angular_speed,
            mid_t,
            edge_order=2 if len(angular_speed) >= 3 else 1,
        )
    else:
        angular_acc = np.array([])

    features.update({
        f"{prefix}_rotation_orthogonality_error_mean": float(np.mean(orth)),
        f"{prefix}_rotation_orthogonality_error_max": float(np.max(orth)),
        f"{prefix}_rotation_determinant_mean": float(np.mean(det)),
        f"{prefix}_rotation_determinant_std": float(np.std(det)),
        f"{prefix}_rotation_path_radians": float(np.sum(angles)) if len(angles) else np.nan,
        f"{prefix}_mean_angular_speed_rad_s": (
            float(np.mean(angular_speed)) if len(angular_speed) else np.nan
        ),
        f"{prefix}_max_angular_speed_rad_s": (
            float(np.max(angular_speed)) if len(angular_speed) else np.nan
        ),
        f"{prefix}_mean_abs_angular_acceleration_rad_s2": (
            float(np.mean(np.abs(angular_acc))) if len(angular_acc) else np.nan
        ),
        f"{prefix}_max_abs_angular_acceleration_rad_s2": (
            float(np.max(np.abs(angular_acc))) if len(angular_acc) else np.nan
        ),
    })
    return features, arrays


def extract_trial_features_fk_only(fk_path: Path, v3) -> dict[str, object]:
    """Extract one operation-level FK-only feature row."""
    metadata = v3.extract_metadata_from_dvst_filename(metadata_path_from_fk(fk_path))
    fk, fk_qc = load_fk_telemetry(fk_path, v3)

    features: dict[str, object] = {}
    features.update(metadata)
    features["fk_file_name"] = fk_path.name
    features.update(fk_qc)

    instrument_usms = sorted(
        fk.loc[fk["Role"].eq("PSM_Instrument"), "usm"].dropna().unique().tolist()
    )
    camera_usms = sorted(
        fk.loc[fk["Role"].eq("ECM_Camera"), "usm"].dropna().unique().tolist()
    )

    features["fk_instrument_usms"] = ",".join(map(str, instrument_usms))
    features["fk_camera_usms"] = ",".join(map(str, camera_usms))
    features["fk_instrument_usm_count"] = len(instrument_usms)
    features["fk_camera_usm_count"] = len(camera_usms)

    trajectories: dict[int, dict[str, np.ndarray]] = {}
    pose_paths: dict[int, float] = {}

    for usm in instrument_usms:
        pose_features, arrays = compute_fk_pose_features(
            fk.loc[fk["usm"] == usm], usm, v3
        )
        features.update(pose_features)
        if arrays:
            trajectories[usm] = arrays
            path = pose_features.get(f"USM{usm}_pose_global_path_l2", np.nan)
            pose_paths[usm] = float(path) if np.isfinite(path) else np.nan

    features["all_psm_pose_global_path_total_m"] = (
        float(np.nansum(list(pose_paths.values()))) if pose_paths else np.nan
    )

    finite_paths = {u: p for u, p in pose_paths.items() if np.isfinite(p)}
    max_path = max(finite_paths.values(), default=0.0)
    active_psms: list[int] = []

    for usm in instrument_usms:
        path = finite_paths.get(usm, np.nan)
        active = bool(np.isfinite(path) and max_path > 0 and path >= 0.02 * max_path)
        features[f"USM{usm}_pose_global_active"] = int(active)
        if active:
            active_psms.append(usm)

    features["active_psm_count"] = len(active_psms)

    pair_cache: dict[tuple[int, int], dict[str, float]] = {}
    for a, b in combinations(instrument_usms, 2):
        prefix = f"USM{a}_USM{b}_pose_global"
        if a in trajectories and b in trajectories:
            pair_features = v3.compute_pair_coordination_features(
                trajectories[a], trajectories[b], prefix
            )
        else:
            pair_features = {f"{prefix}_n_aligned_samples": 0}
        features.update(pair_features)
        pair_cache[(a, b)] = pair_features

    ranked = sorted(finite_paths, key=finite_paths.get, reverse=True)
    if len(ranked) >= 2:
        a, b = sorted(ranked[:2])
        features["active_pair_usm_a"] = a
        features["active_pair_usm_b"] = b
        source_prefix = f"USM{a}_USM{b}_pose_global"
        for key, value in pair_cache.get((a, b), {}).items():
            suffix = key.removeprefix(source_prefix + "_")
            features[f"active_pair_{suffix}"] = value
    else:
        features["active_pair_usm_a"] = np.nan
        features["active_pair_usm_b"] = np.nan

    return features


def build_feature_table(fk_dir: Path, output_dir: Path, v3) -> pd.DataFrame:
    """Extract one row per FK file and save an unmerged feature table."""
    fk_files = sorted(fk_dir.rglob("*_FK.csv"))
    print(f"Found {len(fk_files)} FK output files.")
    if not fk_files:
        raise FileNotFoundError(f"No *_FK.csv files found under: {fk_dir}")

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for i, fk_path in enumerate(fk_files, start=1):
        print(f"[FK-only features {i}/{len(fk_files)}] {fk_path.name}")
        try:
            rows.append(extract_trial_features_fk_only(fk_path, v3))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append({
                "fk_file": str(fk_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    error_path = output_dir / "dataset_v5_fk_only_processing_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)

    if not rows:
        raise ValueError(f"No FK operation was processed. Inspect: {error_path}")

    table = pd.DataFrame(rows)
    duplicate = table.duplicated(v3.MERGE_KEYS, keep=False)
    if duplicate.any():
        duplicate_path = output_dir / "dataset_v5_fk_only_duplicate_keys.csv"
        cols = [
            c for c in v3.MERGE_KEYS + ["file_name", "fk_file_name"]
            if c in table.columns
        ]
        table.loc[duplicate, cols].to_csv(duplicate_path, index=False)
        raise ValueError(f"Duplicate operation keys found. Inspect: {duplicate_path}")

    feature_path = output_dir / "dataset_v5_fk_only_features_unmerged.csv"
    table.to_csv(feature_path, index=False)

    print(f"\nSuccessfully processed {len(table)} of {len(fk_files)} FK files.")
    print(f"Unmerged FK-only features saved to: {feature_path}")
    if errors:
        print(f"Processing errors: {len(errors)}. Inspect: {error_path}")
    return table


def build_dataset(
    fk_dir: Path,
    label_dir: Path,
    output_dir: Path,
    v3,
) -> pd.DataFrame:
    """Build the final labelled FK-only dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(fk_dir, output_dir, v3)

    # Remove tasks that are not part of the study
    excluded_tasks = {
        "chicken_thigh",
        "cyst_model",
    }

    before = len(features)

    features = features[
        ~features["task_clean"].isin(excluded_tasks)
    ].reset_index(drop=True)

    print(
        f"Excluded {before - len(features)} operations "
        f"from tasks: {sorted(excluded_tasks)}"
    )

    labels = v3.build_labels(label_dir)

    labels_path = output_dir / "dataset_v5_fk_only_labels_aggregated.csv"
    labels.to_csv(labels_path, index=False)

    feature_audit_cols = [
        c for c in v3.MERGE_KEYS + ["file_name", "fk_file_name"]
        if c in features.columns
    ]
    label_audit_cols = [
        c for c in v3.MERGE_KEYS + [
            "target_score", "percentage_score", "n_label_records"
        ]
        if c in labels.columns
    ]

    audit = features[feature_audit_cols].merge(
        labels[label_audit_cols],
        on=v3.MERGE_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    audit_path = output_dir / "dataset_v5_fk_only_merge_audit.csv"
    audit.to_csv(audit_path, index=False)

    dataset = features.merge(
        labels,
        on=v3.MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError(f"Merged FK-only dataset is empty. Inspect: {audit_path}")

    dataset = dataset.sort_values(v3.MERGE_KEYS).reset_index(drop=True)
    dataset_path = output_dir / "dataset_v5_fk_only.csv"
    dataset.to_csv(dataset_path, index=False)

    manifest = {
        "dataset_version": "v5_fk_only",
        "one_row_represents": "one session-aware scored operation",
        "raw_dvst_used": False,
        "feature_source": "supervisor FK optimiser output only",
        "cartesian_source": "True_Tip_X, True_Tip_Y, True_Tip_Z",
        "rotation_source": "True_Tip_Matrix",
        "coordinate_frame": "calibrated global/base-referenced FK output",
        "joint_features_included": False,
        "setup_features_included": False,
        "old_endoscope_position_features_included": False,
        "merge_keys": list(v3.MERGE_KEYS),
        "number_fk_feature_rows": int(len(features)),
        "number_aggregated_label_rows": int(len(labels)),
        "number_final_merged_rows": int(len(dataset)),
        "output": str(dataset_path),
        "merge_audit": str(audit_path),
    }
    manifest_path = output_dir / "dataset_v5_fk_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved final dataset: {dataset_path}")
    print(f"Final shape: {dataset.shape}")
    print("\nMerge audit:")
    print(audit["_merge"].value_counts(dropna=False))

    if "task_clean" in dataset.columns:
        print("\nTask counts:")
        print(dataset["task_clean"].value_counts(dropna=False))

    if "target_score" in dataset.columns:
        print("\nTarget summary:")
        print(dataset["target_score"].describe())

    print(f"\nManifest saved to: {manifest_path}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fk-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v3-script", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in [args.fk_dir, args.label_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.v3_script.exists():
        raise FileNotFoundError(args.v3_script)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v3 = load_v3_module(args.v3_script.resolve())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        build_dataset(
            fk_dir=args.fk_dir.resolve(),
            label_dir=args.label_dir.resolve(),
            output_dir=output_dir,
            v3=v3,
        )


if __name__ == "__main__":
    main()
