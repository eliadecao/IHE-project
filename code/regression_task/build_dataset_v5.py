from __future__ import annotations

"""
Build dataset_v5_fk.csv from:
1. original raw DVST telemetry (joint-space and setup features),
2. supervisor FK outputs (global true-tip Cartesian features),
3. M-GEARS label files.

This script intentionally reuses the tested v3 helper functions while replacing
camera-relative EndoscopePosition features with FK-derived global tool-tip poses.

Expected naming:
    raw: DVST_XI_..._trial.csv
    FK:  DVST_XI_..._trial_FK.csv

Example:
    python build_dataset_v5_fk.py \
        --raw-dir "C:/project/Griffin_Training_Dataset/DVST_XI" \
        --fk-dir "C:/project/data/fk_outputs" \
        --label-dir "C:/project/Griffin_Training_Dataset/M-GEARS" \
        --output-dir "C:/project/data/processed" \
        --v3-script "C:/project/build_dataset_v3.py"
"""

import sys
import argparse
import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd


FK_REQUIRED_COLUMNS = {
    "TimeStamp", "USM", "Role",
    "True_Tip_X", "True_Tip_Y", "True_Tip_Z", "True_Tip_Matrix",
}


def load_v3_module(script_path: Path):
    module_name = "dataset_v3_helpers"

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import v3 script: {script_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required for @dataclass and some other decorators,
    # especially under Python 3.14.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a broken partially imported module behind.
        sys.modules.pop(module_name, None)
        raise

    return module


def normalise_fk_stem(path: Path) -> str:
    stem = path.stem
    return stem[:-3] if stem.endswith("_FK") else stem


def parse_matrix(value: object) -> np.ndarray | None:
    if pd.isna(value):
        return None
    arr = np.fromstring(str(value).replace(",", " "), sep=" ", dtype=float)
    if arr.size != 16 or not np.all(np.isfinite(arr)):
        return None
    return arr.reshape(4, 4)


def load_fk_telemetry(fk_path: Path, v3) -> tuple[pd.DataFrame, dict[str, object]]:
    fk = pd.read_csv(fk_path)
    missing = FK_REQUIRED_COLUMNS.difference(fk.columns)
    if missing:
        raise ValueError(f"{fk_path.name} is missing FK columns: {sorted(missing)}")

    fk = fk.copy()
    fk["timestamp_raw"] = pd.to_numeric(fk["TimeStamp"], errors="coerce")
    fk["usm"] = pd.to_numeric(fk["USM"], errors="coerce")
    fk["true_x"] = pd.to_numeric(fk["True_Tip_X"], errors="coerce")
    fk["true_y"] = pd.to_numeric(fk["True_Tip_Y"], errors="coerce")
    fk["true_z"] = pd.to_numeric(fk["True_Tip_Z"], errors="coerce")
    fk = fk.dropna(subset=["timestamp_raw", "usm"])
    fk["usm"] = fk["usm"].astype(int)
    fk = fk.sort_values(["timestamp_raw", "usm"])
    duplicate_count = int(fk.duplicated(["timestamp_raw", "usm"]).sum())
    fk = fk.drop_duplicates(["timestamp_raw", "usm"], keep="first")

    scale, unit = v3.infer_timestamp_scale(fk["timestamp_raw"].to_numpy())
    origin = float(fk["timestamp_raw"].min())
    fk["time_seconds"] = (fk["timestamp_raw"] - origin) * scale
    fk["_matrix"] = [parse_matrix(v) for v in fk["True_Tip_Matrix"].tolist()]

    valid_xyz = np.isfinite(fk[["true_x", "true_y", "true_z"]]).all(axis=1)
    qc = {
        "qc_fk_rows": int(len(fk)),
        "qc_fk_duplicate_timestamp_usm_rows_removed": duplicate_count,
        "qc_fk_timestamp_unit_inferred": unit,
        "qc_fk_valid_xyz_fraction": float(valid_xyz.mean()) if len(fk) else np.nan,
        "qc_fk_usms_present": ",".join(map(str, sorted(fk["usm"].unique()))),
    }
    return fk, qc


def compute_fk_pose_features(
    frame: pd.DataFrame,
    usm: int,
    v3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prefix = f"USM{usm}_pose_global"
    valid = np.isfinite(frame[["true_x", "true_y", "true_z"]]).all(axis=1)
    frame = frame.loc[valid].sort_values("time_seconds")

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
    ranges = np.ptp(smooth_xyz, axis=0)
    x_range, y_range, z_range = map(float, ranges)
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
    matrix_valid = [m is not None for m in matrix_rows]
    features[f"{prefix}_matrix_valid_fraction"] = float(np.mean(matrix_valid))

    valid_t = frame.loc[matrix_valid, "time_seconds"].to_numpy(dtype=float)
    matrices = [m for m in matrix_rows if m is not None]
    if len(matrices) >= 2:
        raw_rot = np.stack([m[:3, :3] for m in matrices])
        orth = np.array([
            np.linalg.norm(r.T @ r - np.eye(3), ord="fro") for r in raw_rot
        ])
        det = np.linalg.det(raw_rot)
        rotations = np.stack([v3.project_to_rotation_matrix(r) for r in raw_rot])
        angles = np.array([
            v3.rotation_angle_between(rotations[i], rotations[i + 1])
            for i in range(len(rotations) - 1)
        ])
        dt = np.diff(valid_t)
        keep = dt > 0
        angles, dt = angles[keep], dt[keep]
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
            f"{prefix}_rotation_path_radians": float(np.sum(angles)) if len(angles) else np.nan,
            f"{prefix}_mean_angular_speed_rad_s": float(np.mean(angular_speed))
                if len(angular_speed) else np.nan,
            f"{prefix}_max_angular_speed_rad_s": float(np.max(angular_speed))
                if len(angular_speed) else np.nan,
            f"{prefix}_mean_abs_angular_acceleration_rad_s2":
                float(np.mean(np.abs(angular_acc))) if len(angular_acc) else np.nan,
            f"{prefix}_max_abs_angular_acceleration_rad_s2":
                float(np.max(np.abs(angular_acc))) if len(angular_acc) else np.nan,
        })

    return features, arrays


def extract_trial_features_v5(
    raw_path: Path,
    fk_path: Path,
    v3,
) -> dict[str, object]:
    metadata = v3.extract_metadata_from_dvst_filename(raw_path)
    loaded = v3.load_raw_telemetry(raw_path)
    raw = loaded.data
    fk, fk_qc = load_fk_telemetry(fk_path, v3)

    features: dict[str, object] = {}
    features.update(metadata)
    features["fk_file_name"] = fk_path.name
    features.update(loaded.qc)
    features.update(fk_qc)
    features.update(v3.compute_setup_qc_features(raw))

    # Preserve joint-space features from the original telemetry.
    joint_paths: dict[int, float] = {}
    for usm in sorted(raw["usm"].unique()):
        joint_features, _ = v3.compute_joint_features(raw[raw["usm"] == usm], usm)
        features.update(joint_features)
        path = joint_features.get(f"USM{usm}_joint_path_l1", np.nan)
        joint_paths[usm] = float(path) if np.isfinite(path) else np.nan
    features["all_usm_joint_path_l1_total"] = float(np.nansum(list(joint_paths.values())))

    # Use Role from the FK output rather than an old hard-coded ECM index.
    instrument_usms = sorted(
        fk.loc[fk["Role"].astype(str).eq("PSM_Instrument"), "usm"].unique().tolist()
    )
    camera_usms = sorted(
        fk.loc[fk["Role"].astype(str).eq("ECM_Camera"), "usm"].unique().tolist()
    )
    features["fk_instrument_usms"] = ",".join(map(str, instrument_usms))
    features["fk_camera_usms"] = ",".join(map(str, camera_usms))

    trajectories: dict[int, dict[str, np.ndarray]] = {}
    pose_paths: dict[int, float] = {}
    for usm in instrument_usms:
        pose_features, arrays = compute_fk_pose_features(fk[fk["usm"] == usm], usm, v3)
        features.update(pose_features)
        if arrays:
            trajectories[usm] = arrays
            p = pose_features.get(f"USM{usm}_pose_global_path_l2", np.nan)
            pose_paths[usm] = float(p) if np.isfinite(p) else np.nan

    features["all_psm_pose_global_path_total_m"] = float(
        np.nansum(list(pose_paths.values()))
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


def build_feature_table(
    raw_dir: Path,
    fk_dir: Path,
    output_dir: Path,
    v3,
) -> pd.DataFrame:
    raw_files = sorted(raw_dir.rglob("*.csv"))
    fk_files = sorted(fk_dir.rglob("*_FK.csv"))
    fk_index = {normalise_fk_stem(path): path for path in fk_files}

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    missing_fk: list[dict[str, str]] = []

    print(f"Found {len(raw_files)} raw files and {len(fk_files)} FK outputs.")
    for i, raw_path in enumerate(raw_files, start=1):
        print(f"[v5 features {i}/{len(raw_files)}] {raw_path.name}")
        fk_path = fk_index.get(raw_path.stem)
        if fk_path is None:
            missing_fk.append({"raw_file": str(raw_path)})
            print("  MISSING FK OUTPUT")
            continue
        try:
            rows.append(extract_trial_features_v5(raw_path, fk_path, v3))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append({
                "raw_file": str(raw_path),
                "fk_file": str(fk_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    pd.DataFrame(errors).to_csv(output_dir / "dataset_v5_fk_processing_errors.csv", index=False)
    pd.DataFrame(missing_fk).to_csv(output_dir / "dataset_v5_fk_missing_outputs.csv", index=False)
    if not rows:
        raise ValueError("No operation was successfully processed.")

    table = pd.DataFrame(rows)
    duplicate = table.duplicated(v3.MERGE_KEYS, keep=False)
    if duplicate.any():
        path = output_dir / "dataset_v5_fk_duplicate_keys.csv"
        table.loc[duplicate, v3.MERGE_KEYS + ["file_name", "fk_file_name"]].to_csv(path, index=False)
        raise ValueError(f"Duplicate operation keys found. Inspect {path}")
    return table


def build_dataset(
    raw_dir: Path,
    fk_dir: Path,
    label_dir: Path,
    output_dir: Path,
    v3,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = build_feature_table(raw_dir, fk_dir, output_dir, v3)
    labels = v3.build_labels(label_dir)

    audit = features[v3.MERGE_KEYS + ["file_name", "fk_file_name"]].merge(
        labels[v3.MERGE_KEYS + ["target_score", "percentage_score"]],
        on=v3.MERGE_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    audit.to_csv(output_dir / "dataset_v5_fk_merge_audit.csv", index=False)

    dataset = features.merge(
        labels,
        on=v3.MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError("Merged v5 dataset is empty; inspect merge audit.")

    dataset = dataset.sort_values(v3.MERGE_KEYS).reset_index(drop=True)
    dataset_path = output_dir / "dataset_v5_fk.csv"
    dataset.to_csv(dataset_path, index=False)

    manifest = {
        "dataset_version": "v5_fk",
        "one_row_represents": "one session-aware scored operation",
        "raw_source": "original DVST telemetry",
        "cartesian_source": "supervisor FK optimiser True_Tip_X/Y/Z and True_Tip_Matrix",
        "coordinate_frame": "calibrated global/base-referenced output from FK script",
        "joint_features": "retained from original USMJointValues",
        "merge_keys": v3.MERGE_KEYS,
        "output": str(dataset_path),
    }
    (output_dir / "dataset_v5_fk_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nSaved: {dataset_path}")
    print(f"Shape: {dataset.shape}")
    print("\nMerge audit:")
    print(audit["_merge"].value_counts())
    print("\nTask counts:")
    print(dataset["task_clean"].value_counts())
    print("\nTarget summary:")
    print(dataset["target_score"].describe())
    return dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--fk-dir", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--v3-script", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.raw_dir, args.fk_dir, args.label_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.v3_script.exists():
        raise FileNotFoundError(args.v3_script)

    v3 = load_v3_module(args.v3_script.resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        build_dataset(
            args.raw_dir.resolve(),
            args.fk_dir.resolve(),
            args.label_dir.resolve(),
            args.output_dir.resolve(),
            v3,
        )


if __name__ == "__main__":
    main()
