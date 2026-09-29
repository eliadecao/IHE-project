from __future__ import annotations

"""
Build a leakage-safe, trial-level Griffin da Vinci Xi kinematic dataset.

Main corrections compared with dataset_v2:
1. Uses session Licence + Date in the merge key, so repeated participant/task/attempt
   recordings from different sessions cannot be assigned the wrong label.
2. Does NOT treat USM0 EndoscopePosition as a base-to-camera transform.
3. Uses EndoscopePosition only as a camera-relative PSM/tool pose (up to unknown scale).
4. Adds joint-space features from USMJointValues for every USM.
5. Keeps task-dependent M-GEARS domains and label-quality information.
6. Produces merge and data-quality audit files instead of silently dropping conflicts.

The script produces one row per scored operation.

Example:
    python build_dataset_v3.py --project-root "C:/path/to/project"

Or specify folders directly:
    python build_dataset_v3.py \
        --dvst-dir ".../Griffin_Training_Dataset/DVST_XI" \
        --label-dir ".../Griffin_Training_Dataset/M-GEARS" \
        --output-dir ".../data/processed"
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - rolling-mean fallback is used
    savgol_filter = None


# =============================================================================
# Configuration
# =============================================================================

ECM_USM = 0
ALL_USMS = (0, 1, 2, 3)
INSTRUMENT_USMS = tuple(u for u in ALL_USMS if u != ECM_USM)

EXPECTED_VECTOR_LENGTHS = {
    "SUSvalues": 4,
    "SUJvalues": 6,          # The supplied Xi CSVs contain six values per row.
    "USMJointValues": 10,
    "EndoscopePosition": 12,
}

# Smoothing is used only before derivatives/path features to reduce telemetry jitter.
SMOOTH_WINDOW_SECONDS = 0.15
SAVGOL_POLYORDER = 3

# Relative threshold because EndoscopePosition translation has an unresolved scale.
IDLE_THRESHOLD_FRACTION_OF_Q95 = 0.05
MIN_PAUSE_SECONDS = 0.20

# Alignment of two instrument trajectories on a common time grid.
PAIR_ALIGNMENT_TOLERANCE_IN_STEPS = 1.5

# Rotation-matrix QC. Matrices are projected to the nearest SO(3) matrix before use.
ROTATION_ORTHOGONALITY_TOL = 2e-2
ROTATION_DETERMINANT_TOL = 2e-2

# Per-joint features are useful for Ridge/Lasso, but can be disabled for a compact table.
INCLUDE_PER_JOINT_FEATURES = False

TARGET_COLUMN = "Total operation M-GEARS score"
PERCENTAGE_COLUMN = "Percentage score"
MGEARS_PREFIX = "M-GEARS - "

MERGE_KEYS = [
    "session_licence",
    "session_datetime",
    "role",
    "participant_id",
    "task_clean",
    "trial",
]

FILENAME_PATTERN = re.compile(
    r"^DVST_XI_"
    r"(?P<licence>.+?)_"
    r"(?P<session_datetime>\d{1,2}_\d{2}\.\d{2}\.\d{2})_"
    r"(?P<role>trainee|expert)_"
    r"(?P<participant_id>\d+)_"
    r"(?P<task>.+)_"
    r"(?P<trial>\d+)$",
    flags=re.IGNORECASE,
)


# =============================================================================
# Small utilities
# =============================================================================


def default_project_root() -> Path:
    """Preserve the original project layout when the script is under src/... ."""
    path = Path(__file__).resolve()
    if len(path.parents) >= 3:
        return path.parents[2]
    return Path.cwd()


def clean_task(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_role(value: object) -> str:
    return str(value).strip().lower()


def clean_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def clean_session_licence(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_session_datetime(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "", text)
    return text


def numeric_slug(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def parse_vector(value: object, expected_length: int) -> np.ndarray | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"[\[\],;]", " ", text)
    arr = np.fromstring(text, sep=" ", dtype=float)

    if arr.size != expected_length or not np.all(np.isfinite(arr)):
        return None
    return arr


def positive_median_diff(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = np.unique(values[np.isfinite(values)])
    if values.size < 2:
        return np.nan
    diffs = np.diff(np.sort(values))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else np.nan


def infer_timestamp_scale(timestamp_values: np.ndarray) -> tuple[float, str]:
    """Return multiplier to seconds and a readable unit label."""
    values = np.asarray(timestamp_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No valid timestamps are available.")

    median_abs = float(np.median(np.abs(values)))
    median_diff = positive_median_diff(values)

    # Epoch-like timestamps are easiest to identify by magnitude.
    if median_abs >= 1e14:
        return 1e-6, "microseconds"
    if median_abs >= 1e11:
        return 1e-3, "milliseconds"
    if median_abs >= 1e8:
        return 1.0, "seconds"

    # Fallback for relative timestamps.
    if np.isfinite(median_diff):
        if median_diff >= 1e3:
            return 1e-6, "microseconds_assumed"
        if median_diff >= 1.0:
            return 1e-3, "milliseconds_assumed"
        return 1.0, "seconds_assumed"

    return 1.0, "seconds_assumed"


def gradient(values: np.ndarray, time_seconds: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(time_seconds, dtype=float)
    if len(t) < 2:
        return np.full_like(values, np.nan, dtype=float)
    edge_order = 2 if len(t) >= 3 else 1
    return np.gradient(values, t, axis=0, edge_order=edge_order)


def smooth_values(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 5 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return values.copy()

    window = int(round(SMOOTH_WINDOW_SECONDS * sample_rate_hz))
    window = max(window, 5)
    if window % 2 == 0:
        window += 1
    if window > values.shape[0]:
        window = values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1
    if window < 5:
        return values.copy()

    if savgol_filter is not None:
        polyorder = min(SAVGOL_POLYORDER, window - 2)
        try:
            return savgol_filter(
                values,
                window_length=window,
                polyorder=polyorder,
                axis=0,
                mode="interp",
            )
        except Exception:
            pass

    return (
        pd.DataFrame(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def relative_idle_threshold(speed: np.ndarray) -> float:
    speed = np.asarray(speed, dtype=float)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return np.nan
    q95 = float(np.quantile(speed, 0.95))
    if q95 <= 0:
        return 0.0
    return IDLE_THRESHOLD_FRACTION_OF_Q95 * q95


def count_true_runs(
    mask: np.ndarray,
    time_seconds: np.ndarray,
    minimum_duration: float,
) -> tuple[int, float]:
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(time_seconds, dtype=float)
    if mask.size == 0 or t.size != mask.size:
        return 0, 0.0

    median_dt = positive_median_diff(t)
    if not np.isfinite(median_dt):
        median_dt = 0.0

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    count = 0
    total_duration = 0.0
    for start, end in zip(starts, ends):
        duration = max(0.0, float(t[end] - t[start] + median_dt))
        if duration >= minimum_duration:
            count += 1
            total_duration += duration
    return count, total_duration


def count_direction_reversals(velocity: np.ndarray) -> int:
    velocity = np.asarray(velocity, dtype=float)
    velocity = velocity[np.isfinite(velocity)]
    if velocity.size < 3:
        return 0

    threshold = relative_idle_threshold(np.abs(velocity))
    if not np.isfinite(threshold):
        return 0

    signs = np.sign(velocity[np.abs(velocity) > threshold])
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


# =============================================================================
# Filename metadata and raw telemetry loading
# =============================================================================


def extract_metadata_from_dvst_filename(file_path: Path) -> dict[str, object]:
    match = FILENAME_PATTERN.match(file_path.stem)
    if not match:
        raise ValueError(
            "Could not parse DVST filename. Expected a name like "
            "DVST_XI_6-8_Mar_2025_06_10.41.14_trainee_31_task_1.csv; "
            f"received: {file_path.name}"
        )

    groups = match.groupdict()
    metadata = {
        "file_name": file_path.name,
        "session_licence_raw": groups["licence"],
        "session_datetime_raw": groups["session_datetime"],
        "session_licence": clean_session_licence(groups["licence"]),
        "session_datetime": clean_session_datetime(groups["session_datetime"]),
        "role": clean_role(groups["role"]),
        "participant_id": clean_id(groups["participant_id"]),
        "task": groups["task"].strip(),
        "task_clean": clean_task(groups["task"]),
        "trial": clean_id(groups["trial"]),
    }
    metadata["recording_id"] = "|".join(str(metadata[key]) for key in MERGE_KEYS)
    return metadata


@dataclass
class RawTelemetry:
    data: pd.DataFrame
    qc: dict[str, object]


def load_raw_telemetry(file_path: Path) -> RawTelemetry:
    required_columns = [
        "TimeStamp",
        "USM",
        "SUSvalues",
        "SUJvalues",
        "USMJointValues",
        "EndoscopePosition",
    ]

    raw = pd.read_csv(file_path, usecols=lambda c: c in required_columns)
    missing = [column for column in required_columns if column not in raw.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    raw = raw[required_columns].copy()
    raw["timestamp_raw"] = pd.to_numeric(raw["TimeStamp"], errors="coerce")
    raw["usm"] = pd.to_numeric(raw["USM"], errors="coerce")
    raw = raw.dropna(subset=["timestamp_raw", "usm"])
    raw["usm"] = raw["usm"].astype(int)

    duplicate_timestamp_usm_rows = int(raw.duplicated(["timestamp_raw", "usm"]).sum())
    raw = raw.sort_values(["timestamp_raw", "usm"])
    raw = raw.drop_duplicates(["timestamp_raw", "usm"], keep="first")

    time_scale, timestamp_unit = infer_timestamp_scale(raw["timestamp_raw"].to_numpy())
    time_origin = float(raw["timestamp_raw"].min())
    raw["time_seconds"] = (raw["timestamp_raw"] - time_origin) * time_scale

    invalid_vector_counts: dict[str, int] = {}
    for column, expected_length in EXPECTED_VECTOR_LENGTHS.items():
        parsed_column = f"_{column}_array"
        raw[parsed_column] = [
            parse_vector(value, expected_length) for value in raw[column].tolist()
        ]
        invalid_vector_counts[column] = int(raw[parsed_column].isna().sum())

    unique_time = np.sort(raw["time_seconds"].unique())
    median_dt = positive_median_diff(unique_time)
    max_gap = float(np.max(np.diff(unique_time))) if unique_time.size >= 2 else np.nan
    duration = float(unique_time[-1] - unique_time[0]) if unique_time.size >= 2 else np.nan

    qc: dict[str, object] = {
        "qc_raw_rows_after_timestamp_cleaning": int(len(raw)),
        "qc_duplicate_timestamp_usm_rows_removed": duplicate_timestamp_usm_rows,
        "qc_unique_timestamps": int(unique_time.size),
        "qc_n_usms": int(raw["usm"].nunique()),
        "qc_usms_present": ",".join(map(str, sorted(raw["usm"].unique()))),
        "qc_timestamp_unit_inferred": timestamp_unit,
        "qc_median_dt_seconds": median_dt,
        "qc_sampling_rate_hz": safe_divide(1.0, median_dt),
        "qc_max_gap_seconds": max_gap,
        "duration_seconds": duration,
    }
    for column, invalid_count in invalid_vector_counts.items():
        qc[f"qc_invalid_{numeric_slug(column)}_rows"] = invalid_count

    return RawTelemetry(data=raw, qc=qc)


# =============================================================================
# Generic movement features
# =============================================================================


def compute_motion_summary(
    time_seconds: np.ndarray,
    values: np.ndarray,
    prefix: str,
    *,
    calculate_dimensionless_jerk: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    t = np.asarray(time_seconds, dtype=float)
    values = np.asarray(values, dtype=float)

    features: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}

    if values.ndim == 1:
        values = values[:, None]

    valid_rows = np.isfinite(t) & np.all(np.isfinite(values), axis=1)
    t = t[valid_rows]
    values = values[valid_rows]

    if t.size:
        order = np.argsort(t)
        t = t[order]
        values = values[order]
        keep = np.concatenate(([True], np.diff(t) > 0))
        t = t[keep]
        values = values[keep]

    features[f"{prefix}_n_samples"] = int(len(t))
    if len(t) < 3:
        return features, arrays

    median_dt = positive_median_diff(t)
    sample_rate = safe_divide(1.0, median_dt)
    duration = float(t[-1] - t[0])
    max_gap = float(np.max(np.diff(t)))

    smoothed = smooth_values(values, sample_rate)

    raw_step = np.linalg.norm(np.diff(values, axis=0), axis=1)
    smooth_step = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    raw_path = float(np.sum(raw_step))
    path = float(np.sum(smooth_step))
    displacement = float(np.linalg.norm(smoothed[-1] - smoothed[0]))

    velocity = gradient(smoothed, t)
    acceleration = gradient(velocity, t)
    jerk = gradient(acceleration, t)

    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    threshold = relative_idle_threshold(speed)
    idle_mask = speed <= threshold if np.isfinite(threshold) else np.zeros_like(speed, dtype=bool)
    pause_count, pause_duration = count_true_runs(idle_mask, t, MIN_PAUSE_SECONDS)

    features.update(
        {
            f"{prefix}_duration_seconds": duration,
            f"{prefix}_median_dt_seconds": median_dt,
            f"{prefix}_sample_rate_hz": sample_rate,
            f"{prefix}_max_gap_seconds": max_gap,
            f"{prefix}_raw_path_l2": raw_path,
            f"{prefix}_path_l2": path,
            f"{prefix}_net_displacement_l2": displacement,
            f"{prefix}_movement_economy": safe_divide(displacement, path),
            f"{prefix}_mean_speed_l2": float(np.mean(speed)),
            f"{prefix}_rms_speed_l2": float(np.sqrt(np.mean(speed ** 2))),
            f"{prefix}_max_speed_l2": float(np.max(speed)),
            f"{prefix}_mean_acceleration_l2": float(np.mean(acceleration_magnitude)),
            f"{prefix}_max_acceleration_l2": float(np.max(acceleration_magnitude)),
            f"{prefix}_mean_jerk_l2": float(np.mean(jerk_magnitude)),
            f"{prefix}_max_jerk_l2": float(np.max(jerk_magnitude)),
            f"{prefix}_idle_threshold_relative": threshold,
            f"{prefix}_idle_fraction_relative": float(np.mean(idle_mask)),
            f"{prefix}_pause_count": int(pause_count),
            f"{prefix}_pause_duration_seconds": float(pause_duration),
            f"{prefix}_pause_fraction": safe_divide(pause_duration, duration),
        }
    )

    jerk_squared = jerk_magnitude ** 2
    jerk_cost = float(np.trapezoid(jerk_squared, t))
    features[f"{prefix}_jerk_cost"] = jerk_cost

    if calculate_dimensionless_jerk and duration > 0 and path > 0 and jerk_cost >= 0:
        dimensionless_jerk = (duration ** 5 / path ** 2) * jerk_cost
        features[f"{prefix}_dimensionless_jerk"] = dimensionless_jerk
        features[f"{prefix}_log_dimensionless_jerk"] = -math.log(
            max(dimensionless_jerk, np.finfo(float).eps)
        )
    else:
        features[f"{prefix}_dimensionless_jerk"] = np.nan
        features[f"{prefix}_log_dimensionless_jerk"] = np.nan

    arrays.update(
        {
            "time": t,
            "raw_values": values,
            "values": smoothed,
            "velocity": velocity,
            "speed": speed,
            "acceleration": acceleration,
            "jerk": jerk,
            "idle_mask": idle_mask,
        }
    )
    return features, arrays


# =============================================================================
# Joint-space features
# =============================================================================


def unwrap_joint_values(joint_values: np.ndarray) -> np.ndarray:
    values = np.asarray(joint_values, dtype=float).copy()
    # Based on the extraction report: q0-q2 are revolute, q3 is insertion,
    # q4 is distal roll, and q5-q8 are distal articulation channels.
    revolute_like_indices = (0, 1, 2, 4, 5, 6, 7, 8)
    for index in revolute_like_indices:
        if index < values.shape[1]:
            values[:, index] = np.unwrap(values[:, index])
    return values


def compute_joint_features(
    usm_frame: pd.DataFrame,
    usm: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prefix = f"USM{usm}_joint"
    valid = usm_frame["_USMJointValues_array"].notna()
    frame = usm_frame.loc[valid].sort_values("time_seconds")

    if frame.empty:
        return {f"{prefix}_n_samples": 0}, {}

    t = frame["time_seconds"].to_numpy(dtype=float)
    q = np.vstack(frame["_USMJointValues_array"].to_numpy())
    q = unwrap_joint_values(q)

    features, arrays = compute_motion_summary(
        t,
        q,
        prefix,
        calculate_dimensionless_jerk=False,  # mixed joint units
    )

    if not arrays:
        return features, arrays

    q_smooth = arrays["values"]
    q_velocity = arrays["velocity"]

    # L1 joint activity is often easier to interpret than a mixed-unit L2 norm.
    abs_dq = np.abs(np.diff(q_smooth, axis=0))
    features[f"{prefix}_path_l1"] = float(np.sum(abs_dq))
    features[f"{prefix}_mean_joint_range"] = float(np.mean(np.ptp(q_smooth, axis=0)))
    features[f"{prefix}_max_joint_range"] = float(np.max(np.ptp(q_smooth, axis=0)))
    features[f"{prefix}_mean_abs_joint_velocity"] = float(np.mean(np.abs(q_velocity)))
    features[f"{prefix}_max_abs_joint_velocity"] = float(np.max(np.abs(q_velocity)))

    group_indices = {
        "base_revolute_q0_q2": (0, 1, 2),
        "insertion_q3": (3,),
        "tool_roll_q4": (4,),
        "distal_q5_q9": (5, 6, 7, 8, 9),
    }
    for group_name, indices in group_indices.items():
        indices = tuple(index for index in indices if index < q_smooth.shape[1])
        if not indices:
            continue
        group_values = q_smooth[:, indices]
        group_velocity = q_velocity[:, indices]
        group_path = float(np.sum(np.abs(np.diff(group_values, axis=0))))
        features[f"{prefix}_{group_name}_path_l1"] = group_path
        features[f"{prefix}_{group_name}_mean_range"] = float(
            np.mean(np.ptp(group_values, axis=0))
        )
        features[f"{prefix}_{group_name}_mean_abs_velocity"] = float(
            np.mean(np.abs(group_velocity))
        )

    if INCLUDE_PER_JOINT_FEATURES:
        for joint_index in range(q_smooth.shape[1]):
            joint = q_smooth[:, joint_index]
            joint_velocity = q_velocity[:, joint_index]
            joint_prefix = f"{prefix}_q{joint_index}"
            features[f"{joint_prefix}_range"] = float(np.ptp(joint))
            features[f"{joint_prefix}_std"] = float(np.std(joint))
            features[f"{joint_prefix}_total_variation"] = float(
                np.sum(np.abs(np.diff(joint)))
            )
            features[f"{joint_prefix}_mean_abs_velocity"] = float(
                np.mean(np.abs(joint_velocity))
            )
            features[f"{joint_prefix}_max_abs_velocity"] = float(
                np.max(np.abs(joint_velocity))
            )
            features[f"{joint_prefix}_reversal_count"] = count_direction_reversals(
                joint_velocity
            )

    return features, arrays


# =============================================================================
# Camera-relative pose features for instrument arms
# =============================================================================


def project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_between(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_pose_features(
    usm_frame: pd.DataFrame,
    usm: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    EndoscopePosition is treated as tool pose relative to the camera optical frame.
    Translation is in unresolved proprietary/relative units, not asserted millimetres.
    """
    prefix = f"USM{usm}_pose_rel"
    valid = usm_frame["_EndoscopePosition_array"].notna()
    frame = usm_frame.loc[valid].sort_values("time_seconds")

    if frame.empty:
        return {f"{prefix}_n_samples": 0}, {}

    t = frame["time_seconds"].to_numpy(dtype=float)
    pose_vectors = np.vstack(frame["_EndoscopePosition_array"].to_numpy())
    translation = pose_vectors[:, :3]
    raw_rotations = pose_vectors[:, 3:].reshape(-1, 3, 3)

    features, arrays = compute_motion_summary(
        t,
        translation,
        prefix,
        calculate_dimensionless_jerk=True,
    )

    if not arrays:
        return features, arrays

    smoothed_translation = arrays["values"]
    ranges = np.ptp(smoothed_translation, axis=0)
    x_range, y_range, z_range = map(float, ranges)
    path = features.get(f"{prefix}_path_l2", np.nan)

    features.update(
        {
            f"{prefix}_x_range": x_range,
            f"{prefix}_y_range": y_range,
            f"{prefix}_z_range": z_range,
            f"{prefix}_workspace_box_volume": x_range * y_range * z_range,
            f"{prefix}_economy_area_xy": safe_divide(math.sqrt(x_range * y_range), path),
            f"{prefix}_economy_area_xz": safe_divide(math.sqrt(x_range * z_range), path),
            f"{prefix}_economy_area_yz": safe_divide(math.sqrt(y_range * z_range), path),
            f"{prefix}_economy_volume": safe_divide(
                np.cbrt(x_range * y_range * z_range), path
            ),
        }
    )

    orthogonality_errors = np.array(
        [np.linalg.norm(r.T @ r - np.eye(3), ord="fro") for r in raw_rotations],
        dtype=float,
    )
    determinants = np.linalg.det(raw_rotations)
    invalid_rotation = (
        (orthogonality_errors > ROTATION_ORTHOGONALITY_TOL)
        | (np.abs(determinants - 1.0) > ROTATION_DETERMINANT_TOL)
    )
    rotations = np.stack([project_to_rotation_matrix(r) for r in raw_rotations])

    angles = np.array(
        [rotation_angle_between(rotations[i], rotations[i + 1]) for i in range(len(rotations) - 1)],
        dtype=float,
    )
    dt = np.diff(t)
    valid_steps = dt > 0
    angles = angles[valid_steps]
    dt = dt[valid_steps]
    angular_speed = angles / dt if dt.size else np.array([], dtype=float)

    if angular_speed.size >= 2:
        angular_speed_time = (t[:-1][valid_steps] + t[1:][valid_steps]) / 2.0
        angular_acceleration = np.gradient(
            angular_speed,
            angular_speed_time,
            edge_order=2 if angular_speed.size >= 3 else 1,
        )
    else:
        angular_acceleration = np.array([], dtype=float)

    features.update(
        {
            f"{prefix}_rotation_orthogonality_error_mean": float(
                np.mean(orthogonality_errors)
            ),
            f"{prefix}_rotation_orthogonality_error_max": float(
                np.max(orthogonality_errors)
            ),
            f"{prefix}_rotation_determinant_mean": float(np.mean(determinants)),
            f"{prefix}_rotation_invalid_fraction": float(np.mean(invalid_rotation)),
            f"{prefix}_rotation_path_radians": float(np.sum(angles)) if angles.size else np.nan,
            f"{prefix}_mean_angular_speed_rad_s": float(np.mean(angular_speed))
            if angular_speed.size
            else np.nan,
            f"{prefix}_max_angular_speed_rad_s": float(np.max(angular_speed))
            if angular_speed.size
            else np.nan,
            f"{prefix}_mean_abs_angular_acceleration_rad_s2": float(
                np.mean(np.abs(angular_acceleration))
            )
            if angular_acceleration.size
            else np.nan,
            f"{prefix}_max_abs_angular_acceleration_rad_s2": float(
                np.max(np.abs(angular_acceleration))
            )
            if angular_acceleration.size
            else np.nan,
        }
    )

    arrays["rotations"] = rotations
    return features, arrays


# =============================================================================
# Setup-structure QC features (not recommended as primary skill predictors)
# =============================================================================


def compute_setup_qc_features(raw: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}

    # SUS is shared across arms; use one row per timestamp.
    sus_frame = raw.drop_duplicates("time_seconds")
    sus_values = [value for value in sus_frame["_SUSvalues_array"] if value is not None]
    if sus_values:
        sus = np.vstack(sus_values)
        for index in range(sus.shape[1]):
            features[f"qc_SUS{index}_median"] = float(np.median(sus[:, index]))
            features[f"qc_SUS{index}_range"] = float(np.ptp(sus[:, index]))

    for usm in sorted(raw["usm"].unique()):
        frame = raw[raw["usm"] == usm]
        suj_values = [value for value in frame["_SUJvalues_array"] if value is not None]
        if not suj_values:
            continue
        suj = np.vstack(suj_values)
        for index in range(suj.shape[1]):
            features[f"qc_USM{usm}_SUJ{index}_median"] = float(np.median(suj[:, index]))
            features[f"qc_USM{usm}_SUJ{index}_range"] = float(np.ptp(suj[:, index]))

    return features


# =============================================================================
# Bimanual coordination
# =============================================================================


def nearest_distance_to_samples(grid: np.ndarray, sample_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sample_time, grid)
    left_indices = np.clip(indices - 1, 0, len(sample_time) - 1)
    right_indices = np.clip(indices, 0, len(sample_time) - 1)
    left_distance = np.abs(grid - sample_time[left_indices])
    right_distance = np.abs(grid - sample_time[right_indices])
    return np.minimum(left_distance, right_distance)


def interpolate_trajectory(
    trajectory: dict[str, np.ndarray],
    grid: np.ndarray,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = trajectory["time"]
    position = trajectory["values"]
    interpolated = np.column_stack(
        [np.interp(grid, time, position[:, dimension]) for dimension in range(position.shape[1])]
    )
    valid = nearest_distance_to_samples(grid, time) <= tolerance_seconds
    return interpolated, valid


def compute_pair_coordination_features(
    trajectory_a: dict[str, np.ndarray],
    trajectory_b: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    features: dict[str, float] = {}

    time_a = trajectory_a.get("time")
    time_b = trajectory_b.get("time")
    if time_a is None or time_b is None or len(time_a) < 3 or len(time_b) < 3:
        features[f"{prefix}_n_aligned_samples"] = 0
        return features

    start = max(float(time_a[0]), float(time_b[0]))
    end = min(float(time_a[-1]), float(time_b[-1]))
    dt_a = positive_median_diff(time_a)
    dt_b = positive_median_diff(time_b)
    common_dt = max(dt_a, dt_b)

    if not np.isfinite(common_dt) or common_dt <= 0 or end <= start:
        features[f"{prefix}_n_aligned_samples"] = 0
        return features

    grid = np.arange(start, end + common_dt / 2.0, common_dt)
    tolerance = PAIR_ALIGNMENT_TOLERANCE_IN_STEPS * common_dt
    position_a, valid_a = interpolate_trajectory(trajectory_a, grid, tolerance)
    position_b, valid_b = interpolate_trajectory(trajectory_b, grid, tolerance)
    valid = valid_a & valid_b

    features[f"{prefix}_alignment_coverage"] = float(np.mean(valid)) if valid.size else 0.0
    grid = grid[valid]
    position_a = position_a[valid]
    position_b = position_b[valid]
    features[f"{prefix}_n_aligned_samples"] = int(len(grid))

    if len(grid) < 3:
        return features

    distance = np.linalg.norm(position_a - position_b, axis=1)
    velocity_a = gradient(position_a, grid)
    velocity_b = gradient(position_b, grid)
    speed_a = np.linalg.norm(velocity_a, axis=1)
    speed_b = np.linalg.norm(velocity_b, axis=1)

    threshold_a = relative_idle_threshold(speed_a)
    threshold_b = relative_idle_threshold(speed_b)
    active_a = speed_a > threshold_a if np.isfinite(threshold_a) else np.zeros_like(speed_a, bool)
    active_b = speed_b > threshold_b if np.isfinite(threshold_b) else np.zeros_like(speed_b, bool)

    if np.std(speed_a) > 0 and np.std(speed_b) > 0:
        speed_correlation = float(np.corrcoef(speed_a, speed_b)[0, 1])
    else:
        speed_correlation = np.nan

    path_a = float(np.sum(np.linalg.norm(np.diff(position_a, axis=0), axis=1)))
    path_b = float(np.sum(np.linalg.norm(np.diff(position_b, axis=0), axis=1)))

    features.update(
        {
            f"{prefix}_mean_distance_rel": float(np.mean(distance)),
            f"{prefix}_std_distance_rel": float(np.std(distance)),
            f"{prefix}_min_distance_rel": float(np.min(distance)),
            f"{prefix}_max_distance_rel": float(np.max(distance)),
            f"{prefix}_distance_range_rel": float(np.ptp(distance)),
            f"{prefix}_mean_abs_relative_speed": float(np.mean(np.abs(speed_a - speed_b))),
            f"{prefix}_speed_correlation": speed_correlation,
            f"{prefix}_simultaneous_active_fraction": float(np.mean(active_a & active_b)),
            f"{prefix}_one_arm_only_active_fraction": float(np.mean(active_a ^ active_b)),
            f"{prefix}_both_idle_fraction": float(np.mean(~active_a & ~active_b)),
            f"{prefix}_path_length_a_rel": path_a,
            f"{prefix}_path_length_b_rel": path_b,
            f"{prefix}_path_length_imbalance": safe_divide(abs(path_a - path_b), path_a + path_b),
        }
    )
    return features


# =============================================================================
# One-operation feature extraction
# =============================================================================


def extract_trial_features_v3(file_path: Path) -> dict[str, object]:
    metadata = extract_metadata_from_dvst_filename(file_path)
    loaded = load_raw_telemetry(file_path)
    raw = loaded.data

    features: dict[str, object] = {}
    features.update(metadata)
    features.update(loaded.qc)
    features.update(compute_setup_qc_features(raw))

    pose_trajectories: dict[int, dict[str, np.ndarray]] = {}
    pose_paths: dict[int, float] = {}
    joint_paths: dict[int, float] = {}

    for usm in sorted(raw["usm"].unique()):
        usm_frame = raw[raw["usm"] == usm]

        joint_features, joint_arrays = compute_joint_features(usm_frame, usm)
        features.update(joint_features)
        joint_path = joint_features.get(f"USM{usm}_joint_path_l1", np.nan)
        joint_paths[usm] = float(joint_path) if np.isfinite(joint_path) else np.nan

        if usm == ECM_USM:
            # EndoscopePosition is explicitly not used for ECM absolute position.
            features[f"USM{usm}_pose_rel_skipped_as_ecm"] = 1
            continue

        pose_features, pose_arrays = compute_pose_features(usm_frame, usm)
        features.update(pose_features)
        if pose_arrays:
            pose_trajectories[usm] = pose_arrays
            pose_path = pose_features.get(f"USM{usm}_pose_rel_path_l2", np.nan)
            pose_paths[usm] = float(pose_path) if np.isfinite(pose_path) else np.nan

    features["all_usm_joint_path_l1_total"] = float(
        np.nansum(list(joint_paths.values()))
    )
    features["all_psm_pose_rel_path_total"] = float(
        np.nansum(list(pose_paths.values()))
    )

    # Relative activity flags among PSMs. These flags are for choosing the active pair,
    # not absolute physical activity thresholds.
    finite_pose_paths = {u: p for u, p in pose_paths.items() if np.isfinite(p)}
    max_pose_path = max(finite_pose_paths.values(), default=0.0)
    active_psms: list[int] = []
    for usm in INSTRUMENT_USMS:
        path = finite_pose_paths.get(usm, np.nan)
        is_active = bool(
            np.isfinite(path)
            and max_pose_path > 0
            and path >= 0.02 * max_pose_path
        )
        features[f"USM{usm}_pose_rel_active"] = int(is_active)
        if is_active:
            active_psms.append(usm)
    features["active_psm_count"] = len(active_psms)

    # Every PSM pair is retained. The two largest-path PSMs also receive generic
    # active_pair_* columns, useful before left/right role mapping is confirmed.
    pair_feature_cache: dict[tuple[int, int], dict[str, float]] = {}
    for usm_a, usm_b in combinations(INSTRUMENT_USMS, 2):
        prefix = f"USM{usm_a}_USM{usm_b}_pose_rel"
        if usm_a in pose_trajectories and usm_b in pose_trajectories:
            pair_features = compute_pair_coordination_features(
                pose_trajectories[usm_a], pose_trajectories[usm_b], prefix
            )
        else:
            pair_features = {f"{prefix}_n_aligned_samples": 0}
        features.update(pair_features)
        pair_feature_cache[(usm_a, usm_b)] = pair_features

    ranked_psms = sorted(
        finite_pose_paths,
        key=lambda usm: finite_pose_paths[usm],
        reverse=True,
    )
    if len(ranked_psms) >= 2:
        active_a, active_b = sorted(ranked_psms[:2])
        features["active_pair_usm_a"] = active_a
        features["active_pair_usm_b"] = active_b
        pair_prefix = f"USM{active_a}_USM{active_b}_pose_rel"
        for key, value in pair_feature_cache.get((active_a, active_b), {}).items():
            suffix = key.removeprefix(pair_prefix + "_")
            features[f"active_pair_{suffix}"] = value
    else:
        features["active_pair_usm_a"] = np.nan
        features["active_pair_usm_b"] = np.nan

    return features


# =============================================================================
# Labels: session-aware merge and multiple-record aggregation
# =============================================================================


def build_labels(label_dir: Path) -> pd.DataFrame:
    label_files = sorted(label_dir.rglob("*.csv"))
    if not label_files:
        raise ValueError(f"No label CSV files found under {label_dir}")

    frames: list[pd.DataFrame] = []
    print(f"\nFound {len(label_files)} label files.")

    for index, file_path in enumerate(label_files, start=1):
        print(f"[labels {index}/{len(label_files)}] {file_path.name}")
        try:
            frame = pd.read_csv(file_path)
            frame["_label_source_file"] = file_path.name
            frames.append(frame)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not frames:
        raise ValueError("No readable label CSV files were found.")

    labels = pd.concat(frames, ignore_index=True, sort=False)
    required = ["Licence", "Date", "User", "Participant ID", "Task", "Attempt", TARGET_COLUMN]
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise ValueError(f"Label tables are missing required columns: {missing}")

    labels["session_licence"] = labels["Licence"].apply(clean_session_licence)
    labels["session_datetime"] = labels["Date"].apply(clean_session_datetime)
    labels["role"] = labels["User"].apply(clean_role)
    labels["participant_id"] = labels["Participant ID"].apply(clean_id)
    labels["task_clean"] = labels["Task"].apply(clean_task)
    labels["trial"] = labels["Attempt"].apply(clean_id)

    domain_columns = [column for column in labels.columns if column.startswith(MGEARS_PREFIX)]
    numeric_label_columns = domain_columns + [TARGET_COLUMN]
    if PERCENTAGE_COLUMN in labels.columns:
        numeric_label_columns.append(PERCENTAGE_COLUMN)

    for column in numeric_label_columns:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")

    output_rows: list[dict[str, object]] = []
    grouped = labels.groupby(MERGE_KEYS, dropna=False, sort=False)

    for key_values, group in grouped:
        row = dict(zip(MERGE_KEYS, key_values))
        row["label_recording_id"] = "|".join(str(row[key]) for key in MERGE_KEYS)
        row["label_source_files"] = ";".join(sorted(set(group["_label_source_file"].astype(str))))
        row["n_label_records"] = int(len(group))

        target_values = group[TARGET_COLUMN].dropna().to_numpy(dtype=float)
        row["target_score"] = float(np.mean(target_values)) if target_values.size else np.nan
        row["target_score_std"] = (
            float(np.std(target_values, ddof=1)) if target_values.size >= 2 else np.nan
        )
        row["n_unique_target_scores"] = int(np.unique(target_values).size)
        row["label_disagreement_flag"] = int(np.unique(target_values).size > 1)

        if PERCENTAGE_COLUMN in group.columns:
            values = group[PERCENTAGE_COLUMN].dropna().to_numpy(dtype=float)
            row["percentage_score"] = float(np.mean(values)) if values.size else np.nan
            row["percentage_score_std"] = (
                float(np.std(values, ddof=1)) if values.size >= 2 else np.nan
            )
        else:
            row["percentage_score"] = np.nan
            row["percentage_score_std"] = np.nan

        row["target_fraction"] = (
            row["percentage_score"] / 100.0
            if np.isfinite(row["percentage_score"])
            else np.nan
        )

        domain_output_names: list[str] = []
        for column in domain_columns:
            output_name = "mgears_" + numeric_slug(column[len(MGEARS_PREFIX):])
            values = group[column].dropna().to_numpy(dtype=float)
            row[output_name] = float(np.mean(values)) if values.size else np.nan
            row[output_name + "_std"] = (
                float(np.std(values, ddof=1)) if values.size >= 2 else np.nan
            )
            domain_output_names.append(output_name)

        applicable_domain_values = [
            row[name] for name in domain_output_names if np.isfinite(row[name])
        ]
        row["n_applicable_mgears_domains"] = len(applicable_domain_values)
        row["mgears_domain_score_sum"] = (
            float(np.sum(applicable_domain_values)) if applicable_domain_values else np.nan
        )
        row["target_minus_domain_sum"] = (
            row["target_score"] - row["mgears_domain_score_sum"]
            if np.isfinite(row["target_score"]) and np.isfinite(row["mgears_domain_score_sum"])
            else np.nan
        )
        if (
            np.isfinite(row["target_score"])
            and np.isfinite(row["percentage_score"])
            and row["percentage_score"] > 0
        ):
            row["inferred_max_score"] = row["target_score"] / (
                row["percentage_score"] / 100.0
            )
        else:
            row["inferred_max_score"] = np.nan

        output_rows.append(row)

    label_table = pd.DataFrame(output_rows)
    label_table = label_table.dropna(subset=["target_score"])

    if label_table.duplicated(MERGE_KEYS).any():
        raise ValueError("Aggregated label table still contains duplicate full session keys.")

    print(f"Aggregated label table shape: {label_table.shape}")
    return label_table


# =============================================================================
# Build X and final dataset
# =============================================================================


def build_feature_table(dvst_dir: Path, error_log_path: Path) -> pd.DataFrame:
    dvst_files = sorted(dvst_dir.rglob("*.csv"))
    if not dvst_files:
        raise ValueError(f"No DVST CSV files found under {dvst_dir}")

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    print(f"Found {len(dvst_files)} DVST files.")

    for index, file_path in enumerate(dvst_files, start=1):
        print(f"[features {index}/{len(dvst_files)}] {file_path.name}")
        try:
            rows.append(extract_trial_features_v3(file_path))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append({"file_name": file_path.name, "error": repr(exc)})

    pd.DataFrame(errors).to_csv(error_log_path, index=False)
    if not rows:
        raise ValueError("No DVST files were successfully processed.")

    feature_table = pd.DataFrame(rows)
    duplicate_mask = feature_table.duplicated(MERGE_KEYS, keep=False)
    if duplicate_mask.any():
        duplicates = feature_table.loc[duplicate_mask, MERGE_KEYS + ["file_name"]]
        duplicate_path = error_log_path.with_name("dataset_v3_duplicate_dvst_keys.csv")
        duplicates.to_csv(duplicate_path, index=False)
        raise ValueError(
            "More than one DVST file has the same full session-aware operation key. "
            f"Inspect {duplicate_path} before continuing."
        )

    print(f"Feature table shape: {feature_table.shape}")
    return feature_table


def build_dataset_v3(
    dvst_dir: Path,
    label_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = output_dir / "dataset_v3_processing_errors.csv"
    feature_table = build_feature_table(dvst_dir, error_log_path)
    label_table = build_labels(label_dir)

    audit = feature_table[MERGE_KEYS + ["file_name"]].merge(
        label_table[MERGE_KEYS + ["target_score", "percentage_score"]],
        on=MERGE_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    audit.to_csv(output_dir / "dataset_v3_merge_audit.csv", index=False)

    unmatched_dvst = audit[audit["_merge"] == "left_only"].copy()
    unmatched_labels = audit[audit["_merge"] == "right_only"].copy()
    unmatched_dvst.to_csv(output_dir / "dataset_v3_unmatched_dvst.csv", index=False)
    unmatched_labels.to_csv(output_dir / "dataset_v3_unmatched_labels.csv", index=False)

    print("\nMerge audit:")
    print(audit["_merge"].value_counts())

    dataset = feature_table.merge(
        label_table,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError("The merged dataset is empty. Inspect dataset_v3_merge_audit.csv.")

    dataset = dataset.sort_values(
        ["session_licence", "session_datetime", "role", "participant_id", "task_clean", "trial"]
    ).reset_index(drop=True)

    dataset_path = output_dir / "dataset_v3.csv"
    dataset.to_csv(dataset_path, index=False)

    # Save a small machine-readable note for later modelling.
    manifest = {
        "dataset_version": "v3",
        "one_row_represents": "one session-aware scored operation",
        "merge_keys": MERGE_KEYS,
        "ecm_usm_assumption": ECM_USM,
        "instrument_usms": list(INSTRUMENT_USMS),
        "pose_interpretation": (
            "EndoscopePosition for PSMs is camera-relative and may have an unknown scale. "
            "No global/base-frame or millimetre claim is made."
        ),
        "recommended_cross_task_target": "percentage_score or target_fraction",
        "recommended_same_task_target": "target_score may be used within a task",
        "columns_not_recommended_as_primary_skill_features": [
            "metadata columns",
            "label columns",
            "columns beginning qc_ (setup/session quality-control variables)",
            "active_pair_usm_a and active_pair_usm_b (identifiers)",
        ],
    }
    with open(output_dir / "dataset_v3_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved dataset: {dataset_path}")
    print(f"Final shape: {dataset.shape}")
    print("\nTask counts:")
    print(dataset["task_clean"].value_counts())
    print("\nTarget summary:")
    print(dataset["target_score"].describe())
    if "percentage_score" in dataset.columns:
        print("\nPercentage-score summary:")
        print(dataset["percentage_score"].describe())

    return dataset


# =============================================================================
# Command line
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--dvst-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.resolve()

    dvst_dir = (
        args.dvst_dir.resolve()
        if args.dvst_dir is not None
        else project_root / "Kinematics/dataset/Griffin_Training_Dataset/DVST_XI"
    )
    label_dir = (
        args.label_dir.resolve()
        if args.label_dir is not None
        else project_root / "Kinematics/dataset/Griffin_Training_Dataset/M-GEARS"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "data/processed"
    )

    print(f"Project root: {project_root}")
    print(f"DVST directory: {dvst_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Output directory: {output_dir}")
    print(f"ECM_USM assumption: {ECM_USM}")

    if not dvst_dir.exists():
        raise FileNotFoundError(f"DVST directory does not exist: {dvst_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        build_dataset_v3(dvst_dir, label_dir, output_dir)


if __name__ == "__main__":
    main()