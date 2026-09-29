from pathlib import Path
import re

import numpy as np
import pandas as pd


# =========================
# Paths
# =========================

project_root = Path(__file__).resolve().parent.parent.parent

DVST_DIR = project_root / "Kinematics/dataset/Griffin_Training_Dataset/DVST_XI"
LABEL_DIR = project_root / "Kinematics/dataset/Griffin_Training_Dataset/M-GEARS"

OUTPUT_PATH = project_root / "data/processed/dataset_v2.csv"

TARGET_COLUMN = "Total operation M-GEARS score"


# =========================
# Basic cleaning helpers
# =========================

def clean_task(task):
    return (
        str(task)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def clean_id(x):
    if pd.isna(x):
        return None
    return str(int(float(x)))


def extract_metadata_from_dvst_filename(file_path: Path):
    """
    Example:
    DVST_XI_6-8_Mar_2025_06_10.41.14_trainee_31_ring rollercoaster_1.csv
    """

    stem = file_path.stem

    match = re.search(
        r"_(trainee|expert)_(\d+)_(.+)_(\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(f"Could not parse filename: {file_path.name}")

    role = match.group(1).lower()
    participant_id = match.group(2)
    task = match.group(3).strip()
    trial = match.group(4)

    return {
        "file_name": file_path.name,
        "role": role,
        "participant_id": participant_id,
        "task": task,
        "task_clean": clean_task(task),
        "trial": trial,
    }


# =========================
# Pose parsing
# =========================

def parse_endoscope_position_to_matrix(value):
    """
    EndoscopePosition has 12 values:
    first 3 = translation x, y, z
    next 9 = 3x3 rotation matrix

    Return:
    4x4 homogeneous transformation matrix
    """

    arr = np.fromstring(str(value), sep=" ")

    if len(arr) != 12:
        return None

    matrix = np.eye(4)

    # Translation
    matrix[0, 3] = arr[0]
    matrix[1, 3] = arr[1]
    matrix[2, 3] = arr[2]

    # Rotation
    matrix[:3, :3] = arr[3:].reshape(3, 3)

    return matrix


def load_pose_table(file_path: Path):
    """
    Load one DVST_XI file and parse EndoscopePosition into pose matrices.
    """

    raw = pd.read_csv(file_path)

    required_cols = ["TimeStamp", "USM", "EndoscopePosition"]
    missing = [c for c in required_cols if c not in raw.columns]

    if missing:
        raise ValueError(f"{file_path.name} missing columns: {missing}")

    raw = raw[required_cols].copy()

    raw["timestamp"] = pd.to_numeric(raw["TimeStamp"], errors="coerce")
    raw["usm"] = pd.to_numeric(raw["USM"], errors="coerce")

    raw = raw.dropna(subset=["timestamp", "usm"])
    raw["usm"] = raw["usm"].astype(int)

    raw["pose_matrix"] = raw["EndoscopePosition"].apply(parse_endoscope_position_to_matrix)

    raw = raw.dropna(subset=["pose_matrix"])

    return raw[["timestamp", "usm", "pose_matrix"]]


# =========================
# Coordinate transformation
# =========================

def build_base_frame_trajectories(pose_df: pd.DataFrame):
    """
    According to the previous report:
    - USM0/endoscope pose is relative to robot base.
    - USM1-USM3/tool poses are relative to endoscope frame.

    Therefore:
    T_base_tool = T_base_endoscope @ T_endoscope_tool

    Output:
    DataFrame with timestamp, usm, x, y, z, and base-frame pose matrix.
    """

    rows = []

    for timestamp, group in pose_df.groupby("timestamp"):
        group = group.sort_values("usm")

        usm0 = group[group["usm"] == 0]

        if usm0.empty:
            continue

        T_base_endo = usm0.iloc[0]["pose_matrix"]

        for _, row in group.iterrows():
            usm = row["usm"]
            T = row["pose_matrix"]

            if usm == 0:
                T_base = T
            else:
                T_base = T_base_endo @ T

            x, y, z = T_base[:3, 3]

            rows.append({
                "timestamp": timestamp,
                "usm": usm,
                "x": x,
                "y": y,
                "z": z,
                "pose_matrix": T_base,
            })

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["timestamp", "usm", "x", "y", "z", "pose_matrix"]
        )

    return pd.DataFrame(rows)


# =========================
# Feature extraction helpers
# =========================

def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def rotation_angle_between(R1, R2):
    """
    Calculate rotation angle between two 3x3 rotation matrices.
    Output is in radians.
    """

    R_delta = R1.T @ R2
    trace_value = np.trace(R_delta)
    cos_angle = (trace_value - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def compute_usm_position_features(usm_df: pd.DataFrame, prefix: str):
    """
    Calculate interpretable kinematic features for one USM.

    Includes:
    - duration
    - path length
    - speed
    - acceleration
    - jerk / smoothness
    - workspace range
    - economy of area / volume
    - rotational velocity / acceleration
    """

    features = {}

    usm_df = usm_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    t = usm_df["timestamp"].to_numpy(dtype=float) / 1000.0
    pos = usm_df[["x", "y", "z"]].to_numpy(dtype=float)
    pose_matrices = usm_df["pose_matrix"].to_list()

    features[f"{prefix}_n_samples"] = len(usm_df)

    nan_feature_names = [
        "duration_seconds",
        "path_length",
        "mean_speed",
        "max_speed",
        "mean_acceleration",
        "max_acceleration",
        "mean_jerk",
        "max_jerk",
        "jerk_cost",
        "normalized_jerk",
        "smoothness_index",
        "idle_fraction",
        "x_range",
        "y_range",
        "z_range",
        "EA_xy",
        "EA_xz",
        "EA_yz",
        "EV",
        "rotation_path_length",
        "mean_rotational_velocity",
        "max_rotational_velocity",
        "mean_rotational_acceleration",
        "max_rotational_acceleration",
    ]

    if len(usm_df) < 2:
        for name in nan_feature_names:
            features[f"{prefix}_{name}"] = np.nan
        return features

    duration = t[-1] - t[0]
    dt = np.diff(t)

    diffs = np.diff(pos, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)

    valid = dt > 0

    if valid.sum() == 0:
        for name in nan_feature_names:
            features[f"{prefix}_{name}"] = np.nan
        return features

    valid_dt = dt[valid]
    valid_diffs = diffs[valid]
    valid_step_distances = step_distances[valid]

    # Velocity
    velocity_vectors = valid_diffs / valid_dt[:, None]
    speed = np.linalg.norm(velocity_vectors, axis=1)

    # Acceleration
    if len(velocity_vectors) >= 2:
        acceleration_dt = (valid_dt[1:] + valid_dt[:-1]) / 2
        acceleration_vectors = np.diff(velocity_vectors, axis=0) / acceleration_dt[:, None]
        acceleration_magnitude = np.linalg.norm(acceleration_vectors, axis=1)
    else:
        acceleration_vectors = np.empty((0, 3))
        acceleration_magnitude = np.array([])

    # Jerk
    if len(acceleration_vectors) >= 2:
        jerk_dt = (valid_dt[2:] + valid_dt[1:-1]) / 2
        jerk_vectors = np.diff(acceleration_vectors, axis=0) / jerk_dt[:, None]
        jerk_magnitude = np.linalg.norm(jerk_vectors, axis=1)
    else:
        jerk_dt = np.array([])
        jerk_magnitude = np.array([])

    path_length = valid_step_distances.sum()

    x_range = np.nanmax(pos[:, 0]) - np.nanmin(pos[:, 0])
    y_range = np.nanmax(pos[:, 1]) - np.nanmin(pos[:, 1])
    z_range = np.nanmax(pos[:, 2]) - np.nanmin(pos[:, 2])

    # Economy of area / volume
    EA_xy = safe_divide(np.sqrt(x_range * y_range), path_length)
    EA_xz = safe_divide(np.sqrt(x_range * z_range), path_length)
    EA_yz = safe_divide(np.sqrt(y_range * z_range), path_length)
    EV = safe_divide(np.cbrt(x_range * y_range * z_range), path_length)

    # Idle threshold may be tuned later
    idle_threshold = 1e-4

    features[f"{prefix}_duration_seconds"] = duration
    features[f"{prefix}_path_length"] = path_length
    features[f"{prefix}_mean_speed"] = np.nanmean(speed) if len(speed) else np.nan
    features[f"{prefix}_max_speed"] = np.nanmax(speed) if len(speed) else np.nan

    features[f"{prefix}_mean_acceleration"] = (
        np.nanmean(acceleration_magnitude) if len(acceleration_magnitude) else np.nan
    )
    features[f"{prefix}_max_acceleration"] = (
        np.nanmax(acceleration_magnitude) if len(acceleration_magnitude) else np.nan
    )

    features[f"{prefix}_mean_jerk"] = (
        np.nanmean(jerk_magnitude) if len(jerk_magnitude) else np.nan
    )
    features[f"{prefix}_max_jerk"] = (
        np.nanmax(jerk_magnitude) if len(jerk_magnitude) else np.nan
    )

    if len(jerk_magnitude) and len(jerk_dt):
        jerk_cost = np.sum((jerk_magnitude ** 2) * jerk_dt)
    else:
        jerk_cost = np.nan

    features[f"{prefix}_jerk_cost"] = jerk_cost

    if duration > 0 and path_length > 0 and not pd.isna(jerk_cost):
        normalized_jerk = (duration ** 5 / path_length ** 2) * jerk_cost
        smoothness_index = 1 / (1 + np.log1p(normalized_jerk))
    else:
        normalized_jerk = np.nan
        smoothness_index = np.nan

    features[f"{prefix}_normalized_jerk"] = normalized_jerk
    features[f"{prefix}_smoothness_index"] = smoothness_index

    features[f"{prefix}_idle_fraction"] = (
        np.mean(speed < idle_threshold) if len(speed) else np.nan
    )

    features[f"{prefix}_x_range"] = x_range
    features[f"{prefix}_y_range"] = y_range
    features[f"{prefix}_z_range"] = z_range

    features[f"{prefix}_EA_xy"] = EA_xy
    features[f"{prefix}_EA_xz"] = EA_xz
    features[f"{prefix}_EA_yz"] = EA_yz
    features[f"{prefix}_EV"] = EV

    # Rotation features
    rotation_angles = []

    for i in range(len(pose_matrices) - 1):
        R1 = pose_matrices[i][:3, :3]
        R2 = pose_matrices[i + 1][:3, :3]
        angle = rotation_angle_between(R1, R2)
        rotation_angles.append(angle)

    rotation_angles = np.array(rotation_angles, dtype=float)

    valid_rotation_angles = rotation_angles[valid]
    rotational_velocity = valid_rotation_angles / valid_dt

    if len(rotational_velocity) >= 2:
        rotational_acceleration_dt = (valid_dt[1:] + valid_dt[:-1]) / 2
        rotational_acceleration = (
            np.diff(rotational_velocity) / rotational_acceleration_dt
        )
    else:
        rotational_acceleration = np.array([])

    features[f"{prefix}_rotation_path_length"] = (
        np.nansum(valid_rotation_angles) if len(valid_rotation_angles) else np.nan
    )
    features[f"{prefix}_mean_rotational_velocity"] = (
        np.nanmean(rotational_velocity) if len(rotational_velocity) else np.nan
    )
    features[f"{prefix}_max_rotational_velocity"] = (
        np.nanmax(rotational_velocity) if len(rotational_velocity) else np.nan
    )
    features[f"{prefix}_mean_rotational_acceleration"] = (
        np.nanmean(np.abs(rotational_acceleration))
        if len(rotational_acceleration)
        else np.nan
    )
    features[f"{prefix}_max_rotational_acceleration"] = (
        np.nanmax(np.abs(rotational_acceleration))
        if len(rotational_acceleration)
        else np.nan
    )

    return features


def get_usm_speed_series(usm_df: pd.DataFrame):
    """
    Return timestamp-level speed series for one USM.
    Speed is assigned to the second timestamp of each movement step.
    """

    usm_df = usm_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    if len(usm_df) < 2:
        return pd.DataFrame(columns=["timestamp", "speed"])

    t = usm_df["timestamp"].to_numpy(dtype=float) / 1000.0
    pos = usm_df[["x", "y", "z"]].to_numpy(dtype=float)

    dt = np.diff(t)
    diffs = np.diff(pos, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)

    valid = dt > 0

    timestamps = usm_df["timestamp"].iloc[1:].to_numpy()[valid]
    speed = step_distances[valid] / dt[valid]

    return pd.DataFrame({
        "timestamp": timestamps,
        "speed": speed,
    })


def compute_pair_coordination_features(traj: pd.DataFrame, usm_a: int, usm_b: int):
    """
    Calculate coordination features between two USMs.

    Features:
    - mean distance
    - min distance
    - max distance
    - std distance
    - mean absolute relative speed
    - speed correlation
    """

    prefix = f"USM{usm_a}_USM{usm_b}"
    features = {}

    a = traj[traj["usm"] == usm_a][["timestamp", "x", "y", "z"]].drop_duplicates("timestamp")
    b = traj[traj["usm"] == usm_b][["timestamp", "x", "y", "z"]].drop_duplicates("timestamp")

    merged = a.merge(
        b,
        on="timestamp",
        suffixes=("_a", "_b"),
        how="inner",
    )

    features[f"{prefix}_n_aligned_samples"] = len(merged)

    if len(merged) < 2:
        features[f"{prefix}_mean_distance"] = np.nan
        features[f"{prefix}_min_distance"] = np.nan
        features[f"{prefix}_max_distance"] = np.nan
        features[f"{prefix}_std_distance"] = np.nan
        features[f"{prefix}_mean_abs_relative_speed"] = np.nan
        features[f"{prefix}_speed_correlation"] = np.nan
        return features

    pos_a = merged[["x_a", "y_a", "z_a"]].to_numpy(dtype=float)
    pos_b = merged[["x_b", "y_b", "z_b"]].to_numpy(dtype=float)

    distances = np.linalg.norm(pos_a - pos_b, axis=1)

    features[f"{prefix}_mean_distance"] = np.nanmean(distances)
    features[f"{prefix}_min_distance"] = np.nanmin(distances)
    features[f"{prefix}_max_distance"] = np.nanmax(distances)
    features[f"{prefix}_std_distance"] = np.nanstd(distances)

    speed_a = get_usm_speed_series(traj[traj["usm"] == usm_a])
    speed_b = get_usm_speed_series(traj[traj["usm"] == usm_b])

    speed_merged = speed_a.merge(
        speed_b,
        on="timestamp",
        suffixes=("_a", "_b"),
        how="inner",
    )

    if len(speed_merged) < 2:
        features[f"{prefix}_mean_abs_relative_speed"] = np.nan
        features[f"{prefix}_speed_correlation"] = np.nan
    else:
        sa = speed_merged["speed_a"].to_numpy(dtype=float)
        sb = speed_merged["speed_b"].to_numpy(dtype=float)

        features[f"{prefix}_mean_abs_relative_speed"] = np.nanmean(np.abs(sa - sb))

        if np.nanstd(sa) == 0 or np.nanstd(sb) == 0:
            features[f"{prefix}_speed_correlation"] = np.nan
        else:
            features[f"{prefix}_speed_correlation"] = np.corrcoef(sa, sb)[0, 1]

    return features


def extract_trial_features_v2(file_path: Path):
    """
    Extract interpretable kinematic features from one DVST_XI file.
    """

    pose_df = load_pose_table(file_path)
    traj = build_base_frame_trajectories(pose_df)

    features = {}

    features["n_rows"] = len(traj)

    if traj.empty:
        features["n_unique_timestamps"] = 0
        features["duration_seconds"] = np.nan
        features["n_usms"] = 0
        features["all_usms_path_length_total"] = 0.0
        features["all_usms_fully_idle"] = 1
        features["active_tool_count"] = 0
        return features

    unique_timestamps = traj["timestamp"].drop_duplicates().sort_values()

    features["n_unique_timestamps"] = len(unique_timestamps)

    if len(unique_timestamps) >= 2:
        features["duration_seconds"] = (
            unique_timestamps.iloc[-1] - unique_timestamps.iloc[0]
        ) / 1000.0
    else:
        features["duration_seconds"] = np.nan

    features["n_usms"] = traj["usm"].nunique()

    # Per-USM features
    for usm in sorted(traj["usm"].unique()):
        usm_df = traj[traj["usm"] == usm]
        prefix = f"USM{usm}"

        usm_features = compute_usm_position_features(usm_df, prefix)
        features.update(usm_features)

    # Data-quality and active-tool features
    path_lengths = []

    for usm in [0, 1, 2, 3]:
        path_lengths.append(features.get(f"USM{usm}_path_length", np.nan))

    total_path_length = np.nansum(path_lengths)

    features["all_usms_path_length_total"] = total_path_length
    features["all_usms_fully_idle"] = int(total_path_length < 1e-9)

    active_tool_count = 0

    for usm in [1, 2, 3]:
        idle_fraction = features.get(f"USM{usm}_idle_fraction", np.nan)

        if pd.notna(idle_fraction) and idle_fraction < 0.95:
            active = 1
        else:
            active = 0

        features[f"USM{usm}_active"] = active
        active_tool_count += active

    features["active_tool_count"] = active_tool_count

    # Pairwise coordination features between instrument arms
    for usm_a, usm_b in [(1, 2), (1, 3), (2, 3)]:
        pair_features = compute_pair_coordination_features(traj, usm_a, usm_b)
        features.update(pair_features)

    return features


# =========================
# Label extraction
# =========================

def build_y_from_label_folder(label_dir: Path):
    csv_files = sorted(label_dir.rglob("*.csv"))

    rows = []

    print(f"\nFound {len(csv_files)} label files.")

    for i, file_path in enumerate(csv_files, start=1):
        print(f"[y {i}/{len(csv_files)}] {file_path.name}")

        try:
            df = pd.read_csv(file_path)

            if TARGET_COLUMN not in df.columns:
                print(f"Skipping {file_path.name}: no target column")
                continue

            rows.append(df)

        except Exception as e:
            print(f"Error reading label file {file_path.name}: {e}")

    if len(rows) == 0:
        raise ValueError("No valid label files found.")

    labels = pd.concat(rows, ignore_index=True)

    required_cols = [
        "User",
        "Participant ID",
        "Task",
        "Attempt",
        TARGET_COLUMN,
    ]

    missing = [c for c in required_cols if c not in labels.columns]

    if missing:
        raise ValueError(f"Missing columns in labels: {missing}")

    y_df = pd.DataFrame()

    y_df["role"] = labels["User"].astype(str).str.strip().str.lower()
    y_df["participant_id"] = labels["Participant ID"].apply(clean_id)
    y_df["task_clean"] = labels["Task"].apply(clean_task)
    y_df["trial"] = labels["Attempt"].apply(clean_id)
    y_df["target_score"] = pd.to_numeric(labels[TARGET_COLUMN], errors="coerce")

    if "Percentage score" in labels.columns:
        y_df["percentage_score"] = pd.to_numeric(labels["Percentage score"], errors="coerce")

    y_df = y_df.dropna(subset=["target_score"])

    y_df = y_df.drop_duplicates(
        subset=["role", "participant_id", "task_clean", "trial"],
        keep="first",
    )

    print("\ny shape:", y_df.shape)

    return y_df


# =========================
# Build X
# =========================

def build_X_from_dvst_folder(dvst_dir: Path):
    csv_files = sorted(dvst_dir.rglob("*.csv"))

    rows = []

    print(f"Found {len(csv_files)} DVST files.")

    for i, file_path in enumerate(csv_files, start=1):
        print(f"[X {i}/{len(csv_files)}] {file_path.name}")

        try:
            metadata = extract_metadata_from_dvst_filename(file_path)
            features = extract_trial_features_v2(file_path)

            row = {}
            row.update(metadata)
            row.update(features)

            rows.append(row)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    X_df = pd.DataFrame(rows)

    print("\nX shape:", X_df.shape)

    return X_df


# =========================
# Build final dataset
# =========================

def build_dataset_v2():
    X_df = build_X_from_dvst_folder(DVST_DIR)
    y_df = build_y_from_label_folder(LABEL_DIR)

    merge_keys = ["role", "participant_id", "task_clean", "trial"]

    print("\nX key examples:")
    print(X_df[merge_keys].head())

    print("\ny key examples:")
    print(y_df[merge_keys].head())

    debug_merge = X_df.merge(
        y_df,
        on=merge_keys,
        how="outer",
        indicator=True,
    )

    print("\nMerge check:")
    print(debug_merge["_merge"].value_counts())

    print("\nDVST files without labels:")
    print(
        debug_merge[debug_merge["_merge"] == "left_only"]
        [merge_keys + ["file_name"]]
        .head(20)
    )

    print("\nLabels without DVST files:")
    print(
        debug_merge[debug_merge["_merge"] == "right_only"]
        [merge_keys]
        .head(20)
    )

    dataset = X_df.merge(
        y_df,
        on=merge_keys,
        how="inner",
    )

    print("\nMerged dataset shape:", dataset.shape)

    if dataset.empty:
        raise ValueError("Merged dataset is empty. Check filename parsing and label keys.")

    print("\nTask counts:")
    print(dataset["task_clean"].value_counts())

    print("\nTarget summary:")
    print(dataset["target_score"].describe())

    if "all_usms_fully_idle" in dataset.columns:
        print("\nFully idle trials:")
        print(dataset["all_usms_fully_idle"].value_counts())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")

    return dataset


if __name__ == "__main__":
    build_dataset_v2()