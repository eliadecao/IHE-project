from __future__ import annotations

"""
Build an FK-only operation-level dataset and merge it with M-GEARS labels.

This version extends the existing FK-only pipeline with camera-related
kinematic metrics extracted directly from the supervisor FK outputs.

New camera metrics include:
    - camera path length
    - camera mean / max speed
    - camera movement episode count
    - camera total moving duration / fraction
    - camera rotation path
    - camera angular speed
    - camera-to-tool distances for USM0 / USM2 / USM3
    - camera-to-primary-tools mean distance (USM0 + USM2)

The camera is explicitly identified from FK output as:
    Role == "ECM_Camera"
or:
    USM == 1

This script does NOT read original raw DVST files.
"""

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


# ============================================================
# FK FORMAT
# ============================================================

FK_REQUIRED_COLUMNS = {
    "TimeStamp",
    "USM",
    "Role",
    "True_Tip_X",
    "True_Tip_Y",
    "True_Tip_Z",
    "True_Tip_Matrix",
}


# ============================================================
# CAMERA CONFIG
# ============================================================

CAMERA_USM = 1

# Primary working PSMs used by the supervisor FK optimisation.
PRIMARY_TOOL_USMS = (0, 2)

# All PSMs potentially available in the FK output.
PSM_USMS = (0, 2, 3)

# Operational definition of a camera movement episode:
# camera translational speed >= 5 mm/s
CAMERA_SPEED_THRESHOLD_M_S = 0.005

# Movement must persist for at least 0.20 s.
CAMERA_MIN_MOVE_DURATION_S = 0.20


# ============================================================
# MODULE LOADING
# ============================================================

def load_v3_module(script_path: Path):
    """
    Dynamically import dataset_v3.py,
    including Python 3.14 dataclass support.
    """

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

    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


# ============================================================
# FILE / MATRIX HELPERS
# ============================================================

def normalise_fk_stem(path: Path) -> str:
    """
    operation_FK.csv -> operation
    """

    stem = path.stem

    return (
        stem[:-3]
        if stem.endswith("_FK")
        else stem
    )


def metadata_path_from_fk(
    fk_path: Path,
) -> Path:
    """
    Create a synthetic non-FK filename for the existing
    metadata parser.
    """

    return fk_path.with_name(
        normalise_fk_stem(fk_path)
        + ".csv"
    )


def parse_matrix(
    value: object,
) -> np.ndarray | None:
    """
    Parse a flattened 4x4 transformation matrix.
    """

    if pd.isna(value):
        return None

    arr = np.fromstring(
        str(value).replace(",", " "),
        sep=" ",
        dtype=float,
    )

    if (
        arr.size != 16
        or not np.all(
            np.isfinite(arr)
        )
    ):
        return None

    return arr.reshape(
        4,
        4,
    )


# ============================================================
# FK LOADING / QC
# ============================================================

def load_fk_telemetry(
    fk_path: Path,
    v3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Load, clean, time-normalise and QC one FK file.
    """

    fk = pd.read_csv(
        fk_path
    )

    missing = (
        FK_REQUIRED_COLUMNS
        .difference(
            fk.columns
        )
    )

    if missing:
        raise ValueError(
            f"{fk_path.name} is missing FK columns: "
            f"{sorted(missing)}"
        )

    fk = fk.copy()

    rows_original = len(
        fk
    )

    fk[
        "timestamp_raw"
    ] = pd.to_numeric(
        fk["TimeStamp"],
        errors="coerce",
    )

    fk[
        "usm"
    ] = pd.to_numeric(
        fk["USM"],
        errors="coerce",
    )

    fk[
        "true_x"
    ] = pd.to_numeric(
        fk["True_Tip_X"],
        errors="coerce",
    )

    fk[
        "true_y"
    ] = pd.to_numeric(
        fk["True_Tip_Y"],
        errors="coerce",
    )

    fk[
        "true_z"
    ] = pd.to_numeric(
        fk["True_Tip_Z"],
        errors="coerce",
    )

    fk = fk.dropna(
        subset=[
            "timestamp_raw",
            "usm",
        ]
    )

    removed_missing_key = (
        rows_original
        - len(fk)
    )

    fk[
        "usm"
    ] = fk[
        "usm"
    ].astype(int)

    fk[
        "Role"
    ] = (
        fk["Role"]
        .astype(str)
        .str.strip()
    )

    fk = fk.sort_values(
        [
            "timestamp_raw",
            "usm",
        ],
        kind="stable",
    )

    duplicate_count = int(
        fk.duplicated(
            [
                "timestamp_raw",
                "usm",
            ]
        ).sum()
    )

    fk = fk.drop_duplicates(
        [
            "timestamp_raw",
            "usm",
        ],
        keep="first",
    )

    if fk.empty:
        raise ValueError(
            f"{fk_path.name} contains no valid "
            f"timestamp/USM rows"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # timestamp converted to actual seconds here.
    # --------------------------------------------------------

    scale, unit = (
        v3.infer_timestamp_scale(
            fk[
                "timestamp_raw"
            ].to_numpy()
        )
    )

    origin = float(
        fk[
            "timestamp_raw"
        ].min()
    )

    fk[
        "time_seconds"
    ] = (
        fk["timestamp_raw"]
        - origin
    ) * scale

    fk[
        "_matrix"
    ] = [
        parse_matrix(v)
        for v
        in fk[
            "True_Tip_Matrix"
        ].tolist()
    ]

    valid_xyz = np.isfinite(
        fk[
            [
                "true_x",
                "true_y",
                "true_z",
            ]
        ]
    ).all(
        axis=1
    )

    duration = (
        float(
            fk[
                "time_seconds"
            ].max()
            -
            fk[
                "time_seconds"
            ].min()
        )
        if len(fk) >= 2
        else np.nan
    )

    qc = {

        "qc_fk_rows_original":
            int(
                rows_original
            ),

        "qc_fk_rows":
            int(
                len(fk)
            ),

        "qc_fk_rows_removed_missing_timestamp_or_usm":
            int(
                removed_missing_key
            ),

        "qc_fk_duplicate_timestamp_usm_rows_removed":
            duplicate_count,

        "qc_fk_timestamp_unit_inferred":
            unit,

        "qc_fk_timestamp_scale":
            float(scale),

        "qc_fk_duration_seconds":
            duration,

        "qc_fk_valid_xyz_fraction":
            float(
                valid_xyz.mean()
            ),

        "qc_fk_usms_present":
            ",".join(
                map(
                    str,
                    sorted(
                        fk[
                            "usm"
                        ].unique()
                    ),
                )
            ),

        "qc_fk_roles_present":
            ",".join(
                sorted(
                    fk[
                        "Role"
                    ]
                    .dropna()
                    .unique()
                )
            ),
    }

    return fk, qc


# ============================================================
# GENERIC PSM POSE FEATURES
# ============================================================

def compute_fk_pose_features(
    frame: pd.DataFrame,
    usm: int,
    v3,
) -> tuple[
    dict[str, float],
    dict[str, np.ndarray],
]:
    """
    Compute global Cartesian and rotational motion
    features for one PSM.
    """

    prefix = (
        f"USM{usm}_pose_global"
    )

    valid = np.isfinite(
        frame[
            [
                "true_x",
                "true_y",
                "true_z",
            ]
        ]
    ).all(
        axis=1
    )

    frame = (
        frame.loc[
            valid
        ]
        .sort_values(
            "time_seconds",
            kind="stable",
        )
        .copy()
    )

    if frame.empty:

        return {
            f"{prefix}_n_samples": 0
        }, {}

    t = frame[
        "time_seconds"
    ].to_numpy(
        dtype=float
    )

    xyz = frame[
        [
            "true_x",
            "true_y",
            "true_z",
        ]
    ].to_numpy(
        dtype=float
    )

    features, arrays = (
        v3.compute_motion_summary(
            t,
            xyz,
            prefix,
            calculate_dimensionless_jerk=True,
        )
    )

    if not arrays:
        return (
            features,
            arrays,
        )

    smooth_xyz = arrays[
        "values"
    ]

    (
        x_range,
        y_range,
        z_range,
    ) = map(
        float,
        np.ptp(
            smooth_xyz,
            axis=0,
        ),
    )

    path = features.get(
        f"{prefix}_path_l2",
        np.nan,
    )

    features.update({

        f"{prefix}_x_range_m":
            x_range,

        f"{prefix}_y_range_m":
            y_range,

        f"{prefix}_z_range_m":
            z_range,

        f"{prefix}_workspace_box_volume_m3":
            (
                x_range
                * y_range
                * z_range
            ),

        f"{prefix}_economy_area_xy":
            v3.safe_divide(
                math.sqrt(
                    x_range
                    * y_range
                ),
                path,
            ),

        f"{prefix}_economy_area_xz":
            v3.safe_divide(
                math.sqrt(
                    x_range
                    * z_range
                ),
                path,
            ),

        f"{prefix}_economy_area_yz":
            v3.safe_divide(
                math.sqrt(
                    y_range
                    * z_range
                ),
                path,
            ),

        f"{prefix}_economy_volume":
            v3.safe_divide(
                np.cbrt(
                    x_range
                    * y_range
                    * z_range
                ),
                path,
            ),
    })

    # --------------------------------------------------------
    # Rotation / orientation QC and dynamics
    # --------------------------------------------------------

    matrix_rows = frame[
        "_matrix"
    ].tolist()

    matrix_valid = np.array(
        [
            m is not None
            for m
            in matrix_rows
        ],
        dtype=bool,
    )

    features[
        f"{prefix}_matrix_valid_fraction"
    ] = float(
        matrix_valid.mean()
    )

    valid_t = frame.loc[
        matrix_valid,
        "time_seconds",
    ].to_numpy(
        dtype=float
    )

    matrices = [
        m
        for m
        in matrix_rows
        if m is not None
    ]

    if len(matrices) < 2:
        return (
            features,
            arrays,
        )

    raw_rot = np.stack(
        [
            m[
                :3,
                :3,
            ]
            for m
            in matrices
        ]
    )

    orth = np.array([
        np.linalg.norm(
            r.T @ r
            - np.eye(3),
            ord="fro",
        )
        for r
        in raw_rot
    ])

    det = np.linalg.det(
        raw_rot
    )

    rotations = np.stack([
        v3.project_to_rotation_matrix(
            r
        )
        for r
        in raw_rot
    ])

    angles = np.array([
        v3.rotation_angle_between(
            rotations[i],
            rotations[i + 1],
        )
        for i
        in range(
            len(rotations)
            - 1
        )
    ])

    dt = np.diff(
        valid_t
    )

    keep = (
        dt > 0
    )

    angles = angles[
        keep
    ]

    dt = dt[
        keep
    ]

    angular_speed = (
        angles / dt
        if len(dt)
        else np.array([])
    )

    if len(
        angular_speed
    ) >= 2:

        mid_t = (
            valid_t[:-1][keep]
            +
            valid_t[1:][keep]
        ) / 2

        angular_acc = (
            np.gradient(
                angular_speed,
                mid_t,
                edge_order=(
                    2
                    if len(
                        angular_speed
                    ) >= 3
                    else 1
                ),
            )
        )

    else:

        angular_acc = (
            np.array([])
        )

    features.update({

        f"{prefix}_rotation_orthogonality_error_mean":
            float(
                np.mean(orth)
            ),

        f"{prefix}_rotation_orthogonality_error_max":
            float(
                np.max(orth)
            ),

        f"{prefix}_rotation_determinant_mean":
            float(
                np.mean(det)
            ),

        f"{prefix}_rotation_determinant_std":
            float(
                np.std(det)
            ),

        f"{prefix}_rotation_path_radians":
            (
                float(
                    np.sum(
                        angles
                    )
                )
                if len(angles)
                else np.nan
            ),

        f"{prefix}_mean_angular_speed_rad_s":
            (
                float(
                    np.mean(
                        angular_speed
                    )
                )
                if len(
                    angular_speed
                )
                else np.nan
            ),

        f"{prefix}_max_angular_speed_rad_s":
            (
                float(
                    np.max(
                        angular_speed
                    )
                )
                if len(
                    angular_speed
                )
                else np.nan
            ),

        f"{prefix}_mean_abs_angular_acceleration_rad_s2":
            (
                float(
                    np.mean(
                        np.abs(
                            angular_acc
                        )
                    )
                )
                if len(
                    angular_acc
                )
                else np.nan
            ),

        f"{prefix}_max_abs_angular_acceleration_rad_s2":
            (
                float(
                    np.max(
                        np.abs(
                            angular_acc
                        )
                    )
                )
                if len(
                    angular_acc
                )
                else np.nan
            ),
    })

    return (
        features,
        arrays,
    )


# ============================================================
# CAMERA MOTION EPISODES
# ============================================================

def compute_camera_move_episodes(
    t: np.ndarray,
    xyz: np.ndarray,
    speed_threshold: float =
        CAMERA_SPEED_THRESHOLD_M_S,
    min_duration: float =
        CAMERA_MIN_MOVE_DURATION_S,
) -> dict[str, float]:
    """
    Count distinct camera motion episodes.

    An episode starts when translational speed crosses
    above `speed_threshold` and ends when it drops below it.

    Very short episodes (< min_duration) are ignored.
    """

    result = {

        "camera_move_count":
            np.nan,

        "camera_moving_duration_s":
            np.nan,

        "camera_moving_fraction":
            np.nan,
    }

    if (
        len(t) < 2
        or len(xyz) < 2
    ):
        return result

    dt = np.diff(
        t
    )

    step = np.linalg.norm(
        np.diff(
            xyz,
            axis=0,
        ),
        axis=1,
    )

    valid = (
        np.isfinite(dt)
        &
        np.isfinite(step)
        &
        (dt > 0)
    )

    if valid.sum() == 0:
        return result

    interval_t0 = t[:-1][
        valid
    ]

    interval_t1 = t[1:][
        valid
    ]

    interval_dt = dt[
        valid
    ]

    speed = (
        step[
            valid
        ]
        /
        interval_dt
    )

    moving = (
        speed
        >= speed_threshold
    )

    episodes: list[
        tuple[
            float,
            float,
        ]
    ] = []

    episode_start = None
    episode_end = None

    for i, is_moving in enumerate(
        moving
    ):

        if is_moving:

            if episode_start is None:
                episode_start = float(
                    interval_t0[i]
                )

            episode_end = float(
                interval_t1[i]
            )

        else:

            if (
                episode_start is not None
                and episode_end is not None
            ):

                duration = (
                    episode_end
                    - episode_start
                )

                if (
                    duration
                    >= min_duration
                ):
                    episodes.append(
                        (
                            episode_start,
                            episode_end,
                        )
                    )

                episode_start = None
                episode_end = None

    # close final episode
    if (
        episode_start is not None
        and episode_end is not None
    ):

        duration = (
            episode_end
            - episode_start
        )

        if (
            duration
            >= min_duration
        ):
            episodes.append(
                (
                    episode_start,
                    episode_end,
                )
            )

    moving_duration = float(
        np.sum([
            end - start
            for start, end
            in episodes
        ])
    )

    total_duration = float(
        np.max(t)
        - np.min(t)
    )

    result[
        "camera_move_count"
    ] = int(
        len(episodes)
    )

    result[
        "camera_moving_duration_s"
    ] = moving_duration

    result[
        "camera_moving_fraction"
    ] = (
        moving_duration
        / total_duration
        if total_duration > 0
        else np.nan
    )

    return result


# ============================================================
# CAMERA FEATURES
# ============================================================

def compute_camera_features(
    frame: pd.DataFrame,
    v3,
) -> tuple[
    dict[str, float],
    dict[str, np.ndarray],
]:
    """
    Compute operation-level camera kinematic features
    from ECM / USM1 FK trajectory.
    """

    prefix = (
        "camera_pose_global"
    )

    valid = np.isfinite(
        frame[
            [
                "true_x",
                "true_y",
                "true_z",
            ]
        ]
    ).all(
        axis=1
    )

    frame = (
        frame.loc[
            valid
        ]
        .sort_values(
            "time_seconds",
            kind="stable",
        )
        .copy()
    )

    if frame.empty:

        return {

            "camera_n_samples":
                0,

            "camera_path_length_m":
                np.nan,

            "camera_mean_speed_m_s":
                np.nan,

            "camera_max_speed_m_s":
                np.nan,

            "camera_move_count":
                np.nan,

            "camera_moving_duration_s":
                np.nan,

            "camera_moving_fraction":
                np.nan,

        }, {}

    t = frame[
        "time_seconds"
    ].to_numpy(
        dtype=float
    )

    xyz = frame[
        [
            "true_x",
            "true_y",
            "true_z",
        ]
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Reuse the same validated motion-summary logic
    # as the PSM features.
    # --------------------------------------------------------

    summary, arrays = (
        v3.compute_motion_summary(
            t,
            xyz,
            prefix,
            calculate_dimensionless_jerk=True,
        )
    )

    features: dict[
        str,
        float,
    ] = {

        "camera_n_samples":
            int(
                len(frame)
            ),
    }

    # Give the clinically interpretable aliases
    # requested by Laurent.
    features[
        "camera_path_length_m"
    ] = summary.get(
        f"{prefix}_path_l2",
        np.nan,
    )

    features[
        "camera_mean_speed_m_s"
    ] = summary.get(
        f"{prefix}_mean_speed_l2",
        np.nan,
    )

    features[
        "camera_rms_speed_m_s"
    ] = summary.get(
        f"{prefix}_rms_speed_l2",
        np.nan,
    )

    features[
        "camera_max_speed_m_s"
    ] = summary.get(
        f"{prefix}_max_speed_l2",
        np.nan,
    )

    features[
        "camera_pause_count"
    ] = summary.get(
        f"{prefix}_pause_count",
        np.nan,
    )

    features[
        "camera_pause_duration_seconds"
    ] = summary.get(
        f"{prefix}_pause_duration_seconds",
        np.nan,
    )

    features[
        "camera_pause_fraction"
    ] = summary.get(
        f"{prefix}_pause_fraction",
        np.nan,
    )

    features[
        "camera_log_dimensionless_jerk"
    ] = summary.get(
        f"{prefix}_log_dimensionless_jerk",
        np.nan,
    )

    # --------------------------------------------------------
    # Camera movement episodes
    # --------------------------------------------------------

    features.update(
        compute_camera_move_episodes(
            t,
            xyz,
        )
    )

    # --------------------------------------------------------
    # Camera rotation features
    # --------------------------------------------------------

    matrix_rows = frame[
        "_matrix"
    ].tolist()

    matrix_valid = np.array(
        [
            m is not None
            for m
            in matrix_rows
        ],
        dtype=bool,
    )

    features[
        "camera_matrix_valid_fraction"
    ] = float(
        matrix_valid.mean()
    )

    matrices = [
        m
        for m
        in matrix_rows
        if m is not None
    ]

    valid_t = frame.loc[
        matrix_valid,
        "time_seconds",
    ].to_numpy(
        dtype=float
    )

    if len(
        matrices
    ) >= 2:

        rotations = np.stack([
            v3.project_to_rotation_matrix(
                m[
                    :3,
                    :3,
                ]
            )
            for m
            in matrices
        ])

        angles = np.array([
            v3.rotation_angle_between(
                rotations[i],
                rotations[i + 1],
            )
            for i
            in range(
                len(rotations)
                - 1
            )
        ])

        dt = np.diff(
            valid_t
        )

        keep = (
            dt > 0
        )

        angles = angles[
            keep
        ]

        dt = dt[
            keep
        ]

        angular_speed = (
            angles / dt
            if len(dt)
            else np.array([])
        )

        features[
            "camera_rotation_path_radians"
        ] = (
            float(
                np.sum(
                    angles
                )
            )
            if len(
                angles
            )
            else np.nan
        )

        features[
            "camera_mean_angular_speed_rad_s"
        ] = (
            float(
                np.mean(
                    angular_speed
                )
            )
            if len(
                angular_speed
            )
            else np.nan
        )

        features[
            "camera_max_angular_speed_rad_s"
        ] = (
            float(
                np.max(
                    angular_speed
                )
            )
            if len(
                angular_speed
            )
            else np.nan
        )

    else:

        features[
            "camera_rotation_path_radians"
        ] = np.nan

        features[
            "camera_mean_angular_speed_rad_s"
        ] = np.nan

        features[
            "camera_max_angular_speed_rad_s"
        ] = np.nan

    return (
        features,
        arrays,
    )


# ============================================================
# CAMERA ↔ TOOL DISTANCE
# ============================================================

def compute_camera_tool_distance_features(
    fk: pd.DataFrame,
    camera_usm: int,
    tool_usm: int,
) -> dict[str, float]:
    """
    Compute synchronized Euclidean distance between
    the camera optical tip and one PSM tool tip.
    """

    prefix = (
        f"camera_to_USM{tool_usm}"
    )

    camera = (
        fk.loc[
            fk[
                "usm"
            ].eq(
                camera_usm
            ),
            [
                "time_seconds",
                "true_x",
                "true_y",
                "true_z",
            ],
        ]
        .dropna()
        .copy()
    )

    tool = (
        fk.loc[
            fk[
                "usm"
            ].eq(
                tool_usm
            ),
            [
                "time_seconds",
                "true_x",
                "true_y",
                "true_z",
            ],
        ]
        .dropna()
        .copy()
    )

    if (
        camera.empty
        or tool.empty
    ):

        return {

            f"{prefix}_n_aligned_samples":
                0,

            f"{prefix}_mean_distance_m":
                np.nan,

            f"{prefix}_std_distance_m":
                np.nan,

            f"{prefix}_min_distance_m":
                np.nan,

            f"{prefix}_max_distance_m":
                np.nan,

        }

    camera = camera.rename(
        columns={
            "true_x": "cam_x",
            "true_y": "cam_y",
            "true_z": "cam_z",
        }
    )

    tool = tool.rename(
        columns={
            "true_x": "tool_x",
            "true_y": "tool_y",
            "true_z": "tool_z",
        }
    )

    # FK output is synchronized by timestamp,
    # so exact time matching is appropriate here.
    aligned = camera.merge(
        tool,
        on="time_seconds",
        how="inner",
        validate="one_to_one",
    )

    if aligned.empty:

        return {

            f"{prefix}_n_aligned_samples":
                0,

            f"{prefix}_mean_distance_m":
                np.nan,

            f"{prefix}_std_distance_m":
                np.nan,

            f"{prefix}_min_distance_m":
                np.nan,

            f"{prefix}_max_distance_m":
                np.nan,

        }

    distance = np.sqrt(

        (
            aligned["cam_x"]
            -
            aligned["tool_x"]
        ) ** 2

        +

        (
            aligned["cam_y"]
            -
            aligned["tool_y"]
        ) ** 2

        +

        (
            aligned["cam_z"]
            -
            aligned["tool_z"]
        ) ** 2
    )

    distance = distance[
        np.isfinite(
            distance
        )
    ]

    if len(
        distance
    ) == 0:

        return {

            f"{prefix}_n_aligned_samples":
                0,

            f"{prefix}_mean_distance_m":
                np.nan,

            f"{prefix}_std_distance_m":
                np.nan,

            f"{prefix}_min_distance_m":
                np.nan,

            f"{prefix}_max_distance_m":
                np.nan,

        }

    return {

        f"{prefix}_n_aligned_samples":
            int(
                len(
                    distance
                )
            ),

        f"{prefix}_mean_distance_m":
            float(
                np.mean(
                    distance
                )
            ),

        f"{prefix}_std_distance_m":
            float(
                np.std(
                    distance
                )
            ),

        f"{prefix}_min_distance_m":
            float(
                np.min(
                    distance
                )
            ),

        f"{prefix}_max_distance_m":
            float(
                np.max(
                    distance
                )
            ),

    }


# ============================================================
# OPERATION-LEVEL FEATURE EXTRACTION
# ============================================================

def extract_trial_features_fk_only(
    fk_path: Path,
    v3,
) -> dict[str, object]:
    """
    Extract one operation-level FK-only feature row.
    """

    metadata = (
        v3.extract_metadata_from_dvst_filename(
            metadata_path_from_fk(
                fk_path
            )
        )
    )

    fk, fk_qc = (
        load_fk_telemetry(
            fk_path,
            v3,
        )
    )

    features: dict[
        str,
        object,
    ] = {}

    features.update(
        metadata
    )

    features[
        "fk_file_name"
    ] = fk_path.name

    features.update(
        fk_qc
    )

    # --------------------------------------------------------
    # Identify PSM / ECM arms
    # --------------------------------------------------------

    instrument_usms = sorted(

        fk.loc[
            fk[
                "Role"
            ].eq(
                "PSM_Instrument"
            ),
            "usm",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    camera_usms = sorted(

        fk.loc[
            fk[
                "Role"
            ].eq(
                "ECM_Camera"
            ),
            "usm",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    features[
        "fk_instrument_usms"
    ] = ",".join(
        map(
            str,
            instrument_usms,
        )
    )

    features[
        "fk_camera_usms"
    ] = ",".join(
        map(
            str,
            camera_usms,
        )
    )

    features[
        "fk_instrument_usm_count"
    ] = len(
        instrument_usms
    )

    features[
        "fk_camera_usm_count"
    ] = len(
        camera_usms
    )

    # ========================================================
    # EXISTING PSM FEATURES
    # ========================================================

    trajectories: dict[
        int,
        dict[
            str,
            np.ndarray,
        ],
    ] = {}

    pose_paths: dict[
        int,
        float,
    ] = {}

    for usm in instrument_usms:

        pose_features, arrays = (
            compute_fk_pose_features(

                fk.loc[
                    fk[
                        "usm"
                    ].eq(
                        usm
                    )
                ],

                usm,

                v3,
            )
        )

        features.update(
            pose_features
        )

        if arrays:

            trajectories[
                usm
            ] = arrays

            path = (
                pose_features.get(
                    f"USM{usm}_pose_global_path_l2",
                    np.nan,
                )
            )

            pose_paths[
                usm
            ] = (
                float(path)
                if np.isfinite(
                    path
                )
                else np.nan
            )

    features[
        "all_psm_pose_global_path_total_m"
    ] = (

        float(
            np.nansum(
                list(
                    pose_paths.values()
                )
            )
        )

        if pose_paths

        else np.nan
    )

    # ========================================================
    # CAMERA FEATURES
    # ========================================================

    camera_arrays = {}

    if (
        CAMERA_USM
        in fk[
            "usm"
        ].unique()
    ):

        camera_features, camera_arrays = (
            compute_camera_features(

                fk.loc[
                    fk[
                        "usm"
                    ].eq(
                        CAMERA_USM
                    )
                ],

                v3,
            )
        )

        features.update(
            camera_features
        )

    else:

        features.update({

            "camera_n_samples":
                0,

            "camera_path_length_m":
                np.nan,

            "camera_mean_speed_m_s":
                np.nan,

            "camera_rms_speed_m_s":
                np.nan,

            "camera_max_speed_m_s":
                np.nan,

            "camera_move_count":
                np.nan,

            "camera_moving_duration_s":
                np.nan,

            "camera_moving_fraction":
                np.nan,

            "camera_pause_count":
                np.nan,

            "camera_pause_duration_seconds":
                np.nan,

            "camera_pause_fraction":
                np.nan,

            "camera_log_dimensionless_jerk":
                np.nan,

            "camera_rotation_path_radians":
                np.nan,

            "camera_mean_angular_speed_rad_s":
                np.nan,

            "camera_max_angular_speed_rad_s":
                np.nan,
        })

    # ========================================================
    # CAMERA ↔ TOOL DISTANCES
    # ========================================================

    for tool_usm in PSM_USMS:

        distance_features = (
            compute_camera_tool_distance_features(

                fk=fk,

                camera_usm=
                    CAMERA_USM,

                tool_usm=
                    tool_usm,
            )
        )

        features.update(
            distance_features
        )

    # --------------------------------------------------------
    # Primary-tool mean camera distance:
    # USM0 + USM2, matching supervisor FK target_psm_indices.
    # --------------------------------------------------------

    primary_distance_values = []

    for tool_usm in PRIMARY_TOOL_USMS:

        value = features.get(

            f"camera_to_USM{tool_usm}_mean_distance_m",

            np.nan,
        )

        if np.isfinite(
            value
        ):
            primary_distance_values.append(
                float(
                    value
                )
            )

    features[
        "camera_to_primary_tools_mean_distance_m"
    ] = (

        float(
            np.mean(
                primary_distance_values
            )
        )

        if primary_distance_values

        else np.nan
    )

    # Also keep all available tools as a secondary metric.
    all_tool_distance_values = []

    for tool_usm in PSM_USMS:

        value = features.get(

            f"camera_to_USM{tool_usm}_mean_distance_m",

            np.nan,
        )

        if np.isfinite(
            value
        ):
            all_tool_distance_values.append(
                float(
                    value
                )
            )

    features[
        "camera_to_all_tools_mean_distance_m"
    ] = (

        float(
            np.mean(
                all_tool_distance_values
            )
        )

        if all_tool_distance_values

        else np.nan
    )

    # ========================================================
    # EXISTING ACTIVE PSM LOGIC
    # ========================================================

    finite_paths = {

        u: p

        for u, p
        in pose_paths.items()

        if np.isfinite(
            p
        )
    }

    max_path = max(
        finite_paths.values(),
        default=0.0,
    )

    active_psms: list[int] = []

    for usm in instrument_usms:

        path = finite_paths.get(
            usm,
            np.nan,
        )

        active = bool(

            np.isfinite(
                path
            )

            and max_path > 0

            and path >= (
                0.02
                * max_path
            )
        )

        features[
            f"USM{usm}_pose_global_active"
        ] = int(
            active
        )

        if active:
            active_psms.append(
                usm
            )

    features[
        "active_psm_count"
    ] = len(
        active_psms
    )

    # ========================================================
    # EXISTING PAIR COORDINATION
    # ========================================================

    pair_cache: dict[
        tuple[
            int,
            int,
        ],
        dict[
            str,
            float,
        ],
    ] = {}

    for a, b in combinations(
        instrument_usms,
        2,
    ):

        prefix = (
            f"USM{a}_USM{b}_pose_global"
        )

        if (
            a in trajectories
            and b in trajectories
        ):

            pair_features = (
                v3.compute_pair_coordination_features(

                    trajectories[
                        a
                    ],

                    trajectories[
                        b
                    ],

                    prefix,
                )
            )

        else:

            pair_features = {

                f"{prefix}_n_aligned_samples":
                    0
            }

        features.update(
            pair_features
        )

        pair_cache[
            (
                a,
                b,
            )
        ] = pair_features

    # --------------------------------------------------------
    # Existing active-pair logic:
    # two PSMs with the largest path length.
    # --------------------------------------------------------

    ranked = sorted(
        finite_paths,
        key=finite_paths.get,
        reverse=True,
    )

    if len(
        ranked
    ) >= 2:

        a, b = sorted(
            ranked[
                :2
            ]
        )

        features[
            "active_pair_usm_a"
        ] = a

        features[
            "active_pair_usm_b"
        ] = b

        source_prefix = (
            f"USM{a}_USM{b}_pose_global"
        )

        for key, value in (
            pair_cache.get(
                (
                    a,
                    b,
                ),
                {},
            )
            .items()
        ):

            suffix = (
                key.removeprefix(
                    source_prefix
                    + "_"
                )
            )

            features[
                f"active_pair_{suffix}"
            ] = value

    else:

        features[
            "active_pair_usm_a"
        ] = np.nan

        features[
            "active_pair_usm_b"
        ] = np.nan

    return features


# ============================================================
# FEATURE TABLE
# ============================================================

def build_feature_table(
    fk_dir: Path,
    output_dir: Path,
    v3,
) -> pd.DataFrame:
    """
    Extract one row per FK file and save
    an unmerged feature table.
    """

    fk_files = sorted(
        fk_dir.rglob(
            "*_FK.csv"
        )
    )

    print(
        f"Found {len(fk_files)} "
        f"FK output files."
    )

    if not fk_files:

        raise FileNotFoundError(
            f"No *_FK.csv files found under: "
            f"{fk_dir}"
        )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    errors: list[
        dict[
            str,
            str,
        ]
    ] = []

    for i, fk_path in enumerate(
        fk_files,
        start=1,
    ):

        print(
            f"[FK+camera features "
            f"{i}/{len(fk_files)}] "
            f"{fk_path.name}"
        )

        try:

            rows.append(
                extract_trial_features_fk_only(
                    fk_path,
                    v3,
                )
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            errors.append({

                "fk_file":
                    str(
                        fk_path
                    ),

                "error":
                    repr(
                        exc
                    ),

                "traceback":
                    traceback.format_exc(),
            })

    error_path = (
        output_dir
        / "dataset_v7_fk_camera_processing_errors.csv"
    )

    pd.DataFrame(
        errors
    ).to_csv(
        error_path,
        index=False,
    )

    if not rows:

        raise ValueError(
            "No FK operation was processed. "
            f"Inspect: {error_path}"
        )

    table = pd.DataFrame(
        rows
    )

    duplicate = (
        table.duplicated(
            v3.MERGE_KEYS,
            keep=False,
        )
    )

    if duplicate.any():

        duplicate_path = (
            output_dir
            / "dataset_v7_fk_camera_duplicate_keys.csv"
        )

        cols = [

            c

            for c
            in (
                v3.MERGE_KEYS
                +
                [
                    "file_name",
                    "fk_file_name",
                ]
            )

            if c
            in table.columns
        ]

        table.loc[
            duplicate,
            cols,
        ].to_csv(
            duplicate_path,
            index=False,
        )

        raise ValueError(
            "Duplicate operation keys found. "
            f"Inspect: {duplicate_path}"
        )

    feature_path = (
        output_dir
        / "dataset_v7_fk_camera_features_unmerged.csv"
    )

    table.to_csv(
        feature_path,
        index=False,
    )

    print(
        f"\nSuccessfully processed "
        f"{len(table)} of "
        f"{len(fk_files)} FK files."
    )

    print(
        "Unmerged FK + camera features "
        f"saved to: {feature_path}"
    )

    if errors:

        print(
            f"Processing errors: "
            f"{len(errors)}. "
            f"Inspect: {error_path}"
        )

    # --------------------------------------------------------
    # CAMERA QC SUMMARY
    # --------------------------------------------------------

    camera_cols = [

        c

        for c
        in [
            "camera_path_length_m",
            "camera_mean_speed_m_s",
            "camera_max_speed_m_s",
            "camera_move_count",
            "camera_moving_duration_s",
            "camera_moving_fraction",
            "camera_rotation_path_radians",
            "camera_mean_angular_speed_rad_s",
            "camera_to_USM0_mean_distance_m",
            "camera_to_USM2_mean_distance_m",
            "camera_to_USM3_mean_distance_m",
            "camera_to_primary_tools_mean_distance_m",
            "camera_to_all_tools_mean_distance_m",
        ]

        if c
        in table.columns
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CAMERA FEATURE QC"
    )

    print(
        "=" * 80
    )

    if camera_cols:

        print(
            table[
                camera_cols
            ]
            .describe()
            .T
            .to_string()
        )

    else:

        print(
            "No camera columns generated."
        )

    if (
        "camera_path_length_m"
        in table.columns
    ):

        print(
            "\nCamera features available for:"
        )

        print(
            f"{table['camera_path_length_m'].notna().sum()}"
            f" / {len(table)} operations"
        )

    if (
        "camera_move_count"
        in table.columns
    ):

        print(
            "\nCamera move count distribution:"
        )

        print(
            table[
                "camera_move_count"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

    return table


# ============================================================
# FINAL LABELLED DATASET
# ============================================================

def build_dataset(
    fk_dir: Path,
    label_dir: Path,
    output_dir: Path,
    v3,
) -> pd.DataFrame:
    """
    Build the final labelled FK + camera dataset.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = (
        build_feature_table(
            fk_dir,
            output_dir,
            v3,
        )
    )

    # --------------------------------------------------------
    # Remove tasks outside current study
    # --------------------------------------------------------

    excluded_tasks = {
        "chicken_thigh",
        "cyst_model",
    }

    before = len(
        features
    )

    features = (
        features[
            ~features[
                "task_clean"
            ].isin(
                excluded_tasks
            )
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nExcluded "
        f"{before - len(features)} operations "
        f"from tasks: "
        f"{sorted(excluded_tasks)}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = (
        v3.build_labels(
            label_dir
        )
    )

    labels_path = (
        output_dir
        / "dataset_v7_fk_camera_labels_aggregated.csv"
    )

    labels.to_csv(
        labels_path,
        index=False,
    )

    # --------------------------------------------------------
    # Merge audit
    # --------------------------------------------------------

    feature_audit_cols = [

        c

        for c
        in (
            v3.MERGE_KEYS
            +
            [
                "file_name",
                "fk_file_name",
            ]
        )

        if c
        in features.columns
    ]

    label_audit_cols = [

        c

        for c
        in (
            v3.MERGE_KEYS
            +
            [
                "target_score",
                "percentage_score",
                "n_label_records",
            ]
        )

        if c
        in labels.columns
    ]

    audit = (
        features[
            feature_audit_cols
        ]
        .merge(

            labels[
                label_audit_cols
            ],

            on=
                v3.MERGE_KEYS,

            how="outer",

            indicator=True,

            validate=
                "one_to_one",
        )
    )

    audit_path = (
        output_dir
        / "dataset_v7_fk_camera_merge_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final merge
    # --------------------------------------------------------

    dataset = features.merge(

        labels,

        on=
            v3.MERGE_KEYS,

        how="inner",

        validate=
            "one_to_one",
    )

    if dataset.empty:

        raise ValueError(
            "Merged FK-camera dataset is empty. "
            f"Inspect: {audit_path}"
        )

    dataset = (
        dataset
        .sort_values(
            v3.MERGE_KEYS
        )
        .reset_index(
            drop=True
        )
    )

    dataset_path = (
        output_dir
        / "dataset_v7_fk_camera.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "dataset_version":
            "v7_fk_camera",

        "one_row_represents":
            "one session-aware scored operation",

        "raw_dvst_used":
            False,

        "feature_source":
            (
                "supervisor FK optimiser "
                "output only"
            ),

        "cartesian_source":
            (
                "True_Tip_X, "
                "True_Tip_Y, "
                "True_Tip_Z"
            ),

        "rotation_source":
            "True_Tip_Matrix",

        "coordinate_frame":
            (
                "calibrated global/base-referenced "
                "FK output"
            ),

        "camera_usm":
            CAMERA_USM,

        "camera_role":
            "ECM_Camera",

        "primary_tool_usms":
            list(
                PRIMARY_TOOL_USMS
            ),

        "psm_usms":
            list(
                PSM_USMS
            ),

        "camera_speed_threshold_m_s":
            CAMERA_SPEED_THRESHOLD_M_S,

        "camera_min_move_duration_s":
            CAMERA_MIN_MOVE_DURATION_S,

        "joint_features_included":
            False,

        "setup_features_included":
            False,

        "old_endoscope_position_features_included":
            False,

        "camera_features_included":
            True,

        "merge_keys":
            list(
                v3.MERGE_KEYS
            ),

        "number_fk_feature_rows":
            int(
                len(
                    features
                )
            ),

        "number_aggregated_label_rows":
            int(
                len(
                    labels
                )
            ),

        "number_final_merged_rows":
            int(
                len(
                    dataset
                )
            ),

        "output":
            str(
                dataset_path
            ),

        "merge_audit":
            str(
                audit_path
            ),
    }

    manifest_path = (
        output_dir
        / "dataset_v7_fk_camera_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print(
        "\nSaved final dataset:"
    )

    print(
        dataset_path
    )

    print(
        f"\nFinal shape: "
        f"{dataset.shape}"
    )

    print(
        "\nMerge audit:"
    )

    print(
        audit[
            "_merge"
        ].value_counts(
            dropna=False
        )
    )

    if (
        "task_clean"
        in dataset.columns
    ):

        print(
            "\nTask counts:"
        )

        print(
            dataset[
                "task_clean"
            ].value_counts(
                dropna=False
            )
        )

    if (
        "target_score"
        in dataset.columns
    ):

        print(
            "\nTarget summary:"
        )

        print(
            dataset[
                "target_score"
            ].describe()
        )

    # --------------------------------------------------------
    # Final camera QC after label merge
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL CAMERA QC"
    )

    print(
        "=" * 80
    )

    final_camera_cols = [

        c

        for c
        in dataset.columns

        if c.startswith(
            "camera_"
        )
    ]

    print(
        f"Camera feature columns: "
        f"{len(final_camera_cols)}"
    )

    if (
        "camera_path_length_m"
        in dataset.columns
    ):

        print(
            "Operations with valid camera path:"
        )

        print(
            f"{dataset['camera_path_length_m'].notna().sum()}"
            f" / {len(dataset)}"
        )

    if (
        "camera_move_count"
        in dataset.columns
    ):

        print(
            "\nCamera movement count:"
        )

        print(
            dataset[
                "camera_move_count"
            ]
            .describe()
        )

    if (
        "camera_to_primary_tools_mean_distance_m"
        in dataset.columns
    ):

        print(
            "\nCamera-to-primary-tools distance:"
        )

        print(
            dataset[
                "camera_to_primary_tools_mean_distance_m"
            ]
            .describe()
        )

    print(
        f"\nManifest saved to: "
        f"{manifest_path}"
    )

    return dataset


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--fk-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v3-script",
        type=Path,
        required=True,
    )

    return (
        parser.parse_args()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    for path in [
        args.fk_dir,
        args.label_dir,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if not (
        args.v3_script.exists()
    ):

        raise FileNotFoundError(
            args.v3_script
        )

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    v3 = (
        load_v3_module(
            args.v3_script
            .resolve()
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=
                RuntimeWarning,
        )

        build_dataset(

            fk_dir=
                args.fk_dir.resolve(),

            label_dir=
                args.label_dir.resolve(),

            output_dir=
                output_dir,

            v3=v3,
        )


if __name__ == "__main__":
    main()
    """python C:/Users/ROG/IHE-project/code/regression_task/build_dataset_with_camera.py `
  --fk-dir "C:/Users/ROG/IHE-project/data/updated_fk" `
  --label-dir "C:/Users/ROG/IHE-project/Kinematics/corrected/Griffin_Training_Dataset/M-GEARS" `
  --output-dir "C:/Users/ROG/IHE-project/data/v7_fk_camera" `
  --v3-script "C:/Users/ROG/IHE-project/code/regression_task/dataset_v3.py"
  """
