"""
da Vinci Xi Forward Kinematics Optimizer

This module parses raw kinematic logs, dynamically solves for Endoscope optical offsets
using Levenberg-Marquardt scale-aware optimization, and generates a 3D Plotly animation
of the physical robot skeleton. Nomenclature strictly follows the official Intuitive
Surgical API Xi kinematics documentation.
"""

import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
"""INPUT_CSV: str = "DVST_XI.csv"
INTERMEDIATE_CSV: str = "DVST_XI_FK_Results.csv"
OUTPUT_HTML: str = "robot_skeleton.html" """

INPUT_CSV: str = "C:/Users/ROG/IHE-project/Kinematics/dataset/Griffin_Training_Dataset/DVST_XI/DVST_XI_6-8_Mar_2025_06_10.41.14_trainee_31_ring rollercoaster_1.csv"
INTERMEDIATE_CSV: str = "C:/Users/ROG/IHE-project/DVST_XI_Test_Results.csv"
OUTPUT_HTML: str = "C:/Users/ROG/IHE-project/test_robot_skeleton.html"

# Universal Surgical Manipulator (USM) index corresponding to the Endoscope Camera
ECM_USM_IDX: int = 1

# List of USMs currently attached and active in the surgical workspace
ACTIVE_USMS: List[int] = [0, 1, 2, 3]

# Unit Conversions
INCHES_TO_METERS: float = 0.0254
METERS_TO_INCHES: float = 1.0 / 0.0254

# The physical metal length of the tools (Assumed standard for drawing purposes)
ECM_SHAFT_LENGTH_METERS: float = 0.600  # From official Intuitive documentation
PSM_SHAFT_LENGTH_METERS: float = 20.0 * INCHES_TO_METERS

# ==============================================================================
# STATIC DH PARAMETERS
# ==============================================================================
# Parameters formatted as: [alpha_degrees, a_inches, theta_offset_degrees, d_inches]

# Setup Structure (SUS) Parameters
SUS_DH: List[List[float]] = [
    [0.0, 0.0, 90.0, 75.50579],
    [0.0, 0.0, 0.0, 0.0],
    [90.0, 0.0, 0.0, 41.0],
    [-90.0, 0.0, 90.0, -2.85579],
]

# Setup Joint (SUJ) Parameters per USM
SUJ_DH_CONFIG: Dict[int, List[List[float]]] = {
    0: [
        [0.0, 0.0, 90.0, 0.0],
        [0.0, 6.6250, 55.0, 0.0],
        [90.0, -0.7980, 0.0, 25.1230],
        [-90.0, 0.0, 0.0, -10.58027],
        [0.0, 0.0, 0.0, -14.77827],
    ],
    1: [
        [0.0, 0.0, 30.0, 0.0],
        [0.0, 6.6250, 92.50, 0.0],
        [90.0, -5.280, 0.0, 17.7210],
        [-90.0, 0.0, 0.0, -10.58056],
        [0.0, 0.0, 0.0, -6.34690],
    ],
    2: [
        [0.0, 0.0, -30.0, 0.0],
        [0.0, 6.6250, 87.50, 0.0],
        [90.0, -5.280, 0.0, 17.7210],
        [-90.0, 0.0, 0.0, -10.58056],
        [0.0, 0.0, 0.0, -6.34690],
    ],
    3: [
        [0.0, 0.0, -90.0, 0.0],
        [0.0, 6.6250, 126.0, 0.0],
        [90.0, -7.9790, 0.0, 23.4220],
        [-90.0, 0.0, 0.0, -10.58027],
        [0.0, 0.0, 0.0, -14.77827],
    ],
}

# Universal Surgical Manipulator (USM) Parameters
USM_DH_CONFIG: Dict[int, List[List[float]]] = {
    0: [
        [62.0, 0.0, 90.0, -31.84792],
        [15.0, 0.0, 0.0, 11.96628],
        [-87.20, -1.07596, -28.030, -0.58455],
        [0.0, 10.0, 112.8860, 0.0],
        [0.0, 12.0, 5.1440, 0.0],
        [-90.0, 4.69966, 0.0, -13.30303],
    ],
    1: [
        [45.0, 0.0, 90.0, -33.0470],
        [15.0, 0.0, 0.0, 11.96628],
        [-87.20, -1.07596, -28.030, -0.58455],
        [0.0, 10.0, 112.8860, 0.0],
        [0.0, 12.0, 5.1440, 0.0],
        [-90.0, 4.69966, 0.0, -13.30303],
    ],
    2: [
        [45.0, 0.0, 90.0, -33.0470],
        [15.0, 0.0, 0.0, 11.96628],
        [-87.20, -1.07596, -28.030, -0.58455],
        [0.0, 10.0, 112.8860, 0.0],
        [0.0, 12.0, 5.1440, 0.0],
        [-90.0, 4.69966, 0.0, -13.30303],
    ],
    3: [
        [62.0, 0.0, 90.0, -31.84792],
        [15.0, 0.0, 0.0, 11.96628],
        [-87.20, -1.07596, -28.030, -0.58455],
        [0.0, 10.0, 112.8860, 0.0],
        [0.0, 12.0, 5.1440, 0.0],
        [-90.0, 4.69966, 0.0, -13.30303],
    ],
}


# ==============================================================================
# CORE MATH FUNCTIONS
# ==============================================================================
def modified_dh_matrix(alpha_deg: float, a: float, theta_rad: float, d: float) -> np.ndarray:
    """
    Computes the Modified Denavit-Hartenberg (MDH) homogeneous transformation matrix.

    This implements Khalil's convention as specified in the da Vinci Xi kinematics
    documentation, where the coordinate frame is attached to the j-th joint.

    Args:
        alpha_deg (float): Twist angle in degrees.
        a (float): Link length in meters.
        theta_rad (float): Joint angle in radians.
        d (float): Link offset in meters.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix of type float.
    """
    alpha_rad: float = np.radians(alpha_deg)
    ca: float = np.cos(alpha_rad)
    sa: float = np.sin(alpha_rad)
    ct: float = np.cos(theta_rad)
    st: float = np.sin(theta_rad)

    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


# ==============================================================================
# KINEMATIC CHAINING
# ==============================================================================
def compute_chain(
        dh_parameters: List[List[float]],
        joint_values: List[float],
        joint_types: List[str],
        initial_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    """
    Computes the forward kinematics for a sequential chain of joints using MDH parameters.

    This function steps through the hardware chain, dynamically converting imperial units
    to metric, applying joint values based on the physical joint type, and tracking the
    absolute 3D origin of each linkage for downstream skeleton visualization.

    Args:
        dh_parameters (List[List[float]]): Static physical parameters for each joint.
            Formatted as [alpha_degrees, a_inches, theta_offset_degrees, d_inches].
        joint_values (List[float]): Active API tracking values for the joints (radians
            for Revolute, meters for Prismatic).
        joint_types (List[str]): Strings defining the mechanical nature of each joint
            ("Revolute", "Prismatic", or "Dummy" for fixed offsets).
        initial_transform (Optional[np.ndarray]): A 4x4 matrix to attach this chain
            to an existing physical structure. Defaults to an Identity matrix.

    Returns:
        Tuple[np.ndarray, List[Tuple[float, float, float]]]:
            - current_transform: The final 4x4 coordinate frame at the end of the chain.
            - points: A chronological list of (x, y, z) physical hinge coordinates.
    """
    current_transform: np.ndarray = initial_transform if initial_transform is not None else np.eye(4)

    # Store the absolute spatial origin point of the starting matrix
    points: List[Tuple[float, float, float]] = [
        (float(current_transform[0, 3]), float(current_transform[1, 3]), float(current_transform[2, 3]))
    ]

    # --- Step through each joint sequentially ---
    for idx, params in enumerate(dh_parameters):
        alpha_deg: float = params[0]
        a_inches: float = params[1]
        theta_offset_deg: float = params[2]
        d_offset_inches: float = params[3]

        a_meters: float = a_inches * INCHES_TO_METERS
        d_offset_meters: float = d_offset_inches * INCHES_TO_METERS

        theta_rad: float
        d_meters: float

        # Apply kinematic sensor values according to mechanical joint constraints
        if joint_types[idx] == "Revolute":
            theta_rad = np.radians(theta_offset_deg) + joint_values[idx]
            d_meters = d_offset_meters
        elif joint_types[idx] == "Prismatic":
            theta_rad = np.radians(theta_offset_deg)
            d_meters = d_offset_meters + joint_values[idx]
        else:
            # Dummy/Fixed joint: No active sensor value applied
            theta_rad = np.radians(theta_offset_deg)
            d_meters = d_offset_meters

        # Step the transformation matrix forward and record the new spatial hinge point
        step_transform: np.ndarray = modified_dh_matrix(alpha_deg, a_meters, theta_rad, d_meters)
        current_transform = np.dot(current_transform, step_transform)

        points.append(
            (float(current_transform[0, 3]), float(current_transform[1, 3]), float(current_transform[2, 3]))
        )

    return current_transform, points


def compute_arm_fk(
        usm_idx: int,
        sus_joint_vals: List[float],
        suj_joint_vals: List[float],
        usm_joint_vals: List[float],
        static_z_offset_meters: float = 0.0,
) -> Tuple[np.ndarray, str, str, str]:
    """
    Computes the full physical forward kinematics for a specific robot arm.

    This maps the coordinate frames sequentially from the robot Setup Structure (SUS),
    through the Setup Joints (SUJ), and down the Universal Surgical Manipulator (USM).
    It applies the physical roll at the carriage and visually extends the metal shaft.

    Args:
        usm_idx (int): The integer index of the Universal Surgical Manipulator (0 to 3).
        sus_joint_vals (List[float]): The 4 joint values for the Setup Structure.
        suj_joint_vals (List[float]): The 4 joint values for the Setup Joints.
        usm_joint_vals (List[float]): The 6 joint values for the USM.
        static_z_offset_meters (float): The physical length of the metal tool shaft
            to draw visually extending from the carriage. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, str, str, str]:
            - matrix_carriage: The 4x4 coordinate frame of the USM carriage.
            - sus_string: Pipe-separated string of SUS 3D coordinates.
            - suj_string: Pipe-separated string of SUJ 3D coordinates.
            - usm_string: Pipe-separated string of USM 3D coordinates (ending at shaft tip).
    """
    # --- 1. Setup Structure (SUS / Gantry) ---
    sus_types: List[str] = ["Prismatic", "Revolute", "Prismatic", "Revolute"]
    matrix_sus, sus_points = compute_chain(SUS_DH, sus_joint_vals, sus_types, np.eye(4))

    # --- 2. Setup Joints (SUJ) ---
    # The first SUJ parameter is a dummy/fixed offset, so we pad a 0.0 to the sensor values.
    suj_active_vals: List[float] = [0.0, suj_joint_vals[0], suj_joint_vals[1], suj_joint_vals[2], suj_joint_vals[3]]
    suj_types: List[str] = ["Dummy", "Revolute", "Prismatic", "Prismatic", "Revolute"]
    matrix_suj, suj_points = compute_chain(SUJ_DH_CONFIG[usm_idx], suj_active_vals, suj_types, matrix_sus)

    # --- 3. Universal Surgical Manipulator (USM) ---
    theta_tornado: float = usm_joint_vals[0]
    theta_yaw: float = usm_joint_vals[1]
    theta_pitch: float = usm_joint_vals[2]
    d_insertion: float = usm_joint_vals[3]

    # The pitch joint mechanically couples two links (pitch and -pitch).
    usm_active_vals: List[float] = [
        theta_tornado, theta_yaw, theta_pitch, -theta_pitch, theta_pitch, d_insertion
    ]
    usm_types: List[str] = ["Revolute", "Revolute", "Revolute", "Revolute", "Revolute", "Prismatic"]
    matrix_usm, usm_points = compute_chain(USM_DH_CONFIG[usm_idx], usm_active_vals, usm_types, matrix_suj)

    # --- 4. Final Carriage Frame & Physical Tool Shaft ---
    matrix_carriage: np.ndarray = np.array(matrix_usm)

    # Pure physical roll mapped at the carriage level (Roll is the 5th USM joint value).
    theta_roll: float = usm_joint_vals[4]
    matrix_carriage_rot: np.ndarray = np.eye(4)
    matrix_carriage_rot[0, 0] = np.cos(theta_roll)
    matrix_carriage_rot[0, 1] = -np.sin(theta_roll)
    matrix_carriage_rot[1, 0] = np.sin(theta_roll)
    matrix_carriage_rot[1, 1] = np.cos(theta_roll)

    matrix_carriage = np.dot(matrix_carriage, matrix_carriage_rot)

    # Append the straight physical shaft point extending down the local Z-axis.
    matrix_physical_offset: np.ndarray = np.eye(4)
    matrix_physical_offset[2, 3] = static_z_offset_meters
    matrix_physical_tip: np.ndarray = np.dot(matrix_carriage, matrix_physical_offset)

    physical_tip_point: Tuple[float, float, float] = (
        float(matrix_physical_tip[0, 3]),
        float(matrix_physical_tip[1, 3]),
        float(matrix_physical_tip[2, 3]),
    )

    # --- 5. Format Point Arrays for CSV Logging ---
    sus_string: str = "|".join([f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in sus_points])
    suj_string: str = "|".join([f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in suj_points])

    # Appending the physical tip ensures the visual skeleton draws all the way down the shaft
    usm_visual_points: List[Tuple[float, float, float]] = usm_points + [physical_tip_point]
    usm_string: str = "|".join([f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in usm_visual_points])

    return matrix_carriage, sus_string, suj_string, usm_string


# ==============================================================================
# LEVENBERG-MARQUARDT OPTIMIZER
# ==============================================================================
def optimize_kinematics(
        data_by_timestamp: Dict[str, List[Dict[str, str]]],
        target_psm_indices: Tuple[int, ...] = (0, 2),
        ecm_idx: int = 1,
) -> np.ndarray:
    """
    Optimizes the camera optical offsets and spatial scaling factor.

    This extracts "bookend" freeze events where the Endoscope (ECM) is moving but
    the instruments (PSMs) are stationary. It runs a Levenberg-Marquardt optimization
    to find the exact optical offsets and API scaling factor that minimize the drift
    of the stationary tooltips.

    Args:
        data_by_timestamp (Dict[str, List[Dict[str, str]]]): Raw CSV rows grouped by timestamp.
        target_psm_indices (Tuple[int, ...]): Integer indices of active PSM arms. Defaults to (0, 2).
        ecm_idx (int): Integer index of the ECM arm. Defaults to 1.

    Returns:
        np.ndarray: A 1D array of 6 optimized parameters:
            [scale, dx_meters, dy_meters, rx_radians, ry_radians, rz_radians].
    """
    print(
        f"Extracting Calibration Events. Anchoring Physical ECM Shaft Length to "
        f"{ECM_SHAFT_LENGTH_METERS * METERS_TO_INCHES:.2f} inches..."
    )

    times: List[float] = []
    ecm_data: List[Dict[str, List[float]]] = []
    psm_data_dict: Dict[int, List[Dict[str, List[float]]]] = {idx: [] for idx in target_psm_indices}
    scope_data_dict: Dict[int, List[List[float]]] = {idx: [] for idx in target_psm_indices}

    # --- 1. Parse and validate synchronized tracking data ---
    for t_str, rows in data_by_timestamp.items():
        t: float = float(t_str)
        ecm_row: Optional[Dict[str, str]] = next((r for r in rows if int(r["USM"]) == ecm_idx), None)

        valid_frame: bool = True
        psm_rows: Dict[int, Dict[str, str]] = {}

        if not ecm_row:
            valid_frame = False
        else:
            for idx in target_psm_indices:
                r: Optional[Dict[str, str]] = next((row for row in rows if int(row["USM"]) == idx), None)
                if r and r.get("EndoscopePosition"):
                    vals: List[float] = list(map(float, r["EndoscopePosition"].split()))
                    if len(vals) == 12:
                        psm_rows[idx] = r
                    else:
                        valid_frame = False
                        break
                else:
                    valid_frame = False
                    break

        if valid_frame and ecm_row is not None:
            times.append(t)
            ecm_data.append(
                {
                    "sus": list(map(float, ecm_row["SUSvalues"].split())),
                    "suj": list(map(float, ecm_row["SUJvalues"].split())),
                    "usm": list(map(float, ecm_row["USMJointValues"].split())),
                }
            )
            for idx in target_psm_indices:
                p_row: Dict[str, str] = psm_rows[idx]
                psm_data_dict[idx].append(
                    {
                        "sus": list(map(float, p_row["SUSvalues"].split())),
                        "suj": list(map(float, p_row["SUJvalues"].split())),
                        "usm": list(map(float, p_row["USMJointValues"].split())),
                    }
                )
                scope_data_dict[idx].append(list(map(float, p_row["EndoscopePosition"].split())))

    # Fallback if no valid synchronized data exists
    if not times:
        print("No valid timestamps found for calibration.")
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Ensure uniqueness in timestamps
    times_arr: np.ndarray
    unique_indices: np.ndarray
    times_arr, unique_indices = np.unique(np.array(times), return_index=True)

    # Explicit int() cast prevents Union slice warnings in strict static type checkers
    ecm_data = [ecm_data[int(i)] for i in unique_indices]
    for idx in target_psm_indices:
        scope_data_dict[idx] = [scope_data_dict[idx][int(i)] for i in unique_indices]
        psm_data_dict[idx] = [psm_data_dict[idx][int(i)] for i in unique_indices]

    # --- 2. Isolate "Bookend" Freeze Events ---
    calibration_pairs: Dict[int, List[Tuple[int, int]]] = {idx: [] for idx in target_psm_indices}
    for idx in target_psm_indices:
        in_freeze: bool = False
        start_idx: int = -1

        for i in range(1, len(times_arr)):
            ecm_v: float = 0.0
            for k in ["sus", "suj", "usm"]:
                ecm_v += float(np.sum(np.abs(np.array(ecm_data[i][k]) - np.array(ecm_data[i - 1][k]))))

            cv_v: float = float(
                np.sum(np.abs(np.array(scope_data_dict[idx][i]) - np.array(scope_data_dict[idx][i - 1])))
            )

            if cv_v < 1e-6 and ecm_v > 0.0005:
                if not in_freeze:
                    in_freeze = True
                    start_idx = i - 1
            elif in_freeze and cv_v > 0.01:
                end_idx: int = i
                in_freeze = False
                psm_parked: bool = True

                for k_idx in range(start_idx, end_idx + 1):
                    psm_v: float = 0.0
                    for jt in ["sus", "suj", "usm"]:
                        curr_jt = np.array(psm_data_dict[idx][k_idx][jt])
                        prev_jt = np.array(psm_data_dict[idx][k_idx - 1][jt])
                        psm_v += float(np.sum(np.abs(curr_jt - prev_jt)))

                    if psm_v > 0.0015:
                        psm_parked = False
                        break

                if psm_parked:
                    calibration_pairs[idx].append((start_idx, end_idx))

            elif in_freeze and ecm_v < 0.0001:
                in_freeze = False

    # --- 3. Residual Function ---
    def residuals(params: np.ndarray) -> np.ndarray:
        """Calculates the positional and rotational drift of the stationary tooltips."""
        scale: float = float(params[0])
        dx_m: float = float(params[1])
        dy_m: float = float(params[2])
        rx: float = float(params[3])
        ry: float = float(params[4])
        rz: float = float(params[5])

        errors: List[float] = []

        for idx in target_psm_indices:
            for idx_pre, idx_snap in calibration_pairs[idx]:
                matrix_globals: List[np.ndarray] = []

                for i in [idx_pre, idx_snap]:
                    matrix_base_ecm, _, _, _ = compute_arm_fk(
                        usm_idx=ecm_idx,
                        sus_joint_vals=ecm_data[i]["sus"],
                        suj_joint_vals=ecm_data[i]["suj"],
                        usm_joint_vals=ecm_data[i]["usm"],
                    )

                    matrix_rx: np.ndarray = np.array([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx), 0.0],
                        [0.0, np.sin(rx), np.cos(rx), 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
                    matrix_ry: np.ndarray = np.array([
                        [np.cos(ry), 0.0, np.sin(ry), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry), 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
                    matrix_rz: np.ndarray = np.array([
                        [np.cos(rz), -np.sin(rz), 0.0, 0.0],
                        [np.sin(rz), np.cos(rz), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])

                    matrix_opt: np.ndarray = np.dot(matrix_rz, np.dot(matrix_ry, matrix_rx))
                    matrix_opt[0, 3] = dx_m
                    matrix_opt[1, 3] = dy_m
                    matrix_opt[2, 3] = ECM_SHAFT_LENGTH_METERS

                    matrix_cam_global: np.ndarray = np.dot(matrix_base_ecm, matrix_opt)

                    # Apply theoretical scale factor to the raw API Translation matrix
                    cv_matrix_array: List[float] = scope_data_dict[idx][i]
                    matrix_ecm_to_psm: np.ndarray = np.eye(4)
                    matrix_ecm_to_psm[0:3, 3] = np.array(cv_matrix_array[0:3]) * scale
                    matrix_ecm_to_psm[:3, :3] = np.array(cv_matrix_array[3:12]).reshape((3, 3))

                    matrix_globals.append(np.dot(matrix_cam_global, matrix_ecm_to_psm))

                # Millimeter scale penalization for stability
                pos_drift_mm: np.ndarray = (matrix_globals[1][0:3, 3] - matrix_globals[0][0:3, 3]) * 1000.0
                rot_drift: np.ndarray = (matrix_globals[1][0:3, 0:3] - matrix_globals[0][0:3, 0:3]).flatten() * 10.0

                errors.extend(pos_drift_mm.tolist())
                errors.extend(rot_drift.tolist())

        return np.array(errors)

    # --- 4. Execution ---
    initial_guess: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Opened limits to allow the optimizer to find true zero
    opt_bounds: Tuple[List[float], List[float]] = (
        [0.01, -0.2, -0.2, -np.pi, -np.pi, -np.pi],
        [10.0, 0.2, 0.2, np.pi, np.pi, np.pi],
    )

    print("Launching Scale-Aware Optimizer (TRF)...")
    result = least_squares(residuals, initial_guess, bounds=opt_bounds, method="trf", verbose=2)

    print("\n" + "=" * 50)
    print("Optimization Results (5-DOF + Scale)")
    print("=" * 50)
    print(f"Scale Factor (S): {result.x[0]:.6f}")
    print(f"Offset X (dx):    {result.x[1] * 1000:.3f} mm")
    print(f"Offset Y (dy):    {result.x[2] * 1000:.3f} mm")
    print(f"Rotation X (rx):  {np.degrees(result.x[3]):.3f} deg")
    print(f"Rotation Y (ry):  {np.degrees(result.x[4]):.3f} deg")
    print(f"Rotation Z (rz):  {np.degrees(result.x[5]):.3f} deg")
    print("=" * 50 + "\n")

    return result.x


# ==============================================================================
# WORKFLOW PROCESSING AND LOGGING
# ==============================================================================
def process_and_log_kinematics(
        data_by_timestamp: Dict[str, List[Dict[str, str]]],
        writer: csv.DictWriter,
        opt_params: np.ndarray,
) -> None:
    """
    Applies optimized optical parameters across the timeline to compute absolute coordinates.

    Maps the mathematically derived optical center and API scale factor to the dataset.
    Implements a state machine to freeze instrument tips during periods of API latency
    (camera clutching) to prevent visual jitter.

    Args:
        data_by_timestamp: The raw CSV rows grouped by timestamp.
        writer: CSV DictWriter for logging the computed absolute coordinates.
        opt_params: Optimized parameters [scale, dx, dy, rx, ry, rz].
    """
    scale: float = float(opt_params[0])
    dx_m: float = float(opt_params[1])
    dy_m: float = float(opt_params[2])
    rx: float = float(opt_params[3])
    ry: float = float(opt_params[4])
    rz: float = float(opt_params[5])

    # --- 1. Extract API tracking data for PSM instruments ---
    scope_data_dict: Dict[int, List[List[float]]] = defaultdict(list)
    valid_times_dict: Dict[int, List[float]] = defaultdict(list)

    for t_str, rows in data_by_timestamp.items():
        t: float = float(t_str)
        for row in rows:
            usm_idx: int = int(row["USM"])
            if usm_idx not in ACTIVE_USMS or usm_idx == ECM_USM_IDX:
                continue

            raw_endo: str = row.get("EndoscopePosition", "")
            if raw_endo:
                vals: List[float] = list(map(float, raw_endo.split()))
                if len(vals) == 12:
                    scope_data_dict[usm_idx].append(vals)
                    valid_times_dict[usm_idx].append(t)

    # --- 2. Build Cubic Interpolators to handle API polling gaps ---
    scope_interpolators: Dict[int, interp1d] = {}
    for usm_idx in scope_data_dict:
        usm_times, unique_idx = np.unique(valid_times_dict[usm_idx], return_index=True)
        usm_data: np.ndarray = np.array(scope_data_dict[usm_idx])[unique_idx]
        scope_interpolators[usm_idx] = interp1d(
            usm_times, usm_data, axis=0, kind="cubic", fill_value="extrapolate"
        )

    # --- 3. Process timeline and project absolute tooltips ---
    prev_ecm_joints: Optional[np.ndarray] = None
    prev_raw_endo: Dict[int, np.ndarray] = {}
    last_stable_tips: Dict[int, np.ndarray] = {}
    freeze_state: Dict[int, Dict[str, Any]] = {}

    for t_str, rows in data_by_timestamp.items():
        t = float(t_str)
        ecm_row = next((r for r in rows if int(r["USM"]) == ECM_USM_IDX), None)
        if not ecm_row:
            continue

        sus_vals = list(map(float, ecm_row["SUSvalues"].split()))
        suj_vals = list(map(float, ecm_row["SUJvalues"].split()))
        ecm_usm_vals = list(map(float, ecm_row["USMJointValues"].split()))

        # Calculate Physical Skeleton Transformation
        T_carriage, sus_str, suj_str, ecm_str = compute_arm_fk(
            ECM_USM_IDX, sus_vals, suj_vals, ecm_usm_vals, ECM_SHAFT_LENGTH_METERS
        )

        # Apply Optical Offset Math
        T_opt: np.ndarray = np.dot(
            np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            np.dot(
                np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]]),
                np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
            )
        )
        T_opt[0, 3], T_opt[1, 3], T_opt[2, 3] = dx_m, dy_m, ECM_SHAFT_LENGTH_METERS

        # Derive pure carriage frame to compute absolute optical tip
        T_carriage_only, _, _, _ = compute_arm_fk(
            ECM_USM_IDX, sus_vals, suj_vals, ecm_usm_vals, 0.0
        )
        T_cam_global: np.ndarray = np.dot(T_carriage_only, T_opt)

        # Track ECM motion for state machine triggering
        curr_ecm: np.ndarray = np.concatenate([sus_vals, suj_vals, ecm_usm_vals])
        ecm_moving: bool = prev_ecm_joints is not None and float(np.sum(np.abs(curr_ecm - prev_ecm_joints))) > 0.0001
        prev_ecm_joints = curr_ecm

        writer.writerow({
            "TimeStamp": t_str,
            "USM": ECM_USM_IDX,
            "Role": "ECM_Camera",
            "Base_Position_Matrix": " ".join(map(str, T_cam_global.flatten())),
            "True_Tip_X": T_cam_global[0, 3],
            "True_Tip_Y": T_cam_global[1, 3],
            "True_Tip_Z": T_cam_global[2, 3],
            "True_Tip_Matrix": " ".join(map(str, T_cam_global.flatten())),
            "SUS_Links": sus_str,
            "SUJ_Links": suj_str,
            "USM_Links": ecm_str
        })

        # --- 4. Project PSM tooltips into absolute world space ---
        for row in rows:
            usm_idx = int(row["USM"])
            if usm_idx not in ACTIVE_USMS or usm_idx == ECM_USM_IDX:
                continue

            T_psm_carriage, _, psm_suj_str, psm_usm_str = compute_arm_fk(
                usm_idx, sus_vals, list(map(float, row["SUJvalues"].split())),
                list(map(float, row["USMJointValues"].split())), PSM_SHAFT_LENGTH_METERS
            )

            # Map absolute base coordinate for matrix logging and downstream math
            T_shaft: np.ndarray = np.eye(4)
            T_shaft[2, 3] = PSM_SHAFT_LENGTH_METERS
            T_base_psm: np.ndarray = np.dot(T_psm_carriage, T_shaft)

            final_x, final_y, final_z = np.nan, np.nan, np.nan
            final_matrix: Optional[np.ndarray] = None
            raw_endo: str = row.get("EndoscopePosition", "")

            # Apply tracking data with jitter-prevention logic
            if usm_idx in scope_interpolators:
                t_i = scope_interpolators[usm_idx].x
                if raw_endo and t_i[0] <= t <= t_i[-1]:
                    curr_raw: np.ndarray = np.array(list(map(float, raw_endo.split())))
                    base_ref: np.ndarray = prev_raw_endo.get(usm_idx, curr_raw)
                    delta: float = float(np.sum(np.abs(curr_raw - base_ref)))
                    prev_raw_endo[usm_idx] = curr_raw

                    if usm_idx not in freeze_state:
                        freeze_state[usm_idx] = {"frozen": False, "cooldown": 0}

                    # State machine: detect clutch/move triggers
                    if ecm_moving:
                        freeze_state[usm_idx].update({"frozen": True, "cooldown": 15})
                    elif freeze_state[usm_idx]["cooldown"] > 0:
                        freeze_state[usm_idx]["cooldown"] -= 1
                        if delta > 0.01: freeze_state[usm_idx].update({"frozen": False, "cooldown": 0})
                    else:
                        freeze_state[usm_idx]["frozen"] = False

                    # Project CV tracking into calibrated global frame
                    scope_pos = scope_interpolators[usm_idx](t)
                    T_ecm_to_psm = np.eye(4)
                    T_ecm_to_psm[0:3, 3] = scope_pos[0:3] * scale
                    T_ecm_to_psm[:3, :3] = scope_pos[3:12].reshape((3, 3))

                    cv_matrix: np.ndarray = np.dot(T_cam_global, T_ecm_to_psm)
                    if freeze_state[usm_idx]["frozen"] and usm_idx in last_stable_tips:
                        final_matrix = last_stable_tips[usm_idx]
                    else:
                        final_matrix = cv_matrix
                        last_stable_tips[usm_idx] = cv_matrix

                    final_x = float(final_matrix[0, 3])
                    final_y = float(final_matrix[1, 3])
                    final_z = float(final_matrix[2, 3])

            writer.writerow({
                "TimeStamp": t_str,
                "USM": usm_idx,
                "Role": "PSM_Instrument",
                "Base_Position_Matrix": " ".join(map(str, T_base_psm.flatten())),
                "True_Tip_X": final_x,
                "True_Tip_Y": final_y,
                "True_Tip_Z": final_z,
                "True_Tip_Matrix": " ".join(map(str, final_matrix.flatten())) if final_matrix is not None else "",
                "SUS_Links": sus_str,
                "SUJ_Links": psm_suj_str,
                "USM_Links": psm_usm_str
            })


# ==============================================================================
# PLOTLY VISUALIZER
# ==============================================================================
def create_combined_animation(csv_filename: str, output_html: str) -> None:
    """
    Generates a highly detailed interactive 3D HTML animation of the robot kinematics.

    This visualizer maps the structural gantry (SUS), the setup joints (SUJ), and
    the active tool arms (USM). It features historical tracking trails, text labels
    for the joint topology, and gray dashed lines to visualize the delta between
    the physical carriage and the mathematical optical tip.
    """
    print("Building full-featured 3D Plotly animation...")
    df: pd.DataFrame = pd.read_csv(csv_filename)

    # --- 1. Downsample and Parse ---
    downsampled_times: np.ndarray = df["TimeStamp"].unique()[::10]
    df = df[df["TimeStamp"].isin(downsampled_times)].copy()
    df["Frame"] = df["TimeStamp"].map({time: idx for idx, time in enumerate(downsampled_times)})

    def parse_links(link_str: str) -> Tuple[List[float], List[float], List[float]]:
        pts = [list(map(float, pt.split(","))) for pt in str(link_str).split("|")]
        return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

    def extract_axes(mat_str: Any) -> Tuple[List[float], List[float], List[float]]:
        """Extracts the local X, Y, and Z directional vectors from the matrix."""
        if pd.isna(mat_str) or not str(mat_str).strip():
            return [np.nan] * 3, [np.nan] * 3, [np.nan] * 3
        m = np.array(list(map(float, str(mat_str).split()))).reshape(4, 4)
        x_dir = m[:3, 0] * 0.05  # 0.05 is the length of the axis line in meters (5cm)
        y_dir = m[:3, 1] * 0.05
        z_dir = m[:3, 2] * 0.05
        return x_dir.tolist(), y_dir.tolist(), z_dir.tolist()

    df["SUS_X"], df["SUS_Y"], df["SUS_Z"] = zip(*df["SUS_Links"].apply(parse_links))
    df["SUJ_X"], df["SUJ_Y"], df["SUJ_Z"] = zip(*df["SUJ_Links"].apply(parse_links))
    df["USM_X"], df["USM_Y"], df["USM_Z"] = zip(*df["USM_Links"].apply(parse_links))

    df["Approx_Tip_X"] = df["USM_X"].apply(lambda x: x[-1])
    df["Approx_Tip_Y"] = df["USM_Y"].apply(lambda y: y[-1])
    df["Approx_Tip_Z"] = df["USM_Z"].apply(lambda z: z[-1])

    axes_data = df["True_Tip_Matrix"].apply(extract_axes)
    df["Axis_X"] = [x[0] for x in axes_data]
    df["Axis_Y"] = [x[1] for x in axes_data]
    df["Axis_Z"] = [x[2] for x in axes_data]

    # --- 2. Calculate Strict Bounding Box (Type-Safe) ---
    def get_bounds(col1: pd.Series, col2: pd.Series, col3: pd.Series) -> Tuple[float, float]:
        combined_floats: List[float] = [
            v for series in (col1, col2, col3) for sublist in series.dropna() for v in sublist
        ]
        return float(min(combined_floats)), float(max(combined_floats))

    raw_x_min, raw_x_max = get_bounds(df["SUS_X"], df["SUJ_X"], df["USM_X"])
    raw_y_min, raw_y_max = get_bounds(df["SUS_Y"], df["SUJ_Y"], df["USM_Y"])
    raw_z_min, raw_z_max = get_bounds(df["SUS_Z"], df["SUJ_Z"], df["USM_Z"])

    x_min, x_max = raw_x_min - 0.2, raw_x_max + 0.2
    y_min, y_max = raw_y_min - 0.2, raw_y_max + 0.2
    z_min, z_max = raw_z_min - 0.2, raw_z_max + 0.2

    # --- 3. Aesthetic Configuration ---
    arm_colors: Dict[int, Dict[str, str]] = {
        0: {"suj": "#8B0000", "usm": "#FF4500"},
        1: {"suj": "#006400", "usm": "#32CD32"},
        2: {"suj": "#00008B", "usm": "#00BFFF"},
        3: {"suj": "#B8860B", "usm": "#FFD700"},
    }

    sus_labels: List[str] = ["SUS Base", "SUS L1", "SUS L2", "SUS L3", "SUS L4"]
    suj_labels: List[str] = ["SUJ Conn", "SUJ Dummy", "SUJ O1", "SUJ O2", "SUJ O3", "SUJ O4"]
    usm_labels: List[str] = ["USM Conn", "OT (RCM)", "O1 (Yaw)", "O2 (Pitch)", "O3 (Rear)", "O4 (Top)", "Carriage", ""]
    txt_font = dict(size=9, color="black")

    # --- 4. Initialize Base Traces (Frame 0) ---
    base_marker = dict(size=14, symbol="square", color="silver", line=dict(color="black", width=2))
    traces: List[go.BaseTraceType] = [
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", marker=base_marker, text=["OB"], name="Robot Base")
    ]

    initial_data: pd.DataFrame = df[df["Frame"] == 0]

    if not initial_data[initial_data["USM"] == 0].empty:
        s_data = initial_data[initial_data["USM"] == 0].iloc[0]
        traces.append(
            go.Scatter3d(x=s_data["SUS_X"], y=s_data["SUS_Y"], z=s_data["SUS_Z"], mode="lines+markers+text",
                         text=sus_labels, textfont=txt_font, line=dict(color="#FF00FF", width=8), name="SUS")
        )

    for usm_idx in ACTIVE_USMS:
        c_suj = arm_colors.get(usm_idx, {"suj": "black"})["suj"]
        c_usm = arm_colors.get(usm_idx, {"usm": "grey"})["usm"]
        arm_rows = initial_data[initial_data["USM"] == usm_idx]

        if not arm_rows.empty:
            arm = arm_rows.iloc[0]
            role = str(arm.get("Role", ""))

            has_tip = pd.notna(arm["True_Tip_X"]) and usm_idx != 3
            tip_x = [arm["True_Tip_X"]] if has_tip else []
            tip_y = [arm["True_Tip_Y"]] if has_tip else []
            tip_z = [arm["True_Tip_Z"]] if has_tip else []

            axis_x = arm["Axis_X"] if has_tip else [np.nan] * 3
            axis_y = arm["Axis_Y"] if has_tip else [np.nan] * 3
            axis_z = arm["Axis_Z"] if has_tip else [np.nan] * 3

            x_end = [tip_x[0] + axis_x[0], tip_y[0] + axis_x[1], tip_z[0] + axis_x[2]] if has_tip else []
            y_end = [tip_x[0] + axis_y[0], tip_y[0] + axis_y[1], tip_z[0] + axis_y[2]] if has_tip else []
            z_end = [tip_x[0] + axis_z[0], tip_y[0] + axis_z[1], tip_z[0] + axis_z[2]] if has_tip else []

            d_x = [arm["Approx_Tip_X"], arm["True_Tip_X"]] if has_tip else []
            d_y = [arm["Approx_Tip_Y"], arm["True_Tip_Y"]] if has_tip else []
            d_z = [arm["Approx_Tip_Z"], arm["True_Tip_Z"]] if has_tip else []

            lbls = usm_labels if usm_idx != 3 else [""] * len(usm_labels)

            traces.extend([
                go.Scatter3d(x=arm["SUJ_X"], y=arm["SUJ_Y"], z=arm["SUJ_Z"], mode="lines+markers+text", text=suj_labels,
                             textposition="bottom center", textfont=txt_font, line=dict(color=c_suj, width=6),
                             name=f"SUJ {usm_idx}"),
                go.Scatter3d(x=arm["USM_X"], y=arm["USM_Y"], z=arm["USM_Z"], mode="lines+markers+text", text=lbls,
                             textposition="top center", textfont=txt_font, line=dict(color=c_usm, width=4),
                             name=f"USM {usm_idx} ({role})"),
                go.Scatter3d(x=tip_x, y=tip_y, z=tip_z, mode="markers", name=f"Tip {usm_idx}",
                             marker=dict(size=6, symbol="diamond", color="black", line=dict(color=c_usm, width=2))),
                go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(width=3, color=c_usm), showlegend=False),
                go.Scatter3d(x=d_x, y=d_y, z=d_z, mode="lines", line=dict(width=2, color="gray", dash="dash"),
                             showlegend=False),
                go.Scatter3d(x=[tip_x[0], x_end[0]] if has_tip else [],
                             y=[tip_y[0], x_end[1]] if has_tip else [],
                             z=[tip_z[0], x_end[2]] if has_tip else [],
                             mode="lines", line=dict(color="red", width=4), showlegend=False),
                go.Scatter3d(x=[tip_x[0], y_end[0]] if has_tip else [],
                             y=[tip_y[0], y_end[1]] if has_tip else [],
                             z=[tip_z[0], y_end[2]] if has_tip else [],
                             mode="lines", line=dict(color="green", width=4), showlegend=False),
                go.Scatter3d(x=[tip_x[0], z_end[0]] if has_tip else [],
                             y=[tip_y[0], z_end[1]] if has_tip else [],
                             z=[tip_z[0], z_end[2]] if has_tip else [],
                             mode="lines", line=dict(color="blue", width=4), showlegend=False)
            ])

    # --- 5. Construct Animation Frames ---
    frames: List[go.Frame] = []
    for frame_idx in range(len(downsampled_times)):
        current_df = df[df["Frame"] == frame_idx]
        history_df = df[df["Frame"] <= frame_idx]

        frame_data: List[go.BaseTraceType] = [go.Scatter3d(x=[0], y=[0], z=[0])]

        if not current_df[current_df["USM"] == 0].empty:
            s_data = current_df[current_df["USM"] == 0].iloc[0]
            frame_data.append(go.Scatter3d(x=s_data["SUS_X"], y=s_data["SUS_Y"], z=s_data["SUS_Z"]))

        for usm_idx in ACTIVE_USMS:
            curr_arm = current_df[current_df["USM"] == usm_idx]
            hist_arm = history_df[history_df["USM"] == usm_idx].dropna(subset=["True_Tip_X"])

            if not curr_arm.empty:
                arm = curr_arm.iloc[0]
                has_tip = pd.notna(arm["True_Tip_X"]) and usm_idx != 3

                tip_x = [arm["True_Tip_X"]] if has_tip else []
                tip_y = [arm["True_Tip_Y"]] if has_tip else []
                tip_z = [arm["True_Tip_Z"]] if has_tip else []

                axis_x = arm["Axis_X"] if has_tip else [np.nan] * 3
                axis_y = arm["Axis_Y"] if has_tip else [np.nan] * 3
                axis_z = arm["Axis_Z"] if has_tip else [np.nan] * 3

                x_end = [tip_x[0] + axis_x[0], tip_y[0] + axis_x[1], tip_z[0] + axis_x[2]] if has_tip else []
                y_end = [tip_x[0] + axis_y[0], tip_y[0] + axis_y[1], tip_z[0] + axis_y[2]] if has_tip else []
                z_end = [tip_x[0] + axis_z[0], tip_y[0] + axis_z[1], tip_z[0] + axis_z[2]] if has_tip else []

                d_x = [arm["Approx_Tip_X"], arm["True_Tip_X"]] if has_tip else []
                d_y = [arm["Approx_Tip_Y"], arm["True_Tip_Y"]] if has_tip else []
                d_z = [arm["Approx_Tip_Z"], arm["True_Tip_Z"]] if has_tip else []

                hist_x = hist_arm["True_Tip_X"].tolist() if usm_idx != 3 else []
                hist_y = hist_arm["True_Tip_Y"].tolist() if usm_idx != 3 else []
                hist_z = hist_arm["True_Tip_Z"].tolist() if usm_idx != 3 else []

                frame_data.extend([
                    go.Scatter3d(x=arm["SUJ_X"], y=arm["SUJ_Y"], z=arm["SUJ_Z"]),
                    go.Scatter3d(x=arm["USM_X"], y=arm["USM_Y"], z=arm["USM_Z"]),
                    go.Scatter3d(x=tip_x, y=tip_y, z=tip_z),
                    go.Scatter3d(x=hist_x, y=hist_y, z=hist_z),
                    go.Scatter3d(x=d_x, y=d_y, z=d_z),
                    go.Scatter3d(x=[tip_x[0], x_end[0]] if has_tip else [],
                                 y=[tip_y[0], x_end[1]] if has_tip else [],
                                 z=[tip_z[0], x_end[2]] if has_tip else []),
                    go.Scatter3d(x=[tip_x[0], y_end[0]] if has_tip else [],
                                 y=[tip_y[0], y_end[1]] if has_tip else [],
                                 z=[tip_z[0], y_end[2]] if has_tip else []),
                    go.Scatter3d(x=[tip_x[0], z_end[0]] if has_tip else [],
                                 y=[tip_y[0], z_end[1]] if has_tip else [],
                                 z=[tip_z[0], z_end[2]] if has_tip else [])
                ])
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # --- 6. Unrolled UI Layout Configuration ---
    x_axis = dict(range=[x_min, x_max])
    y_axis = dict(range=[y_min, y_max])
    z_axis = dict(range=[z_min, z_max])
    scene_dict = dict(xaxis=x_axis, yaxis=y_axis, zaxis=z_axis, aspectmode="data", bgcolor="white")

    # Play Button Setup
    play_frame = dict(duration=80, redraw=True)
    play_transition = dict(duration=0, easing="linear")
    play_args = [None, dict(frame=play_frame, fromcurrent=True, transition=play_transition)]
    play_button = dict(label="Play", method="animate", args=play_args)

    # Pause Button Setup
    pause_frame = dict(duration=0, redraw=False)
    pause_transition = dict(duration=0)
    pause_args = [[None], dict(frame=pause_frame, mode="immediate", transition=pause_transition)]
    pause_button = dict(label="Pause", method="animate", args=pause_args)

    # Update Menus Collection
    update_menus_list = [dict(type="buttons", showactive=False, x=0.1, y=0, xanchor="right", yanchor="top",
                              buttons=[play_button, pause_button])]

    # Slider Loop Setup
    slider_steps: List[Dict[str, Any]] = []
    for f in range(len(downsampled_times)):
        step_frame = dict(duration=0, redraw=True)
        step_transition = dict(duration=0)
        step_args = [[str(f)], dict(mode="immediate", frame=step_frame, transition=step_transition)]
        step_dict = dict(method="animate", args=step_args, label=f"T-{f}")
        slider_steps.append(step_dict)

    sliders_list = [dict(steps=slider_steps, active=0, x=0.15, y=0, xanchor="left", yanchor="top")]

    # --- 7. Compile and Export ---
    fig: go.Figure = go.Figure(data=traces, frames=frames)
    fig.update_layout(title="da Vinci Xi Full Skeleton + Joint Topology", scene=scene_dict,
                      paper_bgcolor="white", font=dict(color="black"),
                      updatemenus=update_menus_list, sliders=sliders_list)

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Done! Open {output_html}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    raw_data_by_timestamp: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    # 1. Ingest Data
    with open(INPUT_CSV, "r", encoding="utf-8") as infile:
        for csv_row in csv.DictReader(infile):
            raw_data_by_timestamp[csv_row["TimeStamp"]].append(csv_row)

    # 2. Derive Kinematic Constants & API Scale Factor
    optimized_parameters: np.ndarray = optimize_kinematics(
        data_by_timestamp=raw_data_by_timestamp,
        target_psm_indices=(0, 2),
        ecm_idx=ECM_USM_IDX
    )

    # 3. Compute Timeline and Generate Output Matrices
    with open(INTERMEDIATE_CSV, "w", newline="", encoding="utf-8") as outfile:
        csv_headers: List[str] = [
            "TimeStamp", "USM", "Role", "Base_Position_Matrix", "SUS_Links", "SUJ_Links", "USM_Links",
            "True_Tip_X", "True_Tip_Y", "True_Tip_Z", "True_Tip_Matrix"
        ]
        csv_writer = csv.DictWriter(outfile, fieldnames=csv_headers)
        csv_writer.writeheader()

        process_and_log_kinematics(
            data_by_timestamp=raw_data_by_timestamp,
            writer=csv_writer,
            opt_params=optimized_parameters
        )

    # 4. Render 3D Animation
    create_combined_animation(INTERMEDIATE_CSV, OUTPUT_HTML)
