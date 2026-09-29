from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET = Path(
    r"C:\Users\ROG\IHE-project\data\v5_fk_corrected\qc_v5\v4"
    r"\dataset_v4_prefiltered_camera_metrics.csv"
)

FK_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\fk"
)


# ============================================================
# HELPERS
# ============================================================

def canonical_name(name: str) -> str:

    name = Path(str(name)).stem.lower()

    for token in [
        "_fk_results",
        "_fk",
        "dvst_xi_",
    ]:
        name = name.replace(token, "")

    return (
        name.replace("-", "_")
            .replace(" ", "_")
            .strip("_")
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATASET)

print("\n" + "=" * 90)
print("CAMERA METRICS FULL QC")
print("=" * 90)

print("Dataset shape:", df.shape)


# ============================================================
# 1. MATCHING QC
# ============================================================

print("\n" + "=" * 90)
print("1. CAMERA METRIC MATCHING")
print("=" * 90)

matched = df["camera_path_length_m"].notna()

print(
    f"Matched:   {matched.sum()} / {len(df)}"
)

print(
    f"Unmatched: {(~matched).sum()} / {len(df)}"
)

name_cols = [
    c for c in [
        "file_name",
        "fk_file_name",
        "task_clean",
        "task",
        "role",
        "participant_id",
        "trial",
    ]
    if c in df.columns
]

if (~matched).sum() > 0:

    print("\nUNMATCHED OPERATIONS:")

    print(
        df.loc[
            ~matched,
            name_cols
        ].to_string(index=False)
    )


# ============================================================
# 2. CAMERA METRIC DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 90)
print("2. CAMERA METRIC DISTRIBUTIONS")
print("=" * 90)

metric_cols = [
    "camera_path_length_m",
    "camera_move_count",
    "camera_moving_duration_s",
    "camera_to_USM0_mean_distance_m",
    "camera_to_USM2_mean_distance_m",
    "camera_to_USM3_mean_distance_m",
    "camera_to_tools_mean_distance_m",
]

for col in metric_cols:

    if col not in df.columns:
        continue

    x = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if len(x) == 0:
        continue

    print(f"\n{col}")

    print(
        f"  N      = {len(x)}"
    )

    print(
        f"  min    = {x.min():.6f}"
    )

    print(
        f"  median = {x.median():.6f}"
    )

    print(
        f"  mean   = {x.mean():.6f}"
    )

    print(
        f"  p75    = {x.quantile(0.75):.6f}"
    )

    print(
        f"  p90    = {x.quantile(0.90):.6f}"
    )

    print(
        f"  p95    = {x.quantile(0.95):.6f}"
    )

    print(
        f"  p99    = {x.quantile(0.99):.6f}"
    )

    print(
        f"  max    = {x.max():.6f}"
    )


# ============================================================
# 3. TOP OUTLIER OPERATIONS
# ============================================================

print("\n" + "=" * 90)
print("3. TOP CAMERA PATH / DISTANCE OUTLIERS")
print("=" * 90)

show_cols = name_cols + [
    c for c in metric_cols
    if c in df.columns
]

for metric in [
    "camera_path_length_m",
    "camera_to_USM0_mean_distance_m",
    "camera_to_USM2_mean_distance_m",
    "camera_to_USM3_mean_distance_m",
]:

    if metric not in df.columns:
        continue

    print(f"\nTOP 10 BY {metric}:")

    print(
        df[
            show_cols
        ]
        .sort_values(
            metric,
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 4. RAW FK TIMESTAMP / CAMERA MOTION QC
# ============================================================

print("\n" + "=" * 90)
print("4. RAW FK TIMESTAMP / CAMERA MOTION QC")
print("=" * 90)

fk_files = list(
    FK_DIR.rglob("*.csv")
)

print(
    "FK files found:",
    len(fk_files)
)

# inspect first 5 usable FK files only
checked = 0

for path in fk_files:

    if checked >= 5:
        break

    try:

        fk = pd.read_csv(path)

        required = {
            "TimeStamp",
            "USM",
            "True_Tip_X",
            "True_Tip_Y",
            "True_Tip_Z",
        }

        if not required.issubset(
            fk.columns
        ):
            continue

        camera = fk[
            fk["USM"] == 1
        ].copy()

        if "Role" in camera.columns:
            role_mask = (
                camera["Role"]
                .astype(str)
                .eq("ECM_Camera")
            )

            if role_mask.any():
                camera = camera[
                    role_mask
                ]

        for col in [
            "TimeStamp",
            "True_Tip_X",
            "True_Tip_Y",
            "True_Tip_Z",
        ]:

            camera[col] = pd.to_numeric(
                camera[col],
                errors="coerce"
            )

        camera = (
            camera
            .dropna(
                subset=[
                    "TimeStamp",
                    "True_Tip_X",
                    "True_Tip_Y",
                    "True_Tip_Z",
                ]
            )
            .sort_values(
                "TimeStamp"
            )
        )

        if len(camera) < 3:
            continue

        times = camera[
            "TimeStamp"
        ].to_numpy(float)

        xyz = camera[
            [
                "True_Tip_X",
                "True_Tip_Y",
                "True_Tip_Z",
            ]
        ].to_numpy(float)

        dt = np.diff(times)

        step = np.linalg.norm(
            np.diff(
                xyz,
                axis=0
            ),
            axis=1
        )

        valid = (
            np.isfinite(dt)
            & np.isfinite(step)
            & (dt > 0)
        )

        if valid.sum() == 0:
            continue

        raw_speed = (
            step[valid]
            / dt[valid]
        )

        total_path = float(
            np.sum(
                step[valid]
            )
        )

        print(
            "\nFILE:",
            path.name
        )

        print(
            f"  camera samples      = {len(camera)}"
        )

        print(
            f"  first timestamp     = {times[0]}"
        )

        print(
            f"  last timestamp      = {times[-1]}"
        )

        print(
            f"  median dt           = "
            f"{np.median(dt[valid]):.8f}"
        )

        print(
            f"  mean dt             = "
            f"{np.mean(dt[valid]):.8f}"
        )

        print(
            f"  min dt              = "
            f"{np.min(dt[valid]):.8f}"
        )

        print(
            f"  max dt              = "
            f"{np.max(dt[valid]):.8f}"
        )

        print(
            f"  median step (m)     = "
            f"{np.median(step[valid]):.8f}"
        )

        print(
            f"  p95 step (m)        = "
            f"{np.percentile(step[valid], 95):.8f}"
        )

        print(
            f"  max step (m)        = "
            f"{np.max(step[valid]):.8f}"
        )

        print(
            f"  total path (m)      = "
            f"{total_path:.6f}"
        )

        print(
            f"  median step/dt      = "
            f"{np.median(raw_speed):.10f}"
        )

        print(
            f"  p95 step/dt         = "
            f"{np.percentile(raw_speed, 95):.10f}"
        )

        print(
            f"  max step/dt         = "
            f"{np.max(raw_speed):.10f}"
        )

        checked += 1

    except Exception:
        continue


# ============================================================
# 5. QUICK CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 90)
print("5. QUICK CONSISTENCY CHECK")
print("=" * 90)

if (
    "camera_move_count" in df.columns
    and
    "camera_path_length_m" in df.columns
):

    moving_path = df[
        df["camera_path_length_m"]
        > 0.01
    ]

    zero_moves = moving_path[
        moving_path[
            "camera_move_count"
        ] == 0
    ]

    print(
        "Operations with camera path > 1 cm:",
        len(moving_path)
    )

    print(
        "Of those, camera_move_count == 0:",
        len(zero_moves)
    )

    if len(moving_path) > 0:

        print(
            "Fraction suspicious:",
            f"{len(zero_moves) / len(moving_path):.1%}"
        )


print("\n" + "=" * 90)
print("QC COMPLETE")
print("=" * 90)
