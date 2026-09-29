import pandas as pd
from pathlib import Path

# ===== paths =====
INPUT = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\dataset_v7_fk_camera.csv"
)

OUTPUT = Path(
    r"C:\Users\ROG\IHE-project\data\v7_fk_camera\dataset_v7_fk_camera_clean.csv"
)

# ===== confirmed bad recordings =====
BAD_FILES = [
    "DVST_XI_Griffin_16-19_07_2024_16_08.46.09_trainee_12_suture_1.csv",
    "DVST_XI_6-8_Mar_2025_07_11.56.53_expert_2_suture_1.csv",
    "DVST_XI_6-8_Mar_2025_08_08.21.32_expert_3_glove cut_1.csv",
    "DVST_XI_Griffin_16-19_07_2024_18_08.46.52_trainee_12_sea spikes_2.csv",
    "DVST_XI_7-11_Mar_2024_09_10.19.01_trainee_10_camera target_1.csv",
    "DVST_XI_Griffin_16-19_07_2024_17_12.20.23_trainee_13_ring rollercoaster_2.csv",
    "DVST_XI_Griffin_29_sep-1_oct_2024_29_09.07.40_trainee_18_camera target_1.csv",
    "DVST_XI_14-16_Feb_2024_14_09.13.16_trainee_2_suture_1.csv",
]

df = pd.read_csv(INPUT)

mask = df["file_name"].isin(BAD_FILES)

print("Before:", len(df))
print("Rows found for removal:", mask.sum())

print("\nRemoved recordings:")
print(
    df.loc[
        mask,
        ["file_name", "task", "qc_fk_duration_seconds"]
    ].to_string(index=False)
)

clean = df.loc[~mask].copy()

clean.to_csv(OUTPUT, index=False)

print("\nAfter:", len(clean))
print("Saved to:")
print(OUTPUT)
