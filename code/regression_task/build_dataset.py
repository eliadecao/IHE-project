from pathlib import Path
import re

import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent
# Paths
DVST_DIR = project_root / "Kinematics/dataset/Griffin_Training_Dataset/DVST_XI"
LABEL_DIR = project_root / "Kinematics/dataset/Griffin_Training_Dataset/M-GEARS"
OUTPUT_PATH = project_root / "data/processed/dataset.csv"

TARGET_COLUMN = "Total operation M-GEARS score"


# Raw DVST settings
VECTOR_COLUMNS = {
    "SUSvalues": "sus",
    "SUJvalues": "suj",
    "USMJointValues": "usm_joint",
    "EndoscopePosition": "endoscope",
}


# Helpers
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


def split_vector_column(series: pd.Series, prefix: str):
    split_df = series.astype(str).str.strip().str.split(r"\s+", expand=True)
    split_df = split_df.apply(pd.to_numeric, errors="coerce")
    split_df.columns = [f"{prefix}_{i}" for i in range(split_df.shape[1])]
    return split_df


# X: feature extraction
def load_and_parse_dvst_file(file_path: Path):
    raw = pd.read_csv(file_path)

    parts = []

    if "TimeStamp" not in raw.columns:
        raise ValueError(f"Missing TimeStamp column in {file_path.name}")

    parts.append(pd.to_numeric(raw["TimeStamp"], errors="coerce").rename("timestamp"))

    if "USM" in raw.columns:
        parts.append(pd.to_numeric(raw["USM"], errors="coerce").rename("usm"))

    for col, prefix in VECTOR_COLUMNS.items():
        if col not in raw.columns:
            print(f"Warning: {file_path.name} missing column {col}")
            continue

        parts.append(split_vector_column(raw[col], prefix))

    return pd.concat(parts, axis=1)


def extract_trial_features(df: pd.DataFrame):
    features = {}

    # time features
    timestamp = pd.to_numeric(df["timestamp"], errors="coerce").dropna()

    features["n_timesteps"] = len(timestamp)

    if len(timestamp) >= 2:
        duration_sec = (timestamp.iloc[-1] - timestamp.iloc[0]) / 1000.0
        dt = timestamp.diff().dropna() / 1000.0

        features["duration_seconds"] = duration_sec
        features["mean_dt_seconds"] = dt.mean()
        features["std_dt_seconds"] = dt.std()
        features["min_dt_seconds"] = dt.min()
        features["max_dt_seconds"] = dt.max()
    else:
        features["duration_seconds"] = np.nan
        features["mean_dt_seconds"] = np.nan
        features["std_dt_seconds"] = np.nan
        features["min_dt_seconds"] = np.nan
        features["max_dt_seconds"] = np.nan

    # signal features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    signal_cols = [c for c in numeric_cols if c != "timestamp"]

    stat_functions = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "min": lambda x: x.min(),
        "max": lambda x: x.max(),
        "median": lambda x: x.median(),
        "range": lambda x: x.max() - x.min(),
        "q25": lambda x: x.quantile(0.25),
        "q75": lambda x: x.quantile(0.75),
    }

    for col in signal_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            for stat_name in stat_functions:
                features[f"{col}_{stat_name}"] = np.nan
        else:
            for stat_name, func in stat_functions.items():
                features[f"{col}_{stat_name}"] = func(values)

        # simple motion/change features
        diff = values.diff().dropna()

        if len(diff) == 0:
            features[f"{col}_diff_mean_abs"] = np.nan
            features[f"{col}_diff_std"] = np.nan
            features[f"{col}_diff_max_abs"] = np.nan
        else:
            features[f"{col}_diff_mean_abs"] = diff.abs().mean()
            features[f"{col}_diff_std"] = diff.std()
            features[f"{col}_diff_max_abs"] = diff.abs().max()

    return features


def build_X_from_dvst_folder(dvst_dir: Path):
    csv_files = sorted(dvst_dir.rglob("*.csv"))

    rows = []

    print(f"Found {len(csv_files)} DVST files.")

    for i, file_path in enumerate(csv_files, start=1):
        print(f"[X {i}/{len(csv_files)}] {file_path.name}")

        try:
            metadata = extract_metadata_from_dvst_filename(file_path)
            parsed_df = load_and_parse_dvst_file(file_path)
            features = extract_trial_features(parsed_df)

            row = {}
            row.update(metadata)
            row.update(features)

            rows.append(row)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    X_df = pd.DataFrame(rows)

    print("\nX shape:", X_df.shape)
    return X_df


# y: label extraction
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

    # avoid duplicate labels
    y_df = y_df.drop_duplicates(
        subset=["role", "participant_id", "task_clean", "trial"],
        keep="first",
    )

    print("\ny shape:", y_df.shape)
    return y_df


# Merge X and y
def build_dataset():
    X_df = build_X_from_dvst_folder(DVST_DIR)
    y_df = build_y_from_label_folder(LABEL_DIR)

    merge_keys = ["role", "participant_id", "task_clean", "trial"]

    print("\nX key examples:")
    print(X_df[merge_keys].head())

    print("\ny key examples:")
    print(y_df[merge_keys].head())

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")

    return dataset


if __name__ == "__main__":
    build_dataset()
