"""
classify_griffin_3class_v5.py
==============================
3-class GEARS-band skill classification, adapted to the new v5 dataset
(dataset_v5_fk_only.csv).

Same idea as classify_griffin_3class.py: define low/mid/high skill bands
from the GEARS score (target_score, 0-30) and predict the band from
movement metrics only (the score itself is never fed to the model).

*** IMPORTANT DATA ISSUE (same as the 2-class script) ***
`participant_id` is reused independently across roles -- e.g. expert #1
and trainee #1 are different real people (different session_licence /
session_datetime). We group by (participant_id + role) for subject-wise
CV instead of participant_id alone. Worth confirming with supervisors.
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

# ======================================================================
# CONFIG -- how to define the 3 skill bands from the GEARS score
# ======================================================================
BAND_MODE = "fixed"          # "quantile" or "fixed"
FIXED_CUTOFFS = [19, 27]     # clinician-defined: low<=19, mid 20-27, high>=28

CSV = os.path.expanduser('/Users/lauraangelican/Desktop/studentship/dataset_v7_fk_camera_clean.csv')   # <- edit path if needed

# ======================================================================
# 1. LOAD + LABEL
# ======================================================================
df = pd.read_csv(CSV).copy()

if BAND_MODE == "quantile":
    df["band"] = pd.qcut(df["target_score"], 3, labels=["low", "mid", "high"])
    print("Bands = equal-size thirds of the GEARS score.")
else:
    lo, hi = FIXED_CUTOFFS
    df["band"] = pd.cut(df["target_score"], bins=[-1, lo, hi, 100],
                         labels=["low", "mid", "high"])
    print(f"Bands = low(<= {lo})  mid({lo + 1}-{hi})  high(>= {hi + 1}) GEARS score.")

band_order = ["low", "mid", "high"]
df["label"] = df["band"].map({b: i for i, b in enumerate(band_order)})

print("\nTrials per band:")
print(df["band"].value_counts().reindex(band_order).to_string())
print()

df["subject_key"] = df["participant_id"].astype(str) + "_" + df["role"]

# ======================================================================
# 2. FEATURES (movement metrics only -- score / mgears domains excluded)
# ======================================================================
exclude = {
    "file_name", "session_licence_raw", "session_datetime_raw", "session_licence",
    "session_datetime", "role", "participant_id", "task", "task_clean", "trial",
    "recording_id", "fk_file_name", "label", "band", "subject_key",
    "qc_fk_rows_original", "qc_fk_rows", "qc_fk_rows_removed_missing_timestamp_or_usm",
    "qc_fk_duplicate_timestamp_usm_rows_removed", "qc_fk_timestamp_unit_inferred",
    "qc_fk_duration_seconds", "qc_fk_valid_xyz_fraction", "qc_fk_usms_present",
    "qc_fk_roles_present", "fk_instrument_usms", "fk_camera_usms",
    "fk_instrument_usm_count", "fk_camera_usm_count",
    "label_recording_id", "label_source_files", "n_label_records", "target_score",
    "target_score_std", "n_unique_target_scores", "label_disagreement_flag",
    "percentage_score", "percentage_score_std", "target_fraction",
    "mgears_depth_perception", "mgears_depth_perception_std",
    "mgears_dexterity_with_multiple_wristed_instruments",
    "mgears_dexterity_with_multiple_wristed_instruments_std",
    "mgears_efficiency_flow_of_operation", "mgears_efficiency_flow_of_operation_std",
    "mgears_force_sensitivity_and_tissue_handling",
    "mgears_force_sensitivity_and_tissue_handling_std",
    "mgears_autonomy", "mgears_autonomy_std",
    "mgears_master_manipulator_workspace_robotic_control",
    "mgears_master_manipulator_workspace_robotic_control_std",
    "mgears_basic_energy_skills", "mgears_basic_energy_skills_std",
    "mgears_overall_performance_quality_of_the_final_product",
    "mgears_overall_performance_quality_of_the_final_product_std",
    "n_applicable_mgears_domains", "mgears_domain_score_sum",
    "target_minus_domain_sum", "inferred_max_score",
    "active_pair_usm_a", "active_pair_usm_b",
    "USM0_pose_global_active", "USM2_pose_global_active", "USM3_pose_global_active",
    "active_psm_count",
}
drop_suffixes = ("_n_samples", "_n_aligned_samples", "_median_dt_seconds",
                  "_sample_rate_hz", "_max_gap_seconds", "_alignment_coverage")

feature_cols = [c for c in df.columns
                if c not in exclude
                and not c.endswith(drop_suffixes)
                and pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols].values
y = df["label"].values
groups = df["subject_key"].values

print(f"{len(df)} trials, {len(feature_cols)} features, {df['subject_key'].nunique()} subjects\n")

# ======================================================================
# 3. MODEL + SUBJECT-WISE, BAND-STRATIFIED CROSS-VALIDATION
# ======================================================================
def make_model(kind):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=0,
                                      class_weight="balanced")
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("clf", clf)])


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

for kind in ["logreg", "forest"]:
    model = make_model(kind)
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)

    print(f"===== {kind.upper()} =====")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.2f}")
    print(classification_report(y, y_pred, target_names=band_order, zero_division=0))
    print("confusion matrix [rows=true, cols=pred], order [low, mid, high]:")
    print(confusion_matrix(y, y_pred), "\n")