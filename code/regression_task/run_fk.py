from __future__ import annotations

"""
Batch runner for the supervisor-provided da Vinci Xi forward-kinematics optimiser.

For every raw DVST_XI CSV:
    raw CSV -> FK optimisation -> <original_stem>_FK.csv

The expensive Plotly animation is disabled by default. Use --html-first N to
render animations only for the first N successfully processed recordings.

Example:
    python run_fk_batch.py \
        --raw-dir "C:/project/Griffin_Training_Dataset/DVST_XI" \
        --output-dir "C:/project/data/fk_outputs" \
        --supervisor-script "C:/project/forward_kinematics_optimized_scale.py"
"""

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import traceback

import numpy as np


def load_supervisor_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("dvxi_fk_supervisor", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import supervisor script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_grouped_raw_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "TimeStamp", "USM", "SUSvalues", "SUJvalues",
            "USMJointValues", "EndoscopePosition"
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            grouped[row["TimeStamp"]].append(row)
    if not grouped:
        raise ValueError(f"{path.name} contains no rows.")
    return grouped


def process_one(
    raw_path: Path,
    output_csv: Path,
    module,
    *,
    render_html: bool,
    output_html: Path | None,
    target_psms: tuple[int, ...],
    ecm_usm: int,
) -> dict[str, object]:
    grouped = load_grouped_raw_csv(raw_path)

    opt = module.optimize_kinematics(
        data_by_timestamp=grouped,
        target_psm_indices=target_psms,
        ecm_idx=ecm_usm,
    )
    opt = np.asarray(opt, dtype=float)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "TimeStamp", "USM", "Role", "Base_Position_Matrix",
        "SUS_Links", "SUJ_Links", "USM_Links",
        "True_Tip_X", "True_Tip_Y", "True_Tip_Z", "True_Tip_Matrix",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        module.process_and_log_kinematics(
            data_by_timestamp=grouped,
            writer=writer,
            opt_params=opt,
        )

    if render_html:
        if output_html is None:
            raise ValueError("output_html is required when render_html=True")
        module.create_combined_animation(str(output_csv), str(output_html))

    return {
        "raw_file": raw_path.name,
        "fk_file": output_csv.name,
        "scale": float(opt[0]),
        "dx_m": float(opt[1]),
        "dy_m": float(opt[2]),
        "rx_rad": float(opt[3]),
        "ry_rad": float(opt[4]),
        "rz_rad": float(opt[5]),
        "rendered_html": bool(render_html),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supervisor-script", type=Path, required=True)
    parser.add_argument(
        "--target-psms",
        type=int,
        nargs="+",
        default=[0, 2],
        help="PSM indices used by the optimiser; default: 0 2",
    )
    parser.add_argument("--ecm-usm", type=int, default=1)
    parser.add_argument(
        "--html-first",
        type=int,
        default=0,
        help="Render Plotly animations for only the first N successful files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    supervisor_script = args.supervisor_script.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if not supervisor_script.exists():
        raise FileNotFoundError(supervisor_script)

    module = load_supervisor_module(supervisor_script)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    log_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, str]] = []

    raw_files = sorted(raw_dir.rglob("*.csv"))
    if not raw_files:
        raise ValueError(f"No CSV files found under {raw_dir}")

    success_count = 0
    for i, raw_path in enumerate(raw_files, start=1):
        relative = raw_path.relative_to(raw_dir)
        output_csv = output_dir / relative.parent / f"{raw_path.stem}_FK.csv"
        render_html = success_count < args.html_first
        output_html = html_dir / relative.parent / f"{raw_path.stem}_robot_skeleton.html"

        print(f"\n[{i}/{len(raw_files)}] {relative}")
        if output_csv.exists() and not args.overwrite:
            print("  SKIP: FK output already exists.")
            log_rows.append({
                "raw_file": raw_path.name,
                "fk_file": output_csv.name,
                "status": "skipped_existing",
            })
            continue

        try:
            result = process_one(
                raw_path,
                output_csv,
                module,
                render_html=render_html,
                output_html=output_html,
                target_psms=tuple(args.target_psms),
                ecm_usm=args.ecm_usm,
            )
            result["status"] = "success"
            log_rows.append(result)
            success_count += 1
            print(f"  SAVED: {output_csv}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            error_rows.append({
                "raw_file": str(raw_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    import pandas as pd
    pd.DataFrame(log_rows).to_csv(output_dir / "fk_batch_manifest.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output_dir / "fk_batch_errors.csv", index=False)

    summary = {
        "n_raw_files": len(raw_files),
        "n_success": sum(row.get("status") == "success" for row in log_rows),
        "n_skipped": sum(row.get("status") == "skipped_existing" for row in log_rows),
        "n_errors": len(error_rows),
        "ecm_usm": args.ecm_usm,
        "target_psms": args.target_psms,
    }
    (output_dir / "fk_batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nBatch summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    