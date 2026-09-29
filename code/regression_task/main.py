"""Unified entry point for the existing v7 FK + camera regression pipeline.

Place beside build_dataset_with_camera.py, or supply --project-root.
Examples: python main.py --dry-run
          python main.py --until folds
          python main.py --from-step baseline
          python main.py --from-step build --until folds

The implementation reuses the reviewed stage modules; it is not a standalone
copy of their modelling algorithms. Existing result files may be overwritten.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import time

STEPS = ("build", "clean", "qc", "prepare", "folds", "baseline", "ablation",
         "nonlinear", "strict", "compare", "domain")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/data/v7_fk_camera")
    parser.add_argument("--dataset", type=Path, help="Existing raw v7 CSV, used when skipping build")
    parser.add_argument("--fk-dir", type=Path, help="Default: PROJECT/data/updated_fk")
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--from-step", choices=STEPS, default="clean")
    parser.add_argument("--until", choices=STEPS, default="domain")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="CLI model stages only; legacy domain stage uses all cores")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without imports or writes")
    args = parser.parse_args()
    if STEPS.index(args.from_step) > STEPS.index(args.until):
        parser.error("--from-step must precede --until")
    for key in ("inner_splits", "n_iter", "bootstrap_iterations"):
        if getattr(args, key) < (2 if key == "inner_splits" else 1):
            parser.error(f"Invalid --{key.replace('_', '-')}")
    if args.n_splits < 3 or args.n_jobs == 0:
        parser.error("--n-splits must be >= 3 and --n-jobs must not be zero")
    if args.dataset and args.from_step == "build":
        parser.error("--dataset cannot be combined with --from-step build")
    return args


def project_root(explicit):
    if explicit:
        root = explicit.resolve()
    else:
        root = next((p for p in Path(__file__).resolve().parents
                     if (p / "code/regression_task/build_dataset_with_camera.py").is_file()), None)
    if root is None or not (root / "code/regression_task").is_dir():
        raise ValueError("Cannot locate project. Supply --project-root PATH.")
    return root


def clean_dataset(source, destination, scripts):
    """Read the existing bad-recording list without executing its top-level I/O."""
    import pandas as pd
    tree = ast.parse((scripts / "clean_dataset.py").read_text(encoding="utf-8-sig"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "BAD_FILES" for t in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected one literal BAD_FILES list in clean_dataset.py")
    bad_files = ast.literal_eval(assignments[0].value)
    if not isinstance(bad_files, list) or not all(isinstance(x, str) for x in bad_files):
        raise ValueError("BAD_FILES must be a list of file names")
    df = pd.read_csv(source)
    if "file_name" not in df:
        raise ValueError("Dataset is missing file_name")
    mask = df["file_name"].isin(bad_files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.loc[mask].to_csv(destination.parent / "removed_bad_recordings.csv", index=False)
    df.loc[~mask].to_csv(destination, index=False)
    print(f"Clean: {len(df)} -> {int((~mask).sum())} rows", flush=True)


def run_module(name, arguments):
    previous = sys.argv
    try:
        sys.argv = [name + ".py", *map(str, arguments)]
        importlib.import_module(name).main()
    finally:
        sys.argv = previous


def run_domain(folds, output, args):
    import pandas as pd
    module = importlib.import_module("run_mgears_domain_regression")
    # Re-read this run's benchmark instead of using the legacy hard-coded values.
    metrics = pd.read_csv(folds.parent / "nonlinear_regression/nonlinear_overall_metrics.csv")
    benchmark = metrics.loc[metrics["model"].eq("extra_trees")]
    if len(benchmark) != 1:
        raise ValueError("Expected one extra_trees benchmark row")
    for metric in ("mae", "rmse", "r2"):
        setattr(module, "DIRECT_TOTAL_BENCHMARK_" + metric.upper(), float(benchmark.iloc[0][metric]))
    module.INPUT_CSV = folds
    module.OUTPUT_DIR = output
    module.INNER_CV_SPLITS = args.inner_splits
    module.RANDOM_SEARCH_ITER = args.n_iter
    module.RANDOM_STATE = args.random_state
    module.main()


def main():
    args = parse_args()
    root = project_root(args.project_root)
    scripts = root / "code/regression_task"
    output = (args.output_dir or root / "data/v7_fk_camera").resolve()
    raw = (args.dataset or output / "dataset_v7_fk_camera.csv").resolve()
    clean = output / "dataset_v7_fk_camera_clean.csv"
    qc = output / "qc_v7"
    prepared = qc / "v4"
    grouped = prepared / "grouped_cv"
    folds = grouped / "dataset_v4_grouped_folds.csv"
    fk = (args.fk_dir or root / "data/updated_fk").resolve()
    labels = (args.label_dir or root / "Kinematics/corrected/Griffin_Training_Dataset/M-GEARS").resolve()
    common = ["--input", folds, "--inner-splits", args.inner_splits, "--n-jobs", args.n_jobs]
    nonlinear = [*common, "--n-iter", args.n_iter, "--random-state", args.random_state]
    # name: (module, arguments, required inputs, primary output)
    plan = {
        "build": ("build_dataset_with_camera", ["--fk-dir", fk, "--label-dir", labels,
                  "--output-dir", output, "--v3-script", scripts / "dataset_v3.py"],
                  [fk, labels, scripts / "dataset_v3.py"], raw),
        "clean": ("clean_dataset", [], [raw], clean),
        "qc": ("qc_dataset_v3", ["--input", clean, "--output-dir", qc], [clean], qc / "dataset_v7_model.csv"),
        "prepare": ("dataset_v4", ["--input", qc / "dataset_v7_model.csv", "--output-dir", prepared],
                    [qc / "dataset_v7_model.csv"], prepared / "dataset_v4_prefiltered.csv"),
        "folds": ("create_group_fold", ["--input", prepared / "dataset_v4_prefiltered.csv",
                  "--output-dir", grouped, "--n-splits", args.n_splits],
                  [prepared / "dataset_v4_prefiltered.csv"], folds),
        "baseline": ("run_grouped_baseline", [*common, "--output-dir", grouped / "baseline_regression"],
                     [folds], grouped / "baseline_regression/overall_oof_metrics.csv"),
        "ablation": ("run_ablation_analysis", [*common, "--output-dir", grouped / "ablation_analysis"],
                     [folds], grouped / "ablation_analysis/ablation_oof_predictions.csv"),
        "nonlinear": ("run_nonlinear_regression", [*nonlinear, "--output-dir", grouped / "nonlinear_regression"],
                      [folds], grouped / "nonlinear_regression/nonlinear_oof_predictions.csv"),
        "strict": ("run_nonlinear_regression_strict_qc", [*nonlinear, "--output-dir", grouped / "nonlinear_regression_strict_qc"],
                   [folds], grouped / "nonlinear_regression_strict_qc/nonlinear_oof_predictions.csv"),
        "compare": ("compare_final_models", ["--grouped-dir", grouped, "--output-dir", grouped / "final_model_comparison",
                    "--bootstrap-iterations", args.bootstrap_iterations, "--random-state", args.random_state],
                    [grouped / "nonlinear_regression/nonlinear_oof_predictions.csv",
                     grouped / "ablation_analysis/ablation_oof_predictions.csv"], grouped / "final_model_comparison/final_operation_level_metrics.csv"),
        "domain": ("run_mgears_domain_regression", [],
                   [folds, grouped / "nonlinear_regression/nonlinear_overall_metrics.csv"], grouped / "mgears_domain_regression/domain_oof_predictions.csv"),
    }
    selected = STEPS[STEPS.index(args.from_step):STEPS.index(args.until) + 1]
    for step in selected:
        module, argv, inputs, result = plan[step]
        if not (scripts / (module + ".py")).is_file():
            raise FileNotFoundError(scripts / (module + ".py"))
        print(f"{step}: {module}.py\n  inputs: {', '.join(map(str, inputs))}\n  output: {result}", flush=True)
    if args.dry_run:
        return
    sys.path.insert(0, str(scripts))
    sys.dont_write_bytecode = True
    os.environ.setdefault("MPLBACKEND", "Agg")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"pipeline_run_{time.time_ns()}.json"
    manifest = {"pipeline": "v7_fk_camera", "python": sys.executable,
                "project_root": str(root), "arguments": vars(args).copy(), "steps": []}
    for step in selected:
        module, argv, inputs, result = plan[step]
        record = {"step": step, "module": module, "arguments": list(map(str, argv)),
                  "status": "running", "started": time.time(), "output": str(result)}
        manifest["steps"].append(record)
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
            for source in inputs:
                if not source.exists():
                    raise FileNotFoundError(f"{step}: missing {source}; run the upstream step first")
            if step == "clean":
                clean_dataset(raw, clean, scripts)
            elif step == "domain":
                run_domain(folds, result.parent, args)
            else:
                run_module(module, argv)
            if not result.exists():
                raise RuntimeError(f"{step} did not produce {result}")
            record["status"] = "completed"
        except BaseException as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            record["seconds"] = round(time.time() - record["started"], 3)
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Pipeline completed. Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
