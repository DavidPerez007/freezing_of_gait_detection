#!/usr/bin/env python3
"""
Notebook Orchestrator — Daphnet + Figshare
==========================================
Executes all notebooks sequentially in the correct order.
First Daphnet (01→04 + metamodel), then Figshare (01→04 + metamodel).

Usage:
    python scripts/run_all_notebooks.py              # run all
    python scripts/run_all_notebooks.py --daphnet    # only Daphnet
    python scripts/run_all_notebooks.py --figshare   # only Figshare
    python scripts/run_all_notebooks.py --timeout 3600  # custom timeout per notebook (seconds)
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAPHNET_NOTEBOOKS = [
    "notebooks/daphnet_dataset/01_exploratory_data_analysis.ipynb",
    "notebooks/daphnet_dataset/02_preprocessing_and_windowing.ipynb",
    "notebooks/daphnet_dataset/03_feature_extraction.ipynb",
    "notebooks/daphnet_dataset/04_loso_pipeline_and_training.ipynb",
    "notebooks/daphnet_dataset/test_metamodel.ipynb",
]

FIGSHARE_NOTEBOOKS = [
    "notebooks/figshare_dataset/01_exploratory_data_analysis.ipynb",
    "notebooks/figshare_dataset/02_preprocessing_and_windowing.ipynb",
    "notebooks/figshare_dataset/03_feature_extraction.ipynb",
    "notebooks/figshare_dataset/04_loso_pipeline_and_training.ipynb",
    "notebooks/figshare_dataset/test_metamodel.ipynb",
]


def run_notebook(nb_path: str, timeout: int = 7200) -> bool:
    """Execute a single notebook in-place using nbconvert."""
    full_path = PROJECT_ROOT / nb_path
    if not full_path.exists():
        print(f"  SKIP — {nb_path} not found")
        return False

    # Execute notebook from its own directory so relative paths work
    nb_dir = full_path.parent
    nb_name = full_path.name

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout", str(timeout),
        "--ExecutePreprocessor.kernel_name", "python3",
        nb_name,
    ]

    print(f"  Running: {nb_path} ...", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(nb_dir),
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # extra margin
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  OK — {nb_path} ({elapsed:.0f}s)")
            return True
        else:
            print(f"  FAILED — {nb_path} ({elapsed:.0f}s)")
            print(f"    stderr: {result.stderr[-500:]}" if result.stderr else "")
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT — {nb_path} ({elapsed:.0f}s, limit={timeout}s)")
        return False
    except Exception as e:
        print(f"  ERROR — {nb_path}: {e}")
        return False


def run_pipeline(name: str, notebooks: list, timeout: int):
    """Run a list of notebooks sequentially."""
    print(f"\n{'='*60}")
    print(f"  {name} PIPELINE")
    print(f"{'='*60}")

    results = {}
    t0 = time.time()

    for nb in notebooks:
        success = run_notebook(nb, timeout=timeout)
        results[nb] = success
        if not success:
            print(f"\n  Pipeline stopped: {nb} failed.")
            print(f"  Fix the notebook and re-run the script.")
            break

    elapsed = time.time() - t0
    passed = sum(results.values())
    total = len(results)
    print(f"\n  {name} summary: {passed}/{total} notebooks passed ({elapsed:.0f}s total)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Daphnet + Figshare notebooks sequentially")
    parser.add_argument("--daphnet", action="store_true", help="Run only Daphnet notebooks")
    parser.add_argument("--figshare", action="store_true", help="Run only Figshare notebooks")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per notebook in seconds (default: 7200 = 2h)")
    parser.add_argument("--no-stop-on-error", action="store_true", help="Continue even if a notebook fails")
    args = parser.parse_args()

    run_daphnet = args.daphnet or (not args.daphnet and not args.figshare)
    run_figshare = args.figshare or (not args.daphnet and not args.figshare)

    print("=" * 60)
    print("  NOTEBOOK ORCHESTRATOR")
    print("=" * 60)
    print(f"  Timeout per notebook: {args.timeout}s")
    print(f"  Daphnet: {'yes' if run_daphnet else 'no'}")
    print(f"  Figshare: {'yes' if run_figshare else 'no'}")

    t_global = time.time()
    all_results = {}

    if run_daphnet:
        all_results.update(run_pipeline("DAPHNET", DAPHNET_NOTEBOOKS, args.timeout))

    if run_figshare:
        all_results.update(run_pipeline("FIGSHARE", FIGSHARE_NOTEBOOKS, args.timeout))

    # Final summary
    elapsed = time.time() - t_global
    passed = sum(all_results.values())
    total = len(all_results)

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY: {passed}/{total} notebooks passed ({elapsed/60:.1f} min total)")
    print(f"{'='*60}")
    for nb, ok in all_results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {nb}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
