#!/usr/bin/env python3
"""Run the full Charite improved pipeline (notebooks 07-08-09) in sequence."""
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOKS = [
    "07_improved_pipeline_preprocessing.ipynb",
    "08_improved_pipeline_loso.ipynb",
    "09_improved_pipeline_fusion.ipynb",
]

def run_notebook(nb_path: Path):
    """Execute a notebook in-place using jupyter nbconvert."""
    print(f"\n{'='*70}")
    print(f"  Running: {nb_path.name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=-1",
            "--ExecutePreprocessor.kernel_name=python3",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED after {elapsed/60:.1f} min")
        print(result.stderr)
        sys.exit(1)
    print(f"  Done in {elapsed/60:.1f} min")


def main():
    t_total = time.time()
    print("Charite Improved Pipeline — Automated Execution")
    for nb_name in NOTEBOOKS:
        run_notebook(NOTEBOOK_DIR / nb_name)
    print(f"\nAll notebooks completed in {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
