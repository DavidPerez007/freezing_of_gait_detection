#!/usr/bin/env python3
"""Run the full Charite improved pipeline (notebooks 07-08-09) in sequence."""
import sys
import time
from pathlib import Path

import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOKS = [
    "07_improved_pipeline_preprocessing.ipynb",
    "08_improved_pipeline_loso.ipynb",
    "09_improved_pipeline_fusion.ipynb",
]

def run_notebook(nb_path: Path):
    """Execute a notebook in-place using nbconvert's ExecutePreprocessor."""
    print(f"\n{'='*70}")
    print(f"  Running: {nb_path.name}")
    print(f"{'='*70}")
    t0 = time.time()
    with nb_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    processor = ExecutePreprocessor(timeout=-1, kernel_name="python3")

    elapsed = time.time() - t0
    try:
        processor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    except CellExecutionError as exc:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed/60:.1f} min")
        print(exc)
        sys.exit(1)

    with nb_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")


def main():
    t_total = time.time()
    print("Charite Improved Pipeline — Automated Execution")
    for nb_name in NOTEBOOKS:
        run_notebook(NOTEBOOK_DIR / nb_name)
    print(f"\nAll notebooks completed in {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
