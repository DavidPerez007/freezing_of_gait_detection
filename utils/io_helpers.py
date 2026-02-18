"""
I/O helper functions for Daphnet FoG Dataset processing.

This module provides utility functions for saving/loading pickle files
and managing output directories.
"""

import pickle
from pathlib import Path
from typing import Any


def save_pickle(data: Any, filepath: str | Path) -> None:
    """
    Save data to a pickle file.

    Parameters
    ----------
    data : Any
        Data to save (can be any pickle-serializable object)
    filepath : str | Path
        Path where to save the pickle file

    Examples
    --------
    >>> save_pickle({'data': [1, 2, 3]}, 'output.pkl')
    """
    filepath = Path(filepath)
    ensure_output_dir(filepath.parent)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Data saved to: {filepath}")


def load_pickle(filepath: str | Path) -> Any:
    """
    Load data from a pickle file.

    Parameters
    ----------
    filepath : str | Path
        Path to the pickle file to load

    Returns
    -------
    Any
        The loaded data

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist

    Examples
    --------
    >>> data = load_pickle('output.pkl')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"📂 Data loaded from: {filepath}")
    return data


def ensure_output_dir(dirpath: str | Path) -> Path:
    """
    Ensure that an output directory exists. Create it if it doesn't.

    Parameters
    ----------
    dirpath : str | Path
        Path to the directory

    Returns
    -------
    Path
        The Path object to the directory

    Examples
    --------
    >>> output_dir = ensure_output_dir('outputs/features')
    >>> print(output_dir)
    PosixPath('outputs/features')
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath
