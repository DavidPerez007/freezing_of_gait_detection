"""
Windowing functions for time-series data segmentation.

This module provides functions for creating sliding windows from continuous
time-series data while preserving temporal continuity within subject/trial boundaries.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from utils.constants import (
    WINDOW_SIZE_SEC,
    WINDOW_OVERLAP,
    SAMPLING_RATE,
    WINDOW_SIZE_SAMPLES,
    STEP_SIZE_SAMPLES
)


class WindowCreator:
    """
    Class for creating sliding windows from time-series sensor data.

    Windows are created with specified duration and overlap, maintaining
    temporal continuity within each subject/trial to avoid boundary artifacts.
    """

    def __init__(
        self,
        window_size: float = WINDOW_SIZE_SEC,
        overlap: float = WINDOW_OVERLAP,
        sampling_rate: int = SAMPLING_RATE
    ):
        """
        Initialize the WindowCreator.

        Parameters
        ----------
        window_size : float, optional
            Window duration in seconds (default: 4.0)
        overlap : float, optional
            Overlap ratio between consecutive windows (default: 0.5)
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

    def get_window_label(self, window_labels: np.ndarray) -> int:
        """
        Determine the label for a window using majority voting.

        Parameters
        ----------
        window_labels : np.ndarray
            Array of labels within the window

        Returns
        -------
        int
            The most frequent label in the window

        Examples
        --------
        >>> creator = WindowCreator()
        >>> labels = np.array([0, 0, 1, 0])
        >>> window_label = creator.get_window_label(labels)
        >>> print(window_label)
        0
        """
        return np.bincount(window_labels).argmax()

    def create_sliding_windows(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: Optional[float] = None,
        overlap: Optional[float] = None,
        sampling_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from continuous time-series data.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features)
        labels : np.ndarray
            Labels for each sample of shape (n_samples,)
        window_size : float, optional
            Window duration in seconds (uses instance default if None)
        overlap : float, optional
            Overlap ratio (uses instance default if None)
        sampling_rate : int, optional
            Sampling rate in Hz (uses instance default if None)

        Returns
        -------
        windows : np.ndarray
            Array of windows of shape (n_windows, window_samples, n_features)
        window_labels : np.ndarray
            Array of window labels of shape (n_windows,)

        Examples
        --------
        >>> creator = WindowCreator(window_size=4.0, overlap=0.5)
        >>> windows, labels = creator.create_sliding_windows(data, labels)
        >>> print(windows.shape)
        (n_windows, 256, 9)
        """
        if window_size is None:
            window_size = self.window_size
        if overlap is None:
            overlap = self.overlap
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        window_samples = int(window_size * sampling_rate)
        step_samples = int(window_samples * (1 - overlap))

        windows = []
        window_labels = []

        # Create windows
        for start in range(0, len(data) - window_samples + 1, step_samples):
            end = start + window_samples

            # Extract window
            window = data[start:end]
            window_label_segment = labels[start:end]

            # Window label: majority voting
            window_label = self.get_window_label(window_label_segment)

            windows.append(window)
            window_labels.append(window_label)

        return np.array(windows), np.array(window_labels)

    def create_windows_per_subject(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        subject_col: str = 'subject',
        trial_col: str = 'trial'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows per subject and trial to maintain temporal continuity.

        This ensures windows don't cross trial boundaries, which would create
        artificial discontinuities in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with sensor data
        feature_cols : List[str]
            List of feature column names
        label_col : str
            Name of the label column
        subject_col : str, optional
            Name of the subject column (default: 'subject')
        trial_col : str, optional
            Name of the trial column (default: 'trial')

        Returns
        -------
        all_windows : np.ndarray
            Array of windows of shape (n_windows, window_samples, n_features)
        all_labels : np.ndarray
            Array of window labels of shape (n_windows,)
        all_subjects : np.ndarray
            Array of subject IDs for each window of shape (n_windows,)

        Examples
        --------
        >>> creator = WindowCreator()
        >>> windows, labels, subjects = creator.create_windows_per_subject(
        ...     df, feature_cols=['ankle_x', 'ankle_y'], label_col='fog_label'
        ... )
        """
        all_windows = []
        all_labels = []
        all_subjects = []

        for subject in df[subject_col].unique():
            subject_trials = df[df[subject_col] == subject][trial_col].unique()

            for trial in subject_trials:
                # Filter data for this subject/trial
                mask = (df[subject_col] == subject) & (df[trial_col] == trial)
                subject_data = df[mask]

                # Extract features and labels
                X_subject = subject_data[feature_cols].values
                y_subject = subject_data[label_col].values

                # Create windows for this subject/trial
                windows, labels = self.create_sliding_windows(X_subject, y_subject)

                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.append(labels)
                    all_subjects.extend([subject] * len(windows))

        # Concatenate all windows
        if len(all_windows) > 0:
            all_windows = np.concatenate(all_windows, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_subjects = np.array(all_subjects)
        else:
            all_windows = np.array([])
            all_labels = np.array([])
            all_subjects = np.array([])

        return all_windows, all_labels, all_subjects

    def create_windows_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        binary_label_col: str = 'binary_label',
        multiclass_label_col: str = 'multiclass_label',
        subject_col: str = 'subject',
        trial_col: str = 'trial'
    ) -> dict:
        """
        Create windows for both binary and multiclass labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with sensor data and labels
        feature_cols : List[str]
            List of feature column names
        binary_label_col : str, optional
            Name of binary label column (default: 'binary_label')
        multiclass_label_col : str, optional
            Name of multiclass label column (default: 'multiclass_label')
        subject_col : str, optional
            Name of subject column (default: 'subject')
        trial_col : str, optional
            Name of trial column (default: 'trial')

        Returns
        -------
        dict
            Dictionary with keys:
            - 'binary': {'windows': np.ndarray, 'labels': np.ndarray, 'subjects': np.ndarray}
            - 'multiclass': {'windows': np.ndarray, 'labels': np.ndarray, 'subjects': np.ndarray}

        Examples
        --------
        >>> creator = WindowCreator()
        >>> result = creator.create_windows_from_df(df, feature_cols)
        >>> binary_windows = result['binary']['windows']
        >>> multiclass_windows = result['multiclass']['windows']
        """
        # Create binary windows
        binary_windows, binary_labels, binary_subjects = self.create_windows_per_subject(
            df, feature_cols, binary_label_col, subject_col, trial_col
        )

        # Create multiclass windows
        multiclass_windows, multiclass_labels, multiclass_subjects = self.create_windows_per_subject(
            df, feature_cols, multiclass_label_col, subject_col, trial_col
        )

        return {
            'binary': {
                'windows': binary_windows,
                'labels': binary_labels,
                'subjects': binary_subjects
            },
            'multiclass': {
                'windows': multiclass_windows,
                'labels': multiclass_labels,
                'subjects': multiclass_subjects
            }
        }

    def get_window_info(self) -> dict:
        """
        Get information about window configuration.

        Returns
        -------
        dict
            Dictionary with window configuration details

        Examples
        --------
        >>> creator = WindowCreator()
        >>> info = creator.get_window_info()
        >>> print(info['window_samples'])
        256
        """
        return {
            'window_size_sec': self.window_size,
            'window_samples': self.window_samples,
            'overlap': self.overlap,
            'step_samples': self.step_samples,
            'sampling_rate': self.sampling_rate
        }
