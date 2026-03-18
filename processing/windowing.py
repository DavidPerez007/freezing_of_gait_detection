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
    STEP_SIZE_SAMPLES,
    WINDOW_LABEL_STRATEGY,
    WINDOW_LABEL_MIN_POSITIVE_RATIO,
    BINARY_FOG
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
        sampling_rate: int = SAMPLING_RATE,
        label_strategy: str = WINDOW_LABEL_STRATEGY,
        positive_labels: Optional[List[int]] = None,
        min_positive_ratio: float = WINDOW_LABEL_MIN_POSITIVE_RATIO
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
        self.label_strategy = label_strategy
        self.positive_labels = list(positive_labels) if positive_labels is not None else [BINARY_FOG]
        self.min_positive_ratio = min_positive_ratio
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

    def get_window_label_stats(self, window_labels: np.ndarray) -> dict:
        """
        Compute label composition statistics inside a window.

        Parameters
        ----------
        window_labels : np.ndarray
            Array of labels within the window

        Returns
        -------
        dict
            Window-level label statistics
        """
        window_labels = np.asarray(window_labels)

        if window_labels.size == 0:
            return {
                'dominant_label': 0,
                'positive_ratio': 0.0,
                'positive_count': 0,
                'window_length': 0,
            }

        dominant_label = int(np.bincount(window_labels.astype(int)).argmax())
        positive_mask = np.isin(window_labels, self.positive_labels)
        positive_count = int(np.sum(positive_mask))
        positive_ratio = float(positive_count / len(window_labels))

        return {
            'dominant_label': dominant_label,
            'positive_ratio': positive_ratio,
            'positive_count': positive_count,
            'window_length': int(len(window_labels)),
        }

    def get_window_label(self, window_labels: np.ndarray) -> int:
        """
        Determine the label for a window using a configurable strategy.

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
        stats = self.get_window_label_stats(window_labels)
        dominant_label = stats['dominant_label']
        positive_ratio = stats['positive_ratio']

        if self.label_strategy == 'majority':
            return dominant_label

        if self.label_strategy == 'any_positive':
            return int(positive_ratio > 0)

        if self.label_strategy == 'center':
            center_idx = len(window_labels) // 2
            return int(window_labels[center_idx] in self.positive_labels)

        if self.label_strategy == 'min_positive_ratio':
            return int(positive_ratio >= self.min_positive_ratio)

        raise ValueError(f"Unknown label strategy: {self.label_strategy}")

    def create_sliding_windows(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: Optional[float] = None,
        overlap: Optional[float] = None,
        sampling_rate: Optional[int] = None,
        return_metadata: bool = False
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
        metadata = []

        # Create windows
        for start in range(0, len(data) - window_samples + 1, step_samples):
            end = start + window_samples

            # Extract window
            window = data[start:end]
            window_label_segment = labels[start:end]

            # Window label: majority voting
            window_label = self.get_window_label(window_label_segment)
            label_stats = self.get_window_label_stats(window_label_segment)

            windows.append(window)
            window_labels.append(window_label)
            metadata.append({
                'start': int(start),
                'end': int(end),
                **label_stats,
            })

        if return_metadata:
            return np.array(windows), np.array(window_labels), metadata

        return np.array(windows), np.array(window_labels)

    def create_windows_per_subject(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        subject_col: str = 'subject',
        trial_col: str = 'trial',
        return_metadata: bool = False
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
        all_metadata = []

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
                if return_metadata:
                    windows, labels, metadata = self.create_sliding_windows(
                        X_subject, y_subject, return_metadata=True
                    )
                else:
                    windows, labels = self.create_sliding_windows(X_subject, y_subject)

                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.append(labels)
                    all_subjects.extend([subject] * len(windows))
                    if return_metadata:
                        for item in metadata:
                            item['subject'] = subject
                            item['trial'] = trial
                        all_metadata.extend(metadata)

        # Concatenate all windows
        if len(all_windows) > 0:
            all_windows = np.concatenate(all_windows, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_subjects = np.array(all_subjects)
        else:
            all_windows = np.array([])
            all_labels = np.array([])
            all_subjects = np.array([])

        if return_metadata:
            return all_windows, all_labels, all_subjects, all_metadata

        return all_windows, all_labels, all_subjects

    def create_windows_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        binary_label_col: str = 'binary_label',
        subject_col: str = 'subject',
        trial_col: str = 'trial',
        binary_label_strategy: Optional[str] = None,
        binary_positive_labels: Optional[List[int]] = None,
        binary_min_positive_ratio: Optional[float] = None,
        return_metadata: bool = False
    ) -> dict:
        """
        Create windows for binary labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with sensor data and labels
        feature_cols : List[str]
            List of feature column names
        binary_label_col : str, optional
            Name of binary label column (default: 'binary_label')
        subject_col : str, optional
            Name of subject column (default: 'subject')
        trial_col : str, optional
            Name of trial column (default: 'trial')

        Returns
        -------
        dict
            Dictionary with key 'binary' containing window arrays, labels, and subjects

        Examples
        --------
        >>> creator = WindowCreator()
        >>> result = creator.create_windows_from_df(df, feature_cols)
        >>> binary_windows = result['binary']['windows']
        """
        binary_creator = WindowCreator(
            window_size=self.window_size,
            overlap=self.overlap,
            sampling_rate=self.sampling_rate,
            label_strategy=binary_label_strategy or self.label_strategy,
            positive_labels=binary_positive_labels or self.positive_labels,
            min_positive_ratio=(
                self.min_positive_ratio if binary_min_positive_ratio is None else binary_min_positive_ratio
            )
        )

        if return_metadata:
            binary_windows, binary_labels, binary_subjects, binary_metadata = binary_creator.create_windows_per_subject(
                df, feature_cols, binary_label_col, subject_col, trial_col, return_metadata=True
            )
        else:
            binary_windows, binary_labels, binary_subjects = binary_creator.create_windows_per_subject(
                df, feature_cols, binary_label_col, subject_col, trial_col
            )

        result = {
            'binary': {
                'windows': binary_windows,
                'labels': binary_labels,
                'subjects': binary_subjects
            }
        }

        if return_metadata:
            result['binary']['metadata'] = binary_metadata

        return result

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
            'sampling_rate': self.sampling_rate,
            'label_strategy': self.label_strategy,
            'positive_labels': self.positive_labels,
            'min_positive_ratio': self.min_positive_ratio,
        }
