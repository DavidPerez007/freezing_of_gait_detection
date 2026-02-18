"""
Label creation functions for Daphnet FoG Dataset.

This module provides functions for creating binary and multiclass labels
from raw Daphnet annotations, including pre-FoG detection.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from utils.constants import (
    ANNOTATION_FREEZE,
    BINARY_NO_FOG,
    BINARY_FOG,
    MULTICLASS_NO_FOG,
    MULTICLASS_FOG,
    MULTICLASS_PRE_FOG,
    PRE_FOG_WINDOW_SAMPLES,
    SAMPLING_RATE
)


class LabelCreator:
    """
    Class for creating binary and multiclass labels from Daphnet annotations.

    The Daphnet dataset has three annotation values:
    - 0: Not part of the experiment
    - 1: Walking without freeze
    - 2: Freeze episode
    """

    def __init__(
        self,
        annotation_col: str = 'annotation',
        subject_col: str = 'subject',
        trial_col: str = 'trial',
        pre_fog_window_sec: float = 0.5,
        sampling_rate: int = SAMPLING_RATE
    ):
        """
        Initialize the LabelCreator.

        Parameters
        ----------
        annotation_col : str, optional
            Name of the annotation column (default: 'annotation')
        subject_col : str, optional
            Name of the subject column (default: 'subject')
        trial_col : str, optional
            Name of the trial column (default: 'trial')
        pre_fog_window_sec : float, optional
            Duration of pre-FoG window in seconds (default: 0.5)
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        """
        self.annotation_col = annotation_col
        self.subject_col = subject_col
        self.trial_col = trial_col
        self.pre_fog_window_sec = pre_fog_window_sec
        self.sampling_rate = sampling_rate
        self.pre_fog_window_samples = int(pre_fog_window_sec * sampling_rate)

    def create_binary_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels from Daphnet annotations.

        Binary labels:
        - 0: No FoG (annotation == 0 or 1)
        - 1: FoG (annotation == 2)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Daphnet annotations

        Returns
        -------
        pd.Series
            Binary labels (0 or 1)

        Examples
        --------
        >>> creator = LabelCreator()
        >>> binary_labels = creator.create_binary_labels(df)
        """
        return (df[self.annotation_col] == ANNOTATION_FREEZE).astype(int)

    def detect_fog_onsets(self, annotations: np.ndarray) -> List[int]:
        """
        Detect FoG episode onset indices in annotation sequence.

        An onset is detected when annotation changes from non-FoG to FoG.

        Parameters
        ----------
        annotations : np.ndarray
            Annotation sequence

        Returns
        -------
        List[int]
            List of indices where FoG episodes start

        Examples
        --------
        >>> annotations = np.array([1, 1, 2, 2, 1, 2])
        >>> creator = LabelCreator()
        >>> onsets = creator.detect_fog_onsets(annotations)
        >>> print(onsets)
        [2, 5]
        """
        fog_starts = []

        for i in range(1, len(annotations)):
            # Transition from non-FoG to FoG
            if annotations[i] == ANNOTATION_FREEZE and annotations[i-1] != ANNOTATION_FREEZE:
                fog_starts.append(i)

        return fog_starts

    def create_multiclass_labels(
        self,
        df: pd.DataFrame,
        include_pre_fog: bool = True
    ) -> pd.Series:
        """
        Create multiclass labels from Daphnet annotations.

        Multiclass labels:
        - 0: No FoG
        - 1: FoG
        - 2: Pre-FoG (optional, 0.5 seconds before FoG onset)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Daphnet annotations
        include_pre_fog : bool, optional
            Whether to include pre-FoG class (default: True)

        Returns
        -------
        pd.Series
            Multiclass labels (0, 1, or 2)

        Examples
        --------
        >>> creator = LabelCreator()
        >>> multiclass_labels = creator.create_multiclass_labels(df)
        """
        # Initialize labels: 0 = No FoG
        labels = np.zeros(len(df), dtype=int)

        # Mark FoG episodes: 1 = FoG
        labels[df[self.annotation_col] == ANNOTATION_FREEZE] = MULTICLASS_FOG

        if not include_pre_fog:
            return pd.Series(labels, index=df.index, name='multiclass_label')

        # Mark pre-FoG windows
        df_copy = df.copy()
        df_copy['_temp_label'] = labels

        for subject in df[self.subject_col].unique():
            subject_trials = df[df[self.subject_col] == subject][self.trial_col].unique()

            for trial in subject_trials:
                # Get indices for this subject/trial
                mask = (df[self.subject_col] == subject) & (df[self.trial_col] == trial)
                indices = df[mask].index.tolist()

                if len(indices) == 0:
                    continue

                # Get annotations for this trial
                trial_annotations = df.loc[indices, self.annotation_col].values

                # Detect FoG onsets
                fog_starts = self.detect_fog_onsets(trial_annotations)

                # Mark pre-FoG windows
                for start_idx in fog_starts:
                    # Calculate pre-FoG range (before onset)
                    pre_fog_start = max(0, start_idx - self.pre_fog_window_samples)
                    pre_fog_end = start_idx

                    # Get corresponding global indices
                    pre_fog_global_indices = indices[pre_fog_start:pre_fog_end]

                    # Only mark as pre-FoG where currently labeled as No FoG
                    for idx in pre_fog_global_indices:
                        if labels[df.index.get_loc(idx)] == MULTICLASS_NO_FOG:
                            labels[df.index.get_loc(idx)] = MULTICLASS_PRE_FOG

        return pd.Series(labels, index=df.index, name='multiclass_label')

    def create_all_labels(
        self,
        df: pd.DataFrame,
        include_pre_fog: bool = True
    ) -> pd.DataFrame:
        """
        Create both binary and multiclass labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Daphnet annotations
        include_pre_fog : bool, optional
            Whether to include pre-FoG class in multiclass labels (default: True)

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'binary_label' and 'multiclass_label' columns

        Examples
        --------
        >>> creator = LabelCreator()
        >>> df_labeled = creator.create_all_labels(df)
        """
        df_labeled = df.copy()

        df_labeled['binary_label'] = self.create_binary_labels(df)
        df_labeled['multiclass_label'] = self.create_multiclass_labels(df, include_pre_fog)

        return df_labeled

    def get_label_distribution(
        self,
        labels: pd.Series,
        label_type: str = 'multiclass'
    ) -> pd.DataFrame:
        """
        Get distribution of labels.

        Parameters
        ----------
        labels : pd.Series
            Label series
        label_type : str, optional
            Type of labels: 'binary' or 'multiclass' (default: 'multiclass')

        Returns
        -------
        pd.DataFrame
            Distribution with counts and percentages

        Examples
        --------
        >>> creator = LabelCreator()
        >>> dist = creator.get_label_distribution(multiclass_labels, 'multiclass')
        """
        counts = labels.value_counts().sort_index()
        percentages = (counts / len(labels) * 100).round(2)

        distribution = pd.DataFrame({
            'count': counts,
            'percentage': percentages
        })

        # Add label names
        if label_type == 'binary':
            from utils.constants import LABEL_NAMES_BINARY
            distribution['label_name'] = distribution.index.map(LABEL_NAMES_BINARY)
        else:
            from utils.constants import LABEL_NAMES_MULTICLASS
            distribution['label_name'] = distribution.index.map(LABEL_NAMES_MULTICLASS)

        return distribution
