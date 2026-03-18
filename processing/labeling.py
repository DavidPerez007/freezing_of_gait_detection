"""
Label creation functions for Daphnet FoG Dataset.

This module provides functions for creating binary labels from raw Daphnet
annotations.
"""

import pandas as pd
from utils.constants import (
    ANNOTATION_FREEZE,
    SAMPLING_RATE
)


class LabelCreator:
    """
    Class for creating binary labels from Daphnet annotations.

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
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        """
        self.annotation_col = annotation_col
        self.subject_col = subject_col
        self.trial_col = trial_col
        self.sampling_rate = sampling_rate

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

    def create_all_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create binary labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Daphnet annotations

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'binary_label' column

        Examples
        --------
        >>> creator = LabelCreator()
        >>> df_labeled = creator.create_all_labels(df)
        """
        df_labeled = df.copy()

        df_labeled['binary_label'] = self.create_binary_labels(df)

        return df_labeled

    def get_label_distribution(
        self,
        labels: pd.Series
    ) -> pd.DataFrame:
        """
        Get distribution of labels.

        Parameters
        ----------
        labels : pd.Series
            Label series

        Returns
        -------
        pd.DataFrame
            Distribution with counts and percentages

        Examples
        --------
        >>> creator = LabelCreator()
        >>> dist = creator.get_label_distribution(binary_labels)
        """
        counts = labels.value_counts().sort_index()
        percentages = (counts / len(labels) * 100).round(2)

        distribution = pd.DataFrame({
            'count': counts,
            'percentage': percentages
        })

        from utils.constants import LABEL_NAMES_BINARY
        distribution['label_name'] = distribution.index.map(LABEL_NAMES_BINARY)

        return distribution
