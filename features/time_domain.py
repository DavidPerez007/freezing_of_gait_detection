"""
Time-domain feature extraction for sensor signals.

This module provides functions for extracting statistical and temporal
features from time-series signals.
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict


class TimeDomainFeatures:
    """
    Class for extracting time-domain features from sensor signals.

    Features include statistical moments, peak detection, and other
    temporal characteristics.
    """

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) of a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        float
            RMS value
        """
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def mean(signal: np.ndarray) -> float:
        """Calculate mean value."""
        return float(np.mean(signal))

    @staticmethod
    def std(signal: np.ndarray) -> float:
        """Calculate standard deviation."""
        return float(np.std(signal))

    @staticmethod
    def skewness(signal: np.ndarray) -> float:
        """Calculate skewness (asymmetry of distribution)."""
        return float(stats.skew(signal))

    @staticmethod
    def kurtosis(signal: np.ndarray) -> float:
        """Calculate kurtosis (tailedness of distribution)."""
        return float(stats.kurtosis(signal))

    @staticmethod
    def median(signal: np.ndarray) -> float:
        """Calculate median value."""
        return float(np.median(signal))

    @staticmethod
    def iqr(signal: np.ndarray) -> float:
        """Calculate Interquartile Range (IQR)."""
        q75, q25 = np.percentile(signal, [75, 25])
        return float(q75 - q25)

    @staticmethod
    def peak_to_peak(signal: np.ndarray) -> float:
        """Calculate peak-to-peak amplitude."""
        return float(np.ptp(signal))

    @staticmethod
    def zero_crossing_rate(signal: np.ndarray) -> float:
        """
        Calculate zero-crossing rate (number of times signal crosses zero).

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        float
            Zero-crossing rate (normalized by signal length)
        """
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)

    @staticmethod
    def cadence_from_peaks(
        signal: np.ndarray,
        sampling_rate: int = 64,
        min_peak_distance_sec: float = 0.3
    ) -> float:
        """
        Estimate cadence (steps per minute) from peak detection.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (typically magnitude of acceleration)
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        min_peak_distance_sec : float, optional
            Minimum distance between peaks in seconds (default: 0.3)

        Returns
        -------
        float
            Estimated cadence in steps per minute
        """
        min_distance = int(min_peak_distance_sec * sampling_rate)
        peaks, _ = find_peaks(signal, distance=min_distance)

        steps = len(peaks)
        duration_min = len(signal) / sampling_rate / 60.0

        if duration_min > 0:
            return steps / duration_min
        else:
            return 0.0

    @classmethod
    def extract_all(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all time-domain features from a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        prefix : str, optional
            Prefix to add to feature names (default: '')

        Returns
        -------
        Dict[str, float]
            Dictionary of feature names and values

        Examples
        --------
        >>> features = TimeDomainFeatures.extract_all(signal, prefix='ankle_x_')
        >>> print(features.keys())
        dict_keys(['ankle_x_mean', 'ankle_x_std', ...])
        """
        features = {}

        # Basic statistics
        features[f'{prefix}mean'] = cls.mean(signal)
        features[f'{prefix}std'] = cls.std(signal)
        features[f'{prefix}skew'] = cls.skewness(signal)
        features[f'{prefix}kurt'] = cls.kurtosis(signal)
        features[f'{prefix}median'] = cls.median(signal)
        features[f'{prefix}iqr'] = cls.iqr(signal)
        features[f'{prefix}rms'] = cls.rms(signal)

        # Range features
        features[f'{prefix}peak_to_peak'] = cls.peak_to_peak(signal)

        # Rate features
        features[f'{prefix}zero_crossing_rate'] = cls.zero_crossing_rate(signal)

        return features
