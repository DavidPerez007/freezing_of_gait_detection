"""
Time-domain feature extraction for sensor signals.

This module provides functions for extracting statistical and temporal
features from time-series signals.
"""

import numpy as np
from scipy import stats
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
        """Calculate zero-crossing rate (normalized by signal length)."""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)

    @staticmethod
    def mean_crossing_rate(signal: np.ndarray) -> float:
        """Calculate mean-crossing rate (crossings around mean, normalized)."""
        centered = signal - np.mean(signal)
        crossings = np.where(np.diff(np.sign(centered)))[0]
        return len(crossings) / len(signal)

    @staticmethod
    def signal_energy(signal: np.ndarray) -> float:
        """Calculate normalized signal energy (sum of squares / length)."""
        return float(np.sum(signal ** 2) / len(signal))

    @staticmethod
    def jerk_rms(signal: np.ndarray, sampling_rate: int = 64) -> float:
        """Calculate RMS of the jerk (first derivative) of the signal."""
        jerk = np.diff(signal) * sampling_rate
        return float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else 0.0

    @staticmethod
    def jerk_std(signal: np.ndarray, sampling_rate: int = 64) -> float:
        """Calculate standard deviation of the jerk (first derivative)."""
        jerk = np.diff(signal) * sampling_rate
        return float(np.std(jerk)) if len(jerk) > 0 else 0.0

    @staticmethod
    def coefficient_of_variation(signal: np.ndarray) -> float:
        """Calculate coefficient of variation (std / |mean|)."""
        m = np.abs(np.mean(signal))
        return float(np.std(signal) / m) if m > 1e-12 else 0.0

    @staticmethod
    def entropy(signal: np.ndarray, n_bins: int = 20) -> float:
        """Calculate signal entropy from histogram approximation."""
        hist, _ = np.histogram(signal, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (signal.max() - signal.min()) / n_bins if signal.max() != signal.min() else 1.0
        probs = hist * bin_width
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

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
        features[f'{prefix}mean_crossing_rate'] = cls.mean_crossing_rate(signal)

        # Energy and dynamics features (important for FoG detection)
        features[f'{prefix}energy'] = cls.signal_energy(signal)
        features[f'{prefix}jerk_rms'] = cls.jerk_rms(signal, sampling_rate)
        features[f'{prefix}jerk_std'] = cls.jerk_std(signal, sampling_rate)
        features[f'{prefix}coeff_variation'] = cls.coefficient_of_variation(signal)
        features[f'{prefix}entropy'] = cls.entropy(signal)

        return features
