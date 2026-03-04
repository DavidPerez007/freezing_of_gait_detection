"""
Frequency-domain feature extraction for sensor signals.

This module provides functions for extracting features from the frequency
spectrum using Welch's method for PSD estimation.
"""

import numpy as np
from scipy.signal import welch
from typing import Dict, Tuple
from utils.constants import FREQ_BAND_LOCOMOTION, FREQ_BAND_FREEZING


class FrequencyDomainFeatures:
    """
    Class for extracting frequency-domain features from sensor signals.

    Uses Welch's method for Power Spectral Density (PSD) estimation.
    """

    EPSILON = 1e-12

    @staticmethod
    def _robust_trapz(y: np.ndarray, x: np.ndarray) -> float:
        """
        Robust trapezoidal integration.

        Parameters
        ----------
        y : np.ndarray
            Y values
        x : np.ndarray
            X values

        Returns
        -------
        float
            Integrated value
        """
        # Prefer numpy.trapezoid (newer NumPy) then numpy.trapz
        try:
            if hasattr(np, 'trapezoid'):
                return float(np.trapezoid(y, x))
            elif hasattr(np, 'trapz'):
                return float(np.trapz(y, x))
        except Exception:
            pass

        # Manual implementation as fallback
        y = np.asarray(y)
        x = np.asarray(x)

        if y.size == 0 or x.size == 0 or y.size != x.size:
            return 0.0
        if y.size == 1:
            return 0.0

        return float(0.5 * np.sum((y[:-1] + y[1:]) * (x[1:] - x[:-1])))

    @staticmethod
    def compute_psd(
        signal: np.ndarray,
        sampling_rate: int = 64,
        nperseg: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        nperseg : int, optional
            Length of each segment for Welch's method (default: 256)

        Returns
        -------
        freqs : np.ndarray
            Frequency bins
        psd : np.ndarray
            Power spectral density values
        """
        nperseg_actual = min(nperseg, len(signal))
        freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg_actual)
        return freqs, psd

    @classmethod
    def psd_peak_frequency(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64
    ) -> float:
        """
        Find the dominant frequency (peak in PSD).

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)

        Returns
        -------
        float
            Dominant frequency in Hz
        """
        freqs, psd = cls.compute_psd(signal, sampling_rate)

        if psd.size > 0:
            peak_idx = np.argmax(psd)
            return float(freqs[peak_idx])
        else:
            return np.nan

    @classmethod
    def psd_total_energy(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64
    ) -> float:
        """
        Calculate total energy in the power spectrum.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)

        Returns
        -------
        float
            Total spectral energy
        """
        freqs, psd = cls.compute_psd(signal, sampling_rate)

        if psd.size > 0:
            return cls._robust_trapz(psd, freqs)
        else:
            return 0.0

    @classmethod
    def band_power(
        cls,
        signal: np.ndarray,
        low_freq: float,
        high_freq: float,
        sampling_rate: int = 64
    ) -> float:
        """
        Calculate power in a specific frequency band.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        low_freq : float
            Lower bound of frequency band in Hz
        high_freq : float
            Upper bound of frequency band in Hz
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)

        Returns
        -------
        float
            Power in the specified frequency band
        """
        freqs, psd = cls.compute_psd(signal, sampling_rate)

        # Find indices within the frequency band
        mask = (freqs >= low_freq) & (freqs <= high_freq)

        if np.any(mask):
            return cls._robust_trapz(psd[mask], freqs[mask])
        else:
            return 0.0

    @classmethod
    def freezing_index(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64,
        freeze_band: Tuple[float, float] = FREQ_BAND_FREEZING,
        loco_band: Tuple[float, float] = FREQ_BAND_LOCOMOTION
    ) -> float:
        """
        Calculate Freezing Index (ratio of freeze band to locomotion band power).

        The freezing index is a well-known metric for FoG detection:
        - Locomotion band: 0.5-3 Hz (normal walking)
        - Freezing band: 3-8 Hz (trembling during freeze)

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        freeze_band : Tuple[float, float], optional
            Freeze frequency band (default: (3.0, 8.0))
        loco_band : Tuple[float, float], optional
            Locomotion frequency band (default: (0.5, 3.0))

        Returns
        -------
        float
            Freezing index (freeze_power / loco_power)

        Notes
        -----
        Higher values indicate more freeze-like behavior.
        """
        power_freeze = cls.band_power(signal, freeze_band[0], freeze_band[1], sampling_rate)
        power_loco = cls.band_power(signal, loco_band[0], loco_band[1], sampling_rate)

        if power_loco > cls.EPSILON:
            return power_freeze / power_loco
        return np.nan

    @classmethod
    def spectral_centroid(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64
    ) -> float:
        """
        Calculate spectral centroid (center of mass of spectrum).

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)

        Returns
        -------
        float
            Spectral centroid in Hz
        """
        freqs, psd = cls.compute_psd(signal, sampling_rate)

        if psd.size > 0 and np.sum(psd) > 0:
            centroid = np.sum(freqs * psd) / np.sum(psd)
            return float(centroid)
        else:
            return np.nan

    @classmethod
    def extract_all(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all frequency-domain features from a signal.

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
        >>> features = FrequencyDomainFeatures.extract_all(signal, prefix='ankle_x_')
        >>> print(features['ankle_x_freezing_index'])
        1.234
        """
        features = {}

        # Compute PSD once and reuse across all features (avoids 8 redundant Welch calls)
        freqs, psd = cls.compute_psd(signal, sampling_rate)

        if psd.size == 0:
            features[f'{prefix}psd_peak_freq']        = np.nan
            features[f'{prefix}psd_total_energy']     = 0.0
            features[f'{prefix}spectral_centroid']    = np.nan
            features[f'{prefix}power_loco_band']      = 0.0
            features[f'{prefix}power_freeze_band']    = 0.0
            features[f'{prefix}freezing_index']       = np.nan
            features[f'{prefix}locomotion_band_index'] = 0.0
            return features

        # Peak frequency
        features[f'{prefix}psd_peak_freq'] = float(freqs[np.argmax(psd)])

        # Total energy and spectral centroid
        total_energy = cls._robust_trapz(psd, freqs)
        features[f'{prefix}psd_total_energy'] = total_energy
        features[f'{prefix}spectral_centroid'] = (
            float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else np.nan
        )

        # Band powers — masks reuse the pre-computed freqs array
        loco_low,   loco_high   = FREQ_BAND_LOCOMOTION
        freeze_low, freeze_high = FREQ_BAND_FREEZING

        mask_loco   = (freqs >= loco_low)   & (freqs <= loco_high)
        mask_freeze = (freqs >= freeze_low) & (freqs <= freeze_high)

        power_loco   = cls._robust_trapz(psd[mask_loco],   freqs[mask_loco])   if np.any(mask_loco)   else 0.0
        power_freeze = cls._robust_trapz(psd[mask_freeze], freqs[mask_freeze]) if np.any(mask_freeze) else 0.0

        features[f'{prefix}power_loco_band']   = power_loco
        features[f'{prefix}power_freeze_band'] = power_freeze

        # Freezing index (key FoG metric): ratio of freeze to locomotion band power
        features[f'{prefix}freezing_index'] = (
            power_freeze / power_loco if power_loco > cls.EPSILON else np.nan
        )

        # Locomotion Band Index: proportion of total power in the locomotion band
        features[f'{prefix}locomotion_band_index'] = (
            power_loco / total_energy if total_energy > 0 else 0.0
        )

        return features

    @classmethod
    def locomotion_band_index(
        cls,
        signal: np.ndarray,
        sampling_rate: int = 64,
        loco_band: Tuple[float, float] = FREQ_BAND_LOCOMOTION
    ) -> float:
        """
        Calculate Locomotion Band Index (LBI).

        LBI = power_locomotion_band / total_psd_power

        Complements the Freezing Index by capturing how dominant normal walking
        rhythms are. During FoG, LBI drops as the locomotion band loses energy.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        loco_band : Tuple[float, float], optional
            Locomotion frequency band (default: (0.5, 3.0))

        Returns
        -------
        float
            Locomotion band index in [0, 1]
        """
        power_loco  = cls.band_power(signal, loco_band[0], loco_band[1], sampling_rate)
        power_total = cls.psd_total_energy(signal, sampling_rate)

        if power_total > 0:
            return float(power_loco / power_total)
        else:
            return 0.0
