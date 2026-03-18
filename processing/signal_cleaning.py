"""
Signal cleaning functions for FoG Dataset processing.

This module provides functions for detecting and correcting outliers,
handling missing values, and bandpass filtering in time-series sensor data.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Optional
from utils.constants import (
    OUTLIER_THRESH_MULTIPLIER,
    OUTLIER_POLY_ORDER,
    MISSING_VALUE_POLY_ORDER,
    OUTLIER_INTERPOLATION_METHOD,
    MISSING_VALUE_INTERPOLATION_METHOD,
    MAX_OUTLIER_FRACTION_PER_CHANNEL,
    MAX_MISSING_FRACTION_PER_CHANNEL,
)


class SignalCleaner:
    """
    Class for cleaning sensor signals by detecting and correcting outliers
    and handling missing values.

    The class uses MAD (Median Absolute Deviation) for outlier detection
    and polynomial interpolation for correction.
    """

    def __init__(
        self,
        outlier_thresh_mul: float = OUTLIER_THRESH_MULTIPLIER,
        outlier_poly_order: int = OUTLIER_POLY_ORDER,
        missing_poly_order: int = MISSING_VALUE_POLY_ORDER,
        outlier_interpolation_method: str = OUTLIER_INTERPOLATION_METHOD,
        missing_interpolation_method: str = MISSING_VALUE_INTERPOLATION_METHOD,
        max_outlier_fraction: float = MAX_OUTLIER_FRACTION_PER_CHANNEL,
        max_missing_fraction: float = MAX_MISSING_FRACTION_PER_CHANNEL
    ):
        """
        Initialize the SignalCleaner.

        Parameters
        ----------
        outlier_thresh_mul : float, optional
            MAD threshold multiplier for outlier detection (default: 3.0)
        outlier_poly_order : int, optional
            Polynomial order for outlier interpolation (default: 3)
        missing_poly_order : int, optional
            Polynomial order for missing value interpolation (default: 3)
        """
        self.outlier_thresh_mul = outlier_thresh_mul
        self.outlier_poly_order = outlier_poly_order
        self.missing_poly_order = missing_poly_order
        self.outlier_interpolation_method = outlier_interpolation_method
        self.missing_interpolation_method = missing_interpolation_method
        self.max_outlier_fraction = max_outlier_fraction
        self.max_missing_fraction = max_missing_fraction

    def _interpolate_masked_values(
        self,
        signal: np.ndarray,
        bad_mask: np.ndarray,
        poly_order: int,
        method: str,
        max_fraction: float
    ) -> np.ndarray:
        """Interpolate masked values conservatively to preserve FoG dynamics."""
        signal = np.array(signal, dtype=float).copy()
        bad_mask = np.array(bad_mask, dtype=bool)

        if not np.any(bad_mask):
            return signal

        if bad_mask.mean() > max_fraction:
            return signal

        idx = np.arange(signal.shape[0])
        good_idx = idx[~bad_mask]
        bad_idx = idx[bad_mask]

        if good_idx.size == 0:
            signal[bad_mask] = 0.0
            return signal

        if good_idx.size == 1:
            signal[bad_mask] = signal[good_idx[0]]
            return signal

        if method == 'polynomial':
            deg = min(poly_order, good_idx.size - 1)
            try:
                coeffs = np.polyfit(good_idx, signal[good_idx], deg)
                signal[bad_idx] = np.polyval(coeffs, bad_idx)
                return signal
            except Exception:
                pass

        signal[bad_idx] = np.interp(bad_idx, good_idx, signal[good_idx])
        return signal

    def detect_outliers_mad(
        self,
        signal: np.ndarray,
        thresh_mul: Optional[float] = None
    ) -> np.ndarray:
        """
        Detect outliers in a signal using MAD (Median Absolute Deviation).

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array)
        thresh_mul : float, optional
            MAD threshold multiplier (uses instance default if None)

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates an outlier

        Notes
        -----
        Outlier threshold = thresh_mul * 1.4826 * MAD
        The constant 1.4826 makes MAD comparable to standard deviation
        for normally distributed data.
        """
        if thresh_mul is None:
            thresh_mul = self.outlier_thresh_mul

        median = np.median(signal)
        mad = np.median(np.abs(signal - median))

        # Handle case where MAD is zero or NaN
        if mad == 0 or np.isnan(mad):
            mad = np.std(signal) if np.std(signal) > 0 else 1.0

        threshold = thresh_mul * 1.4826 * mad
        outlier_mask = np.abs(signal - median) > threshold

        return outlier_mask

    def interpolate_outliers(
        self,
        windows: np.ndarray,
        poly_order: Optional[int] = None,
        thresh_mul: Optional[float] = None
    ) -> np.ndarray:
        """
        Interpolate outliers in windowed sensor data.

        Detects outliers using MAD per channel and replaces them using
        polynomial interpolation fitted on non-outlier points.

        Parameters
        ----------
        windows : np.ndarray
            Input windows of shape (n_windows, n_samples, n_channels)
        poly_order : int, optional
            Polynomial order for interpolation (uses instance default if None)
        thresh_mul : float, optional
            MAD threshold multiplier (uses instance default if None)

        Returns
        -------
        np.ndarray
            Cleaned windows with same shape as input

        Notes
        -----
        - Processes each channel independently
        - Falls back to linear interpolation if not enough points
        - Falls back to median if insufficient good points for interpolation
        """
        if poly_order is None:
            poly_order = self.outlier_poly_order
        if thresh_mul is None:
            thresh_mul = self.outlier_thresh_mul

        windows_clean = np.array(windows, dtype=float).copy()

        if windows_clean.size == 0:
            return windows_clean

        n_windows, n_samples, n_channels = windows_clean.shape

        for i in range(n_windows):
            for ch in range(n_channels):
                signal = windows_clean[i, :, ch]

                # Detect outliers
                outlier_mask = self.detect_outliers_mad(signal, thresh_mul)

                if not np.any(outlier_mask):
                    continue  # No outliers, skip

                signal = self._interpolate_masked_values(
                    signal,
                    outlier_mask,
                    poly_order=poly_order,
                    method=self.outlier_interpolation_method,
                    max_fraction=self.max_outlier_fraction,
                )

                windows_clean[i, :, ch] = signal

        return windows_clean

    def interpolate_missing_values(
        self,
        windows: np.ndarray,
        poly_order: Optional[int] = None
    ) -> np.ndarray:
        """
        Interpolate missing values (NaN) in windowed sensor data.

        Parameters
        ----------
        windows : np.ndarray
            Input windows of shape (n_windows, n_samples, n_channels)
        poly_order : int, optional
            Polynomial order for interpolation (uses instance default if None)

        Returns
        -------
        np.ndarray
            Windows with missing values filled

        Notes
        -----
        - Processes each channel independently
        - Falls back to linear interpolation if not enough points
        - Uses median if insufficient valid points
        """
        if poly_order is None:
            poly_order = self.missing_poly_order

        windows_filled = np.array(windows, dtype=float).copy()

        if windows_filled.size == 0:
            return windows_filled

        n_windows, n_samples, n_channels = windows_filled.shape

        for i in range(n_windows):
            for ch in range(n_channels):
                signal = windows_filled[i, :, ch]
                nan_mask = np.isnan(signal)

                if not np.any(nan_mask):
                    continue  # No NaNs, skip

                signal = self._interpolate_masked_values(
                    signal,
                    nan_mask,
                    poly_order=poly_order,
                    method=self.missing_interpolation_method,
                    max_fraction=self.max_missing_fraction,
                )

                if np.isnan(signal).any():
                    fill_value = np.nanmedian(signal)
                    if np.isnan(fill_value):
                        fill_value = 0.0
                    signal[np.isnan(signal)] = fill_value

                windows_filled[i, :, ch] = signal

        return windows_filled

    @staticmethod
    def bandpass_filter_windows(
        windows: np.ndarray,
        sampling_rate: int,
        low_freq: float = 0.5,
        high_freq: float = 25.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to each channel of each window.

        Removes gravity/DC offset (below low_freq) and high-frequency noise
        (above high_freq), preserving the FoG-relevant 0.5-8 Hz band.

        Parameters
        ----------
        windows : np.ndarray
            Input windows of shape (n_windows, n_samples, n_channels)
        sampling_rate : int
            Sampling rate in Hz
        low_freq : float
            Lower cutoff frequency in Hz (default: 0.5)
        high_freq : float
            Upper cutoff frequency in Hz (default: 25.0)
        order : int
            Butterworth filter order (default: 4)

        Returns
        -------
        np.ndarray
            Filtered windows with same shape as input
        """
        nyquist = sampling_rate / 2.0
        high_freq = min(high_freq, nyquist - 1.0)
        if low_freq >= high_freq:
            return windows.copy()

        sos = butter(order, [low_freq / nyquist, high_freq / nyquist],
                     btype='band', output='sos')

        filtered = windows.copy().astype(float)
        n_windows, n_samples, n_channels = filtered.shape

        for i in range(n_windows):
            for ch in range(n_channels):
                try:
                    filtered[i, :, ch] = sosfiltfilt(sos, filtered[i, :, ch])
                except ValueError:
                    pass  # skip if window too short for filter

        return filtered

    @staticmethod
    def bandpass_filter_signal(
        signal: np.ndarray,
        sampling_rate: int,
        low_freq: float = 0.5,
        high_freq: float = 25.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to a multi-channel continuous signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape (n_samples,) or (n_samples, n_channels)
        sampling_rate : int
            Sampling rate in Hz
        low_freq : float
            Lower cutoff frequency in Hz (default: 0.5)
        high_freq : float
            Upper cutoff frequency in Hz (default: 25.0)
        order : int
            Butterworth filter order (default: 4)

        Returns
        -------
        np.ndarray
            Filtered signal with same shape as input
        """
        nyquist = sampling_rate / 2.0
        high_freq = min(high_freq, nyquist - 1.0)
        if low_freq >= high_freq:
            return signal.copy()

        sos = butter(order, [low_freq / nyquist, high_freq / nyquist],
                     btype='band', output='sos')

        filtered = signal.copy().astype(float)
        if filtered.ndim == 1:
            filtered = sosfiltfilt(sos, filtered)
        else:
            for ch in range(filtered.shape[1]):
                filtered[:, ch] = sosfiltfilt(sos, filtered[:, ch])

        return filtered

    def clean_windows(
        self,
        windows: np.ndarray,
        interpolate_outliers: bool = True,
        interpolate_missing: bool = True
    ) -> np.ndarray:
        """
        Clean windows by detecting and correcting outliers and missing values.

        Parameters
        ----------
        windows : np.ndarray
            Input windows of shape (n_windows, n_samples, n_channels)
        interpolate_outliers : bool, optional
            Whether to detect and interpolate outliers (default: True)
        interpolate_missing : bool, optional
            Whether to interpolate missing values (default: True)

        Returns
        -------
        np.ndarray
            Cleaned windows

        Examples
        --------
        >>> cleaner = SignalCleaner()
        >>> clean_windows = cleaner.clean_windows(raw_windows)
        """
        cleaned = windows.copy()

        if interpolate_outliers:
            cleaned = self.interpolate_outliers(cleaned)

        if interpolate_missing:
            cleaned = self.interpolate_missing_values(cleaned)

        return cleaned
