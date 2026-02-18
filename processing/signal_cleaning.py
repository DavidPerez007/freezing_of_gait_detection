"""
Signal cleaning functions for Daphnet FoG Dataset processing.

This module provides functions for detecting and correcting outliers,
and handling missing values in time-series sensor data.
"""

import numpy as np
from typing import Optional
from utils.constants import OUTLIER_THRESH_MULTIPLIER, OUTLIER_POLY_ORDER, MISSING_VALUE_POLY_ORDER


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
        missing_poly_order: int = MISSING_VALUE_POLY_ORDER
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

                # Get indices
                idx = np.arange(n_samples)
                good_idx = idx[~outlier_mask]
                bad_idx = idx[outlier_mask]

                # Interpolate outliers
                if good_idx.size >= 2:
                    deg = min(poly_order, good_idx.size - 1)
                    try:
                        coeffs = np.polyfit(good_idx, signal[good_idx], deg)
                        signal[bad_idx] = np.polyval(coeffs, bad_idx)
                    except Exception:
                        # Fallback to linear interpolation
                        if good_idx.size >= 2:
                            signal[bad_idx] = np.interp(bad_idx, good_idx, signal[good_idx])
                        else:
                            signal[outlier_mask] = np.median(signal)
                else:
                    # Not enough good points, use median
                    signal[outlier_mask] = np.median(signal)

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

                # Get indices
                idx = np.arange(n_samples)
                good_idx = idx[~nan_mask]
                bad_idx = idx[nan_mask]

                # Interpolate missing values
                if good_idx.size >= 2:
                    deg = min(poly_order, good_idx.size - 1)
                    try:
                        coeffs = np.polyfit(good_idx, signal[good_idx], deg)
                        signal[bad_idx] = np.polyval(coeffs, bad_idx)
                    except Exception:
                        # Fallback to linear interpolation
                        try:
                            signal[bad_idx] = np.interp(bad_idx, good_idx, signal[good_idx])
                        except Exception:
                            # Fallback to median
                            signal[bad_idx] = np.nanmedian(signal[good_idx]) if good_idx.size > 0 else 0.0
                elif good_idx.size == 1:
                    # Only one valid point, use it
                    signal[nan_mask] = signal[good_idx[0]]
                else:
                    # No valid points, fill with zeros
                    signal[nan_mask] = 0.0

                windows_filled[i, :, ch] = signal

        return windows_filled

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
