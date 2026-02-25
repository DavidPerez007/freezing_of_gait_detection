"""
Main feature extraction module combining all feature types.

This module provides a unified interface for extracting time-domain,
frequency-domain, wavelet, and nonlinear features from sensor signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from .wavelet_features import WaveletFeatures
from .nonlinear_features import NonlinearFeatures
from utils.constants import SAMPLING_RATE


class FeatureExtractor:
    """
    Unified feature extractor for sensor signals.

    Combines time-domain, frequency-domain, wavelet, and nonlinear features
    into a single comprehensive feature set.
    """

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        extract_time: bool = True,
        extract_frequency: bool = True,
        extract_wavelet: bool = True,
        extract_nonlinear: bool = False
    ):
        """
        Initialize the FeatureExtractor.

        Parameters
        ----------
        sampling_rate : int, optional
            Sampling rate in Hz (default: 64)
        extract_time : bool, optional
            Whether to extract time-domain features (default: True)
        extract_frequency : bool, optional
            Whether to extract frequency-domain features (default: True)
        extract_wavelet : bool, optional
            Whether to extract wavelet features (default: True)
        extract_nonlinear : bool, optional
            Whether to extract nonlinear features (default: True)
        """
        self.sampling_rate = sampling_rate
        self.extract_time = extract_time
        self.extract_frequency = extract_frequency
        self.extract_wavelet = extract_wavelet
        self.extract_nonlinear = extract_nonlinear

        # Initialize feature extractors
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures()
        self.wavelet_features = WaveletFeatures()
        self.nonlinear_features = NonlinearFeatures()

    def extract_from_signal(
        self,
        signal: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all enabled features from a single signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array)
        prefix : str, optional
            Prefix to add to feature names (default: '')

        Returns
        -------
        Dict[str, float]
            Dictionary of feature names and values

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_from_signal(signal, prefix='ankle_x_')
        """
        features = {}

        # Time-domain features
        if self.extract_time:
            time_feats = self.time_features.extract_all(signal, self.sampling_rate, prefix)
            features.update(time_feats)

        # Frequency-domain features
        if self.extract_frequency:
            freq_feats = self.freq_features.extract_all(signal, self.sampling_rate, prefix)
            features.update(freq_feats)

        # Wavelet features
        if self.extract_wavelet:
            wavelet_feats = self.wavelet_features.extract_all(signal, prefix)
            features.update(wavelet_feats)

        # Nonlinear features
        if self.extract_nonlinear:
            nonlinear_feats = self.nonlinear_features.extract_all(signal, prefix)
            features.update(nonlinear_feats)

        return features

    def extract_from_window(
        self,
        window: np.ndarray,
        include_magnitude: bool = True,
        channel_groups: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, float]:
        """
        Extract features from a multi-channel window.

        Parameters
        ----------
        window : np.ndarray
            Input window of shape (n_samples, n_channels)
        include_magnitude : bool, optional
            Whether to include magnitude features (default: True)
        channel_groups : dict, optional
            Groups of channel indices for per-group magnitude computation.
            Keys are group names (used as prefix), values are lists of indices.
            Example: {'acc_left_foot': [0,1,2], 'gyr_left_foot': [3,4,5]}
            If None, computes a single magnitude across all channels (legacy).

        Returns
        -------
        Dict[str, float]
            Dictionary of feature names and values

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_from_window(window)
        >>> # With channel groups (physically correct for mixed-unit signals):
        >>> groups = {'acc': [0,1,2], 'gyr': [3,4,5]}
        >>> features = extractor.extract_from_window(window, channel_groups=groups)
        """
        features = {}
        n_samples, n_channels = window.shape

        # Extract features for each channel
        for ch in range(n_channels):
            signal = window[:, ch]
            prefix = f'ch{ch}_'
            ch_features = self.extract_from_signal(signal, prefix)
            features.update(ch_features)

        # Extract magnitude features
        if include_magnitude:
            if channel_groups is not None:
                # Per-group magnitude: physically correct when channels have different units
                cadence_signal = None
                for group_name, indices in channel_groups.items():
                    group_data = window[:, indices]
                    magnitude = np.linalg.norm(group_data, axis=1)
                    mag_features = self.extract_from_signal(magnitude, f'{group_name}_mag_')
                    features.update(mag_features)
                    # Use first acc group for cadence (accelerometer is best for step detection)
                    if cadence_signal is None and group_name.startswith('acc'):
                        cadence_signal = magnitude
                # Fallback: use first group if no acc group found
                if cadence_signal is None:
                    first_indices = list(channel_groups.values())[0]
                    cadence_signal = np.linalg.norm(window[:, first_indices], axis=1)
                cadence = self.time_features.cadence_from_peaks(cadence_signal, self.sampling_rate)
                features['cadence'] = cadence
            else:
                # Legacy: single magnitude across all channels
                magnitude = np.linalg.norm(window, axis=1)
                mag_features = self.extract_from_signal(magnitude, 'mag_')
                features.update(mag_features)
                cadence = self.time_features.cadence_from_peaks(magnitude, self.sampling_rate)
                features['cadence'] = cadence

        return features

    def extract_from_windows(
        self,
        windows: np.ndarray,
        include_magnitude: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
        channel_groups: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame:
        """
        Extract features from multiple windows.

        Parameters
        ----------
        windows : np.ndarray
            Input windows of shape (n_windows, n_samples, n_channels)
        include_magnitude : bool, optional
            Whether to include magnitude features (default: True)
        verbose : bool, optional
            Whether to show progress (default: False)
        n_jobs : int, optional
            Number of parallel jobs. -1 uses all CPU cores (default: 1)
        channel_groups : dict, optional
            Groups of channel indices for per-group magnitude computation.
            Example: {'acc_left_foot': [0,1,2], 'gyr_left_foot': [3,4,5]}
            If None, computes a single magnitude across all channels (legacy).

        Returns
        -------
        pd.DataFrame
            DataFrame with features (rows=windows, columns=features)

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> features_df = extractor.extract_from_windows(windows)
        >>> print(features_df.shape)
        (n_windows, n_features)

        # With channel groups (physically correct for mixed-unit signals):
        >>> groups = {'acc_left_foot': [0,1,2], 'gyr_left_foot': [3,4,5]}
        >>> features_df = extractor.extract_from_windows(windows, channel_groups=groups)
        """
        if n_jobs == 1:
            if verbose:
                from tqdm import tqdm
                iterator = tqdm(windows, desc="Extracting features")
            else:
                iterator = windows

            feature_list = []
            for window in iterator:
                features = self.extract_from_window(window, include_magnitude, channel_groups)
                feature_list.append(features)
        else:
            from joblib import Parallel, delayed

            if verbose:
                print(f"🚀 Extracting features in parallel ({n_jobs} jobs)...")

            feature_list = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(self.extract_from_window)(window, include_magnitude, channel_groups)
                for window in windows
            )

        return pd.DataFrame(feature_list)

    def get_feature_names(
        self,
        n_channels: int = 9,
        include_magnitude: bool = True
    ) -> List[str]:
        """
        Get list of feature names that would be extracted.

        Parameters
        ----------
        n_channels : int, optional
            Number of channels (default: 9)
        include_magnitude : bool, optional
            Whether magnitude features are included (default: True)

        Returns
        -------
        List[str]
            List of feature names

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> feature_names = extractor.get_feature_names()
        >>> print(len(feature_names))
        # Returns number of features
        """
        # Extract from a dummy window to get feature names
        dummy_window = np.random.randn(256, n_channels)
        features = self.extract_from_window(dummy_window, include_magnitude)
        return list(features.keys())

    def get_config(self) -> Dict:
        """
        Get configuration of the feature extractor.

        Returns
        -------
        Dict
            Configuration dictionary

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> config = extractor.get_config()
        >>> print(config)
        {'sampling_rate': 64, 'extract_time': True, ...}
        """
        return {
            'sampling_rate': self.sampling_rate,
            'extract_time': self.extract_time,
            'extract_frequency': self.extract_frequency,
            'extract_wavelet': self.extract_wavelet,
            'extract_nonlinear': self.extract_nonlinear
        }
