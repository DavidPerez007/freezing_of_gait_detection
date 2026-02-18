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
        include_magnitude: bool = True
    ) -> Dict[str, float]:
        """
        Extract features from a multi-channel window.

        Parameters
        ----------
        window : np.ndarray
            Input window of shape (n_samples, n_channels)
        include_magnitude : bool, optional
            Whether to include magnitude (norm) of all channels (default: True)

        Returns
        -------
        Dict[str, float]
            Dictionary of feature names and values

        Examples
        --------
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_from_window(window)
        >>> print(len(features))
        # Returns many features (depending on n_channels and enabled extractors)
        """
        features = {}
        n_samples, n_channels = window.shape

        # Extract features for each channel
        for ch in range(n_channels):
            signal = window[:, ch]
            prefix = f'ch{ch}_'
            ch_features = self.extract_from_signal(signal, prefix)
            features.update(ch_features)

        # Extract features from magnitude (if enabled)
        if include_magnitude:
            magnitude = np.linalg.norm(window, axis=1)
            mag_features = self.extract_from_signal(magnitude, 'mag_')
            features.update(mag_features)

            # Add cadence estimation from magnitude
            cadence = self.time_features.cadence_from_peaks(magnitude, self.sampling_rate)
            features['cadence'] = cadence

        return features

    def extract_from_windows(
        self,
        windows: np.ndarray,
        include_magnitude: bool = True,
        verbose: bool = False
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
        """
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(windows, desc="Extracting features")
        else:
            iterator = windows

        feature_list = []
        for window in iterator:
            features = self.extract_from_window(window, include_magnitude)
            feature_list.append(features)

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
