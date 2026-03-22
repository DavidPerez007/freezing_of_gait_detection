"""
Wavelet-based feature extraction for sensor signals.

This module provides functions for extracting time-frequency features
using Discrete Wavelet Transform (DWT).
"""

import numpy as np
import pywt
from typing import Dict, List
from utils.constants import WAVELET_TYPE, WAVELET_LEVEL


class WaveletFeatures:
    """
    Class for extracting wavelet-based features from sensor signals.

    Uses Discrete Wavelet Transform for multi-resolution analysis.
    """

    def __init__(
        self,
        wavelet: str = WAVELET_TYPE,
        level: int = WAVELET_LEVEL
    ):
        """
        Initialize the WaveletFeatures extractor.

        Parameters
        ----------
        wavelet : str, optional
            Wavelet type (default: 'db4' - Daubechies 4)
        level : int, optional
            Decomposition level (default: 3)
        """
        self.wavelet = wavelet
        self.level = level

    def decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Perform wavelet decomposition.

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        List[np.ndarray]
            List of wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
            where cA is approximation coefficients and cD is detail coefficients

        Examples
        --------
        >>> wf = WaveletFeatures()
        >>> coeffs = wf.decompose(signal)
        >>> print(len(coeffs))  # level + 1
        4
        """
        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            return coeffs
        except Exception:
            # Return empty coefficients if decomposition fails
            return [np.array([]) for _ in range(self.level + 1)]

    def wavelet_energies(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy of wavelet coefficients at each level.

        Energy at level i = sum(|coeffs_i|^2)

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        Dict[str, float]
            Dictionary with energy at each level
            Keys: 'wavelet_energy_0', 'wavelet_energy_1', etc.
        """
        coeffs = self.decompose(signal)
        energies = {}

        for i, coeff in enumerate(coeffs):
            if coeff.size > 0:
                energy = float(np.sum(np.array(coeff)**2))
            else:
                energy = np.nan
            energies[f'wavelet_energy_{i}'] = energy

        return energies

    def wavelet_entropy(self, signal: np.ndarray) -> float:
        """
        Calculate wavelet entropy.

        Wavelet entropy measures the disorder/complexity in the signal
        using the energy distribution across decomposition levels.

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        float
            Wavelet entropy value

        Notes
        -----
        E_j = sum(c_j²) per level j; p_j = E_j / sum(E_j);
        Entropy = -sum(p_j * log(p_j))  (Rosso et al., 2001)
        """
        coeffs = self.decompose(signal)

        # Per-level energy: E_j = sum(c_j²)  (Rosso et al., 2001)
        energies = np.array([np.sum(c ** 2) for c in coeffs if c.size > 0])

        if energies.size == 0:
            return np.nan

        total = energies.sum()
        if total == 0:
            return np.nan

        # Probability distribution over levels; entropy over non-zero terms
        prob = energies / total
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob + 1e-12))

        return float(entropy)

    def wavelet_variance(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Calculate variance of wavelet coefficients at each level.

        Parameters
        ----------
        signal : np.ndarray
            Input signal

        Returns
        -------
        Dict[str, float]
            Dictionary with variance at each level
        """
        coeffs = self.decompose(signal)
        variances = {}

        for i, coeff in enumerate(coeffs):
            if coeff.size > 0:
                variance = float(np.var(coeff))
            else:
                variance = np.nan
            variances[f'wavelet_var_{i}'] = variance

        return variances

    def extract_all(
        self,
        signal: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all wavelet features from a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        prefix : str, optional
            Prefix to add to feature names (default: '')

        Returns
        -------
        Dict[str, float]
            Dictionary of feature names and values

        Examples
        --------
        >>> wf = WaveletFeatures()
        >>> features = wf.extract_all(signal, prefix='ankle_x_')
        >>> print(features.keys())
        dict_keys(['ankle_x_wavelet_energy_0', 'ankle_x_wavelet_energy_1', ...])
        """
        features = {}

        # Wavelet energies
        energies = self.wavelet_energies(signal)
        for key, value in energies.items():
            features[f'{prefix}{key}'] = value

        # Wavelet entropy
        features[f'{prefix}wavelet_entropy'] = self.wavelet_entropy(signal)

        # Wavelet variances
        variances = self.wavelet_variance(signal)
        for key, value in variances.items():
            features[f'{prefix}{key}'] = value

        return features
