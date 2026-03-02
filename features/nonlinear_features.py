"""
Nonlinear feature extraction for sensor signals.

This module provides functions for extracting complexity and chaos-related
features including sample entropy and Higuchi fractal dimension.
"""

import numpy as np
from typing import Dict
from utils.constants import SAMPEN_M, SAMPEN_R_MULTIPLIER, HIGUCHI_KMAX


class NonlinearFeatures:
    """
    Class for extracting nonlinear/complexity features from sensor signals.

    Includes entropy measures and fractal dimension calculations.
    """

    @staticmethod
    def sample_entropy(
        signal: np.ndarray,
        m: int = SAMPEN_M,
        r: float = None
    ) -> float:
        """
        Calculate Sample Entropy (SampEn) of a signal.

        Sample Entropy measures the complexity and regularity of a time series.
        Lower values indicate more regular/predictable patterns.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        m : int, optional
            Embedding dimension (default: 2)
        r : float, optional
            Tolerance (default: 0.2 * std(signal))

        Returns
        -------
        float
            Sample entropy value

        Notes
        -----
        SampEn = -log(A/B) where:
        - A: number of template matches of length m+1
        - B: number of template matches of length m

        Higher values indicate more complexity/irregularity.

        References
        ----------
        Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
        analysis using approximate entropy and sample entropy.
        """
        x = np.array(signal)
        N = len(x)

        if N <= m + 1:
            return np.nan

        if r is None:
            r = SAMPEN_R_MULTIPLIER * np.std(x)

        def _phi(m_val):
            """Helper function to calculate template matches."""
            # Create templates of length m_val
            templates = np.array([x[i:i+m_val] for i in range(N - m_val + 1)])

            # Count matches
            count = 0
            n_templates = len(templates)

            for i in range(n_templates):
                # Calculate maximum distance to all other templates
                distances = np.max(np.abs(templates - templates[i]), axis=1)

                # Count templates within tolerance r (excluding self-match)
                count += np.sum(distances <= r) - 1

            # Normalize by total number of comparisons
            denom = n_templates * (n_templates - 1)
            return count / denom if denom > 0 else 0

        try:
            phi_m = _phi(m)
            phi_m_plus_1 = _phi(m + 1)

            if phi_m > 0 and phi_m_plus_1 > 0:
                sampen = -np.log(phi_m_plus_1 / phi_m)
                return float(sampen)
            else:
                return np.nan
        except Exception:
            return np.nan

    @staticmethod
    def higuchi_fractal_dimension(
        signal: np.ndarray,
        kmax: int = HIGUCHI_KMAX
    ) -> float:
        """
        Calculate Higuchi Fractal Dimension of a signal.

        Fractal dimension measures the complexity/roughness of a time series.
        Values typically range from 1 (smooth) to 2 (very rough/complex).

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        kmax : int, optional
            Maximum k value (default: 10)

        Returns
        -------
        float
            Higuchi fractal dimension

        Notes
        -----
        Higher values indicate more complex/irregular signals.

        References
        ----------
        Higuchi, T. (1988). Approach to an irregular time series on the
        basis of the fractal theory. Physica D: Nonlinear Phenomena.
        """
        x = np.asarray(signal)
        N = x.size

        if N < 4:
            return np.nan

        L = []  # Length of curve for each k

        for k in range(1, kmax + 1):
            Lk = 0.0

            for m in range(k):
                # Get subsequence starting at m with step k
                indices = np.arange(m, N, k)

                if indices.size < 2:
                    continue

                # Calculate length of curve for this m
                Lm = np.sum(np.abs(np.diff(x[indices]))) * (N - 1) / ((indices.size - 1) * k)
                Lk += Lm

            # Average length for this k
            L.append(Lk / k)

        L = np.array(L)

        if len(L) < 2:
            return np.nan

        # Fit log-log plot to estimate fractal dimension
        k_values = np.arange(1, len(L) + 1)

        try:
            # L(k) ~ k^(-FD)  →  log(L) = -FD·log(k) + const  →  FD = -slope
            coeffs = np.polyfit(np.log(k_values), np.log(L), 1)
            fractal_dim = -coeffs[0]  # Negate slope to get positive FD
            return float(fractal_dim)
        except Exception:
            return np.nan

    @staticmethod
    def approximate_entropy(
        signal: np.ndarray,
        m: int = 2,
        r: float = None
    ) -> float:
        """
        Calculate Approximate Entropy (ApEn) of a signal.

        Similar to Sample Entropy but includes self-matches.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        m : int, optional
            Embedding dimension (default: 2)
        r : float, optional
            Tolerance (default: 0.2 * std(signal))

        Returns
        -------
        float
            Approximate entropy value
        """
        x = np.array(signal)
        N = len(x)

        if N <= m:
            return np.nan

        if r is None:
            r = 0.2 * np.std(x)

        def _phi(m_val):
            """Helper function."""
            templates = np.array([x[i:i+m_val] for i in range(N - m_val + 1)])
            n_templates = len(templates)

            phi_sum = 0.0
            for i in range(n_templates):
                distances = np.max(np.abs(templates - templates[i]), axis=1)
                # Count matches (including self-match)
                matches = np.sum(distances <= r)
                if matches > 0:
                    phi_sum += np.log(matches / n_templates)

            return phi_sum / n_templates

        try:
            apen = _phi(m) - _phi(m + 1)
            return float(apen)
        except Exception:
            return np.nan

    @classmethod
    def extract_all(
        cls,
        signal: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all nonlinear features from a signal.

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
        >>> features = NonlinearFeatures.extract_all(signal, prefix='ankle_x_')
        >>> print(features.keys())
        dict_keys(['ankle_x_sample_entropy', 'ankle_x_higuchi_fd', ...])
        """
        features = {}

        # Entropy measures
        features[f'{prefix}sample_entropy'] = cls.sample_entropy(signal)
        features[f'{prefix}approx_entropy'] = cls.approximate_entropy(signal)

        # Fractal dimension
        features[f'{prefix}higuchi_fd'] = cls.higuchi_fractal_dimension(signal)

        return features
