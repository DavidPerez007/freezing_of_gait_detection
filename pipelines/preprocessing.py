"""
Preprocessing pipeline for machine learning.

This module provides functions for scaling features, applying SMOTE,
and handling missing values in preparation for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
from typing import Tuple, Optional
from utils.constants import (
    SCALER_TYPE,
    SMOTE_RANDOM_STATE,
    SMOTE_K_NEIGHBORS
)


class DataPreprocessor:
    """
    Class for preprocessing data for machine learning.

    Handles scaling, SMOTE for class imbalance, and missing value imputation.
    """

    def __init__(
        self,
        scaler_type: str = SCALER_TYPE,
        smote_random_state: int = SMOTE_RANDOM_STATE,
        smote_k_neighbors: int = SMOTE_K_NEIGHBORS,
        poly_order: int = 3
    ):
        """
        Initialize the DataPreprocessor.

        Parameters
        ----------
        scaler_type : str, optional
            Type of scaler: 'standard' or 'minmax' (default: 'standard')
        smote_random_state : int, optional
            Random state for SMOTE (default: 42)
        smote_k_neighbors : int, optional
            Number of neighbors for SMOTE (default: 5)
        poly_order : int, optional
            Polynomial order for interpolation (default: 3)
        """
        self.scaler_type = scaler_type
        self.smote_random_state = smote_random_state
        self.smote_k_neighbors = smote_k_neighbors
        self.poly_order = poly_order

        self.scaler = None

    def create_scaler(self) -> StandardScaler | MinMaxScaler:
        """
        Create a scaler based on configuration.

        Returns
        -------
        StandardScaler | MinMaxScaler
            Configured scaler
        """
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def scale_features(
        self,
        X_train: np.ndarray | pd.DataFrame,
        X_test: np.ndarray | pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using training data statistics.

        IMPORTANT: Scaler is fit ONLY on training data to avoid data leakage.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame
            Training features
        X_test : np.ndarray | pd.DataFrame
            Test features

        Returns
        -------
        X_train_scaled : np.ndarray
            Scaled training features
        X_test_scaled : np.ndarray
            Scaled test features

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        """
        self.scaler = self.create_scaler()

        # Fit on train, transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def apply_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance training data.

        IMPORTANT: SMOTE is applied ONLY to training data to avoid data leakage.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        k_neighbors : int, optional
            Number of neighbors for SMOTE (uses instance default if None)
        random_state : int, optional
            Random state (uses instance default if None)

        Returns
        -------
        X_resampled : np.ndarray
            Resampled training features
        y_resampled : np.ndarray
            Resampled training labels

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> X_res, y_res = preprocessor.apply_smote(X_train, y_train)
        """
        if k_neighbors is None:
            k_neighbors = self.smote_k_neighbors
        if random_state is None:
            random_state = self.smote_random_state

        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

        try:
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  SMOTE failed: {e}")
            print("   Returning original data without resampling")
            return X_train, y_train

    def apply_adasyn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN to balance training data.

        ADASYN (Adaptive Synthetic Sampling) is more suitable for time-series data
        as it focuses on harder-to-learn examples near decision boundaries.

        IMPORTANT: ADASYN is applied ONLY to training data to avoid data leakage.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        k_neighbors : int, optional
            Number of neighbors for ADASYN (uses instance default if None)
        random_state : int, optional
            Random state (uses instance default if None)

        Returns
        -------
        X_resampled : np.ndarray
            Resampled training features
        y_resampled : np.ndarray
            Resampled training labels

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> X_res, y_res = preprocessor.apply_adasyn(X_train, y_train)
        """
        if k_neighbors is None:
            k_neighbors = self.smote_k_neighbors
        if random_state is None:
            random_state = self.smote_random_state

        adasyn = ADASYN(n_neighbors=k_neighbors, random_state=random_state)

        try:
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  ADASYN failed: {e}")
            print("   Returning original data without resampling")
            return X_train, y_train

    def handle_missing_values(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        poly_order: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values using polynomial interpolation.

        For each feature, fits a polynomial using non-NaN values and interpolates
        missing values. More suitable for time-series data than median imputation.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features
        X_test : pd.DataFrame | np.ndarray
            Test features
        poly_order : int, optional
            Polynomial order for interpolation (default: 3)

        Returns
        -------
        X_train_imputed : np.ndarray
            Imputed training features
        X_test_imputed : np.ndarray
            Imputed test features

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> X_train_imp, X_test_imp = preprocessor.handle_missing_values(X_train, X_test)
        """
        # Convert to arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Replace infinities with NaN
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)

        # Interpolate each dataset independently
        X_train_imputed = self._interpolate_missing(X_train.copy(), poly_order)
        X_test_imputed = self._interpolate_missing(X_test.copy(), poly_order)

        return X_train_imputed, X_test_imputed

    def _interpolate_missing(
        self,
        X: np.ndarray,
        poly_order: int = 3
    ) -> np.ndarray:
        """
        Interpolate missing values in a feature matrix using polynomial interpolation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        poly_order : int
            Polynomial order for interpolation

        Returns
        -------
        np.ndarray
            Feature matrix with interpolated values
        """
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            feature = X[:, feat_idx]
            nan_mask = np.isnan(feature)

            if not np.any(nan_mask):
                continue  # No missing values in this feature

            # Get indices
            idx = np.arange(n_samples)
            good_idx = idx[~nan_mask]
            bad_idx = idx[nan_mask]

            if len(good_idx) < 2:
                # Not enough non-NaN values, use median of non-NaN values
                if len(good_idx) > 0:
                    feature[bad_idx] = np.nanmedian(feature[good_idx])
                else:
                    feature[bad_idx] = 0.0
            else:
                # Polynomial interpolation
                deg = min(poly_order, len(good_idx) - 1)
                try:
                    coeffs = np.polyfit(good_idx, feature[good_idx], deg)
                    feature[bad_idx] = np.polyval(coeffs, bad_idx)
                except Exception:
                    # Fallback to linear interpolation
                    try:
                        feature[bad_idx] = np.interp(bad_idx, good_idx, feature[good_idx])
                    except Exception:
                        # Last resort: use median
                        feature[bad_idx] = np.nanmedian(feature[good_idx]) if len(good_idx) > 0 else 0.0

            X[:, feat_idx] = feature

        return X

    def preprocess_fold(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        resampling_method: str = 'adasyn',
        handle_missing: bool = True,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for a single fold.

        Steps (in order):
        1. Handle missing values (if enabled)
        2. Scale features (if enabled)
        3. Apply resampling (if enabled, only on train)

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_test : pd.DataFrame | np.ndarray
            Test features
        resampling_method : str, optional
            Resampling method: 'adasyn', 'smote', or 'none' (default: 'adasyn')
        handle_missing : bool, optional
            Whether to handle missing values (default: True)
        scale : bool, optional
            Whether to scale features (default: True)

        Returns
        -------
        X_train_processed : np.ndarray
            Processed training features
        y_train_processed : np.ndarray
            Processed training labels (resampled if method applied)
        X_test_processed : np.ndarray
            Processed test features

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> X_train_proc, y_train_proc, X_test_proc = preprocessor.preprocess_fold(
        ...     X_train, y_train, X_test, resampling_method='adasyn'
        ... )
        """
        # Step 1: Handle missing values
        if handle_missing:
            X_train_proc, X_test_proc = self.handle_missing_values(X_train, X_test)
        else:
            X_train_proc = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_proc = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        # Step 2: Scale features
        if scale:
            X_train_proc, X_test_proc = self.scale_features(X_train_proc, X_test_proc)

        # Step 3: Apply resampling (only on train)
        if resampling_method == 'adasyn':
            X_train_proc, y_train_proc = self.apply_adasyn(X_train_proc, y_train)
        elif resampling_method == 'smote':
            X_train_proc, y_train_proc = self.apply_smote(X_train_proc, y_train)
        elif resampling_method == 'none':
            y_train_proc = y_train
        else:
            raise ValueError(f"Unknown resampling method: {resampling_method}")

        return X_train_proc, y_train_proc, X_test_proc

    def get_scaler(self):
        """Get the fitted scaler (for later use)."""
        return self.scaler
