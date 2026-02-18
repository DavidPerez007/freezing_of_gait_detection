"""
Leave-One-Subject-Out (LOSO) pipeline for FoG detection.

This module provides a complete pipeline for LOSO cross-validation including
signal cleaning, feature extraction, and preprocessing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import LeaveOneGroupOut

from processing.signal_cleaning import SignalCleaner
from features.extractors import FeatureExtractor
from pipelines.preprocessing import DataPreprocessor
from utils.io_helpers import save_pickle, ensure_output_dir
from utils.constants import DAPHNET_FEATURES_DIR


class LOSOPipeline:
    """
    Complete Leave-One-Subject-Out cross-validation pipeline.

    Handles the entire workflow from raw windows to preprocessed,
    feature-extracted, and SMOTE-resampled data for each fold.
    """

    def __init__(
        self,
        output_dir: Path = DAPHNET_FEATURES_DIR,
        clean_signals: bool = True,
        extract_features: bool = True,
        apply_preprocessing: bool = True
    ):
        """
        Initialize the LOSO pipeline.

        Parameters
        ----------
        output_dir : Path, optional
            Directory to save processed data (default: from constants)
        clean_signals : bool, optional
            Whether to clean signals before feature extraction (default: True)
        extract_features : bool, optional
            Whether to extract features (default: True)
        apply_preprocessing : bool, optional
            Whether to apply preprocessing (scaling, SMOTE) (default: True)
        """
        self.output_dir = ensure_output_dir(output_dir)
        self.clean_signals = clean_signals
        self.extract_features = extract_features
        self.apply_preprocessing = apply_preprocessing

        # Initialize components
        self.cleaner = SignalCleaner()
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = DataPreprocessor()

    def create_splits(
        self,
        df: pd.DataFrame,
        subject_col: str = 'subject'
    ) -> List[Dict]:
        """
        Create LOSO splits from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with subject information
        subject_col : str, optional
            Name of subject column (default: 'subject')

        Returns
        -------
        List[Dict]
            List of split dictionaries, each containing:
            - fold: Fold number
            - test_subject: Subject ID used for testing
            - train_idx: Indices for training
            - test_idx: Indices for testing

        Examples
        --------
        >>> pipeline = LOSOPipeline()
        >>> splits = pipeline.create_splits(df)
        >>> print(len(splits))  # Number of subjects
        10
        """
        groups = df[subject_col].values
        logo = LeaveOneGroupOut()

        splits = []
        for fold, (train_idx, test_idx) in enumerate(logo.split(df, groups=groups)):
            test_subject = groups[test_idx][0]

            split = {
                'fold': fold,
                'test_subject': test_subject,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            }
            splits.append(split)

        return splits

    def process_windows(
        self,
        windows: np.ndarray,
        clean: bool = True
    ) -> np.ndarray:
        """
        Clean windows if enabled.

        Parameters
        ----------
        windows : np.ndarray
            Input windows (n_windows, n_samples, n_channels)
        clean : bool, optional
            Whether to clean (default: uses instance setting)

        Returns
        -------
        np.ndarray
            Cleaned or original windows
        """
        if clean and self.clean_signals:
            return self.cleaner.clean_windows(windows)
        else:
            return windows

    def extract_features_from_windows(
        self,
        windows: np.ndarray,
        extract: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Extract features from windows if enabled.

        Parameters
        ----------
        windows : np.ndarray
            Input windows
        extract : bool, optional
            Whether to extract features (default: uses instance setting)
        verbose : bool, optional
            Whether to show progress (default: False)

        Returns
        -------
        pd.DataFrame
            Extracted features or empty DataFrame
        """
        if extract and self.extract_features:
            return self.feature_extractor.extract_from_windows(windows, verbose=verbose)
        else:
            # Return windows flattened if no feature extraction
            n_windows = windows.shape[0]
            return pd.DataFrame(windows.reshape(n_windows, -1))

    def process_fold(
        self,
        X_train_windows: np.ndarray,
        y_train: np.ndarray,
        X_test_windows: np.ndarray,
        y_test: np.ndarray,
        test_subject: str,
        verbose: bool = False
    ) -> Dict:
        """
        Process a single LOSO fold completely.

        Steps:
        1. Clean windows (train and test)
        2. Extract features
        3. Apply preprocessing (imputation, scaling, SMOTE on train only)

        Parameters
        ----------
        X_train_windows : np.ndarray
            Training windows
        y_train : np.ndarray
            Training labels
        X_test_windows : np.ndarray
            Test windows
        y_test : np.ndarray
            Test labels
        test_subject : str
            Subject ID for this test fold
        verbose : bool, optional
            Whether to show progress (default: False)

        Returns
        -------
        Dict
            Dictionary with processed data:
            - X_train_features: Training features (DataFrame)
            - X_test_features: Test features (DataFrame)
            - X_train_processed: Preprocessed training features (ndarray)
            - y_train_processed: Preprocessed training labels (ndarray)
            - X_test_processed: Preprocessed test features (ndarray)
            - y_test: Test labels (ndarray)
            - test_subject: Subject ID

        Examples
        --------
        >>> pipeline = LOSOPipeline()
        >>> fold_data = pipeline.process_fold(X_train, y_train, X_test, y_test, 'S01')
        """
        if verbose:
            print(f"\n🔄 Processing fold: Test Subject = {test_subject}")

        # Step 1: Clean windows
        if verbose:
            print("  1/3 Cleaning signals...")
        X_train_clean = self.process_windows(X_train_windows)
        X_test_clean = self.process_windows(X_test_windows)

        # Step 2: Extract features
        if verbose:
            print("  2/3 Extracting features...")
        X_train_features = self.extract_features_from_windows(X_train_clean, verbose=False)
        X_test_features = self.extract_features_from_windows(X_test_clean, verbose=False)

        # Step 3: Preprocess
        if verbose:
            print("  3/3 Preprocessing (scaling + SMOTE)...")

        if self.apply_preprocessing:
            X_train_proc, y_train_proc, X_test_proc = self.preprocessor.preprocess_fold(
                X_train_features, y_train, X_test_features
            )
        else:
            X_train_proc = X_train_features.values
            y_train_proc = y_train
            X_test_proc = X_test_features.values

        if verbose:
            print(f"  ✅ Done! Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")

        return {
            'X_train_features': X_train_features,
            'X_test_features': X_test_features,
            'X_train_processed': X_train_proc,
            'y_train_processed': y_train_proc,
            'X_test_processed': X_test_proc,
            'y_test': y_test,
            'test_subject': test_subject
        }

    def save_fold_data(
        self,
        fold_data: Dict,
        fold_number: int,
        save_features: bool = True,
        save_processed: bool = True
    ) -> None:
        """
        Save processed fold data to disk.

        Parameters
        ----------
        fold_data : Dict
            Processed fold data from process_fold()
        fold_number : int
            Fold number
        save_features : bool, optional
            Whether to save raw features (default: True)
        save_processed : bool, optional
            Whether to save processed data (default: True)

        Examples
        --------
        >>> pipeline = LOSOPipeline()
        >>> pipeline.save_fold_data(fold_data, fold_number=0)
        """
        test_subject = fold_data['test_subject']
        fold_dir = ensure_output_dir(self.output_dir / f'fold_subj_{test_subject}')

        # Save raw features
        if save_features:
            fold_data['X_train_features'].to_csv(fold_dir / 'X_train_features.csv', index=False)
            fold_data['X_test_features'].to_csv(fold_dir / 'X_test_features.csv', index=False)
            pd.Series(fold_data['y_test'], name='label').to_csv(fold_dir / 'y_test.csv', index=False)

        # Save processed data
        if save_processed:
            feature_cols = fold_data['X_train_features'].columns
            pd.DataFrame(
                fold_data['X_train_processed'],
                columns=feature_cols
            ).to_csv(fold_dir / 'X_train_resampled.csv', index=False)

            pd.Series(
                fold_data['y_train_processed'],
                name='label'
            ).to_csv(fold_dir / 'y_train_resampled.csv', index=False)

            pd.DataFrame(
                fold_data['X_test_processed'],
                columns=feature_cols
            ).to_csv(fold_dir / 'X_test_scaled.csv', index=False)

        print(f"✅ Saved fold {fold_number} data to: {fold_dir}")

    def run_pipeline(
        self,
        loso_window_splits: List[Dict],
        verbose: bool = True,
        save_data: bool = True
    ) -> List[Dict]:
        """
        Run complete LOSO pipeline on all folds.

        Parameters
        ----------
        loso_window_splits : List[Dict]
            List of LOSO window splits, each containing:
            - fold: Fold number
            - test_subject: Subject ID
            - X_train: Training windows
            - X_test: Test windows
            - y_train: Training labels
            - y_test: Test labels
        verbose : bool, optional
            Whether to show progress (default: True)
        save_data : bool, optional
            Whether to save processed data (default: True)

        Returns
        -------
        List[Dict]
            List of processed fold data

        Examples
        --------
        >>> pipeline = LOSOPipeline()
        >>> results = pipeline.run_pipeline(loso_splits)
        """
        processed_folds = []

        for split in loso_window_splits:
            fold_data = self.process_fold(
                split['X_train'],
                split['y_train'],
                split['X_test'],
                split['y_test'],
                split['test_subject'],
                verbose=verbose
            )

            if save_data:
                self.save_fold_data(fold_data, split['fold'])

            processed_folds.append(fold_data)

        if verbose:
            print(f"\n🎉 Pipeline complete! Processed {len(processed_folds)} folds.")

        return processed_folds
