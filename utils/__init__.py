"""
Utilities module for Daphnet FoG Dataset processing.

This module provides utility functions for I/O operations, visualization,
and constants used throughout the project.
"""

from .constants import *
from .io_helpers import save_pickle, load_pickle, ensure_output_dir
from .visualization import (
    plot_window_example,
    plot_fog_distribution_per_fold,
    plot_feature_correlation,
    plot_label_distribution,
    plot_subject_distribution
)

__all__ = [
    # Constants (imported from constants.py)
    'SAMPLING_RATE',
    'WINDOW_SIZE_SEC',
    'WINDOW_OVERLAP',
    'WINDOW_SIZE_SAMPLES',
    'STEP_SIZE_SAMPLES',
    'SENSOR_NAMES',
    'AXES',
    'FEATURE_COLUMNS',
    'ANNOTATION_NOT_EXPERIMENT',
    'ANNOTATION_NO_FREEZE',
    'ANNOTATION_FREEZE',
    'BINARY_NO_FOG',
    'BINARY_FOG',
    'LABEL_NAMES_BINARY',
    'PROJECT_ROOT',
    'DATASETS_DIR',
    'OUTPUTS_DIR',
    'DAPHNET_RAW_DIR',
    'DAPHNET_CSV_DIR',
    'DAPHNET_FEATURES_DIR',
    'DAPHNET_COMPLETE_CSV',
    'DAPHNET_SEGMENTED_CSV',
    'DAPHNET_LOSO_BINARY_PKL',
    'OUTLIER_THRESH_MULTIPLIER',
    'OUTLIER_POLY_ORDER',
    'MISSING_VALUE_POLY_ORDER',
    'WAVELET_TYPE',
    'WAVELET_LEVEL',
    'FREQ_BAND_LOCOMOTION',
    'FREQ_BAND_FREEZING',
    'SAMPEN_M',
    'SAMPEN_R_MULTIPLIER',
    'HIGUCHI_KMAX',
    'N_SUBJECTS',
    'SUBJECT_IDS',
    'SCALER_TYPE',
    'SMOTE_RANDOM_STATE',
    'SMOTE_K_NEIGHBORS',
    'IMPUTATION_STRATEGY',
    # I/O helpers
    'save_pickle',
    'load_pickle',
    'ensure_output_dir',
    # Visualization
    'plot_window_example',
    'plot_fog_distribution_per_fold',
    'plot_feature_correlation',
    'plot_label_distribution',
    'plot_subject_distribution',
]
