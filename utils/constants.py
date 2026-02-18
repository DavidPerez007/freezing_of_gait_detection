"""
Constants and configuration values for Daphnet FoG Dataset processing.

This module centralizes all constant values used across the project,
including sampling rates, window configurations, sensor names, and file paths.
"""

from pathlib import Path
from typing import List

# ==============================================================================
# Sampling Configuration
# ==============================================================================

SAMPLING_RATE: int = 64  # Hz - Daphnet dataset sampling frequency


# ==============================================================================
# Window Configuration
# ==============================================================================

WINDOW_SIZE_SEC: float = 4.0  # seconds - Duration of sliding windows
WINDOW_OVERLAP: float = 0.5  # 50% overlap between consecutive windows
WINDOW_SIZE_SAMPLES: int = int(WINDOW_SIZE_SEC * SAMPLING_RATE)  # 256 samples
STEP_SIZE_SAMPLES: int = int(WINDOW_SIZE_SAMPLES * (1 - WINDOW_OVERLAP))  # 128 samples


# ==============================================================================
# Pre-FoG Configuration
# ==============================================================================

PRE_FOG_WINDOW_SEC: float = 0.5  # seconds - Duration of pre-FoG annotation window
PRE_FOG_WINDOW_SAMPLES: int = int(PRE_FOG_WINDOW_SEC * SAMPLING_RATE)  # 32 samples


# ==============================================================================
# Sensor Configuration
# ==============================================================================

SENSOR_NAMES: List[str] = ['ankle', 'thigh', 'trunk']
AXES: List[str] = ['forward', 'vertical', 'lateral']

# All feature column names (9 accelerometer channels)
FEATURE_COLUMNS: List[str] = [
    'ankle_acc_forward', 'ankle_acc_vertical', 'ankle_acc_lateral',
    'thigh_acc_forward', 'thigh_acc_vertical', 'thigh_acc_lateral',
    'trunk_acc_forward', 'trunk_acc_vertical', 'trunk_acc_lateral'
]


# ==============================================================================
# Label Configuration
# ==============================================================================

# Daphnet annotation values
ANNOTATION_NOT_EXPERIMENT: int = 0
ANNOTATION_NO_FREEZE: int = 1
ANNOTATION_FREEZE: int = 2

# Binary label values
BINARY_NO_FOG: int = 0
BINARY_FOG: int = 1

# Multiclass label values
MULTICLASS_NO_FOG: int = 0
MULTICLASS_FOG: int = 1
MULTICLASS_PRE_FOG: int = 2

# Label names for visualization
LABEL_NAMES_BINARY = {
    BINARY_NO_FOG: 'No FoG',
    BINARY_FOG: 'FoG'
}

LABEL_NAMES_MULTICLASS = {
    MULTICLASS_NO_FOG: 'No FoG',
    MULTICLASS_FOG: 'FoG',
    MULTICLASS_PRE_FOG: 'Pre-FoG'
}


# ==============================================================================
# File Paths
# ==============================================================================

# Base directories (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / 'Datasets'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Daphnet-specific paths
DAPHNET_RAW_DIR = DATASETS_DIR / 'Daphnet fog' / 'dataset'
DAPHNET_CSV_DIR = OUTPUTS_DIR / 'datasets_csv'
DAPHNET_FEATURES_DIR = OUTPUTS_DIR / 'daphnet_features'

# Dataset file names
DAPHNET_COMPLETE_CSV = 'daphnet_complete_dataset.csv'
DAPHNET_SEGMENTED_CSV = 'daphnet_segmented_dataset.csv'
DAPHNET_LOSO_BINARY_PKL = 'daphnet_loso_windows_binary.pkl'
DAPHNET_LOSO_MULTICLASS_PKL = 'daphnet_loso_windows_multiclass.pkl'


# ==============================================================================
# Signal Processing Configuration
# ==============================================================================

# Outlier detection
OUTLIER_THRESH_MULTIPLIER: float = 3.0  # MAD threshold multiplier
OUTLIER_POLY_ORDER: int = 3  # Polynomial order for interpolation

# Missing value handling
MISSING_VALUE_POLY_ORDER: int = 3  # Polynomial order for interpolation


# ==============================================================================
# Feature Extraction Configuration
# ==============================================================================

# Wavelet configuration
WAVELET_TYPE: str = 'db4'  # Daubechies 4
WAVELET_LEVEL: int = 3  # Decomposition level

# Frequency bands for PSD analysis
FREQ_BAND_LOCOMOTION: tuple = (0.5, 3.0)  # Hz
FREQ_BAND_FREEZING: tuple = (3.0, 8.0)  # Hz

# Sample Entropy configuration
SAMPEN_M: int = 2  # Embedding dimension
SAMPEN_R_MULTIPLIER: float = 0.2  # r = 0.2 * std

# Higuchi Fractal Dimension configuration
HIGUCHI_KMAX: int = 10  # Maximum k value


# ==============================================================================
# LOSO Configuration
# ==============================================================================

N_SUBJECTS: int = 10  # Number of subjects in Daphnet dataset
SUBJECT_IDS: List[str] = [f'S{i:02d}' for i in range(1, N_SUBJECTS + 1)]


# ==============================================================================
# Preprocessing Configuration
# ==============================================================================

# Scaling
SCALER_TYPE: str = 'standard'  # 'standard' or 'minmax'

# SMOTE
SMOTE_RANDOM_STATE: int = 42
SMOTE_K_NEIGHBORS: int = 5

# Imputation
IMPUTATION_STRATEGY: str = 'median'  # 'mean', 'median', 'most_frequent'
