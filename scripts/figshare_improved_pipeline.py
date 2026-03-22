#!/usr/bin/env python3
"""
Figshare Freezing of Gait (FoG) Detection — Improved Pipeline
==============================================================
Fixes ALL identified issues from diagnostic analysis:
1. Handles 12/35 monoclase subjects properly in LOSO metrics
2. Single z-normalization (signal level only) — no double normalization
3. Adds RobustScaler in pipeline
4. Feature selection increased to k=80 (from k=12/sensor)
5. VarianceThreshold removes constant features before SelectKBest
6. Bandpass 0.5-25 Hz for 128 Hz data
7. Exploits gyroscope (more discriminative than accelerometer per literature)
8. Computes derived signals: magnitudes, jerk, Freeze Index from both acc & gyr

Compares 6 classifiers and 3 fusion techniques:
  1. Feature-level fusion (all sensors + derived signals)
  2. Decision-level stacking (acc-model + gyr-model + combined -> LR meta)
  3. Weighted soft voting (optimised weights)

Usage: python scripts/figshare_improved_pipeline.py
"""
from __future__ import annotations

import sys, os, time, json, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ── Project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loaders import FigshareDatasetLoader
from features.extractors import FeatureExtractor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("figshare")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FS = 128  # Figshare sampling rate
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)  # 512
TRAIN_OVERLAP = 0.50
TEST_OVERLAP = 0.0
LABEL_THRESH = 0.50  # majority voting
BP_LOW, BP_HIGH, BP_ORDER = 0.5, 25.0, 4
NPERSEG = min(256, WINDOW_SAMPLES)
K_FEATURES = 80
SEED = 42
N_INNER_CV = 3
N_SEARCH_ITER = 20

# Figshare columns
ACC_COLS = ["acc_ml_lower_back", "acc_ap_lower_back", "acc_si_lower_back"]
GYR_COLS = ["gyr_ml_lower_back", "gyr_ap_lower_back", "gyr_si_lower_back"]
ALL_SENSOR_COLS = ACC_COLS + GYR_COLS
FOG_COL = "freezing_flag"

DATASET_PATH = PROJECT_ROOT / "Datasets" / "Figshare a public dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figshare_improved_results"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def interpolate_missing(data: np.ndarray) -> np.ndarray:
    """Fill NaN values per channel using cubic spline interpolation.
    Falls back to linear for short segments, ffill/bfill for edges."""
    from scipy.interpolate import CubicSpline
    data = data.copy()
    if data.ndim == 1:
        mask = np.isnan(data)
        if not mask.any():
            return data
        good = np.flatnonzero(~mask)
        bad = np.flatnonzero(mask)
        if len(good) == 0:
            data[:] = 0.0
            return data
        if len(good) >= 4:
            cs = CubicSpline(good, data[good], extrapolate=True)
            data[bad] = cs(bad)
        elif len(good) >= 2:
            data[bad] = np.interp(bad, good, data[good])
        else:
            data[mask] = data[good[0]]
        return data
    for col in range(data.shape[1]):
        data[:, col] = interpolate_missing(data[:, col])
    return data


def bandpass_filter(data: np.ndarray, fs: int = FS, low: float = BP_LOW,
                    high: float = BP_HIGH, order: int = BP_ORDER) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(np.float64)
    return np.column_stack([sosfiltfilt(sos, data[:, i]) for i in range(data.shape[1])])


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Per-trial robust z-score normalization (median / MAD)."""
    med = np.nanmedian(data, axis=0)
    mad = np.nanmedian(np.abs(data - med), axis=0) * 1.4826
    mad[mad < 1e-10] = 1.0
    return (data - med) / mad


def create_windows(signal: np.ndarray, labels: np.ndarray, win_samples: int,
                   overlap: float, label_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window segmentation with majority-vote labelling."""
    step = max(1, int(win_samples * (1 - overlap)))
    n = len(signal)
    windows, win_labels = [], []
    for start in range(0, n - win_samples + 1, step):
        end = start + win_samples
        w = signal[start:end]
        lbl_slice = labels[start:end]
        if np.any(np.isnan(w)):
            continue
        fog_ratio = np.mean(lbl_slice)
        win_labels.append(1 if fog_ratio > label_threshold else 0)
        windows.append(w)
    if not windows:
        return np.empty((0, win_samples, signal.shape[1])), np.empty(0)
    return np.array(windows), np.array(win_labels)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold via Youden's J statistic."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess() -> Dict[int, Dict]:
    """Load Figshare data, preprocess per subject/trial."""
    log.info("Loading Figshare dataset from %s", DATASET_PATH)
    loader = FigshareDatasetLoader(str(DATASET_PATH))
    df = loader.load_all_data(verbose=False, trial_type="walking")  # Walking trials only

    log.info("Loaded %d samples from %d subjects", len(df), df["subject_id"].nunique())

    subjects_data = {}
    for sid in sorted(df["subject_id"].unique()):
        sub_df = df[df["subject_id"] == sid]
        all_windows, all_labels = [], []

        for sess_id in sorted(sub_df["session_id"].unique()):
            trial = sub_df[sub_df["session_id"] == sess_id]

            if FOG_COL not in trial.columns:
                continue

            signal_raw = trial[ALL_SENSOR_COLS].values.astype(np.float64)
            labels_raw = trial[FOG_COL].values.astype(np.float64)

            # Skip trials with all NaN
            if np.all(np.isnan(signal_raw)):
                continue

            # Interpolate missing values before filtering
            signal_raw = interpolate_missing(signal_raw)
            labels_raw = np.nan_to_num(labels_raw, nan=0.0)

            # Bandpass filter
            try:
                signal_filt = bandpass_filter(signal_raw, FS)
            except Exception:
                continue

            # Compute derived signals
            acc = signal_filt[:, :3]
            gyr = signal_filt[:, 3:]

            acc_mag = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
            gyr_mag = np.sqrt(np.sum(gyr ** 2, axis=1, keepdims=True))

            # Jerk (derivative of acceleration)
            jerk = np.gradient(acc, 1.0 / FS, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk ** 2, axis=1, keepdims=True))

            # Combine: 6 original + acc_mag + gyr_mag + jerk_mag = 9 channels
            signal_full = np.hstack([signal_filt, acc_mag, gyr_mag, jerk_mag])

            # Z-score normalize per trial (ONCE only)
            signal_norm = zscore_normalize(signal_full)

            # Create windows
            w, l = create_windows(signal_norm, labels_raw, WINDOW_SAMPLES, TRAIN_OVERLAP, LABEL_THRESH)
            if len(w) > 0:
                all_windows.append(w)
                all_labels.append(l)

        if all_windows:
            subjects_data[sid] = {
                "windows": np.concatenate(all_windows, axis=0),
                "labels": np.concatenate(all_labels, axis=0),
            }
            n_fog = int(np.sum(subjects_data[sid]["labels"]))
            n_total = len(subjects_data[sid]["labels"])
            log.info("  Subject S%02d: %d windows (%d FoG, %d NoFoG)",
                     sid, n_total, n_fog, n_total - n_fog)

    return subjects_data


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_features_for_subject(windows: np.ndarray, fs: int = FS) -> pd.DataFrame:
    """Extract features from all windows of a subject."""
    extractor = FeatureExtractor(
        sampling_rate=fs,
        extract_time=True,
        extract_frequency=True,
        extract_wavelet=True,
        extract_nonlinear=False,
    )
    rows = []
    for i in range(len(windows)):
        feats = extractor.extract_from_window(windows[i], include_magnitude=False,
                                               channel_groups=None)
        rows.append(feats)
    return pd.DataFrame(rows)


def extract_all_features(subjects_data: Dict) -> Dict[int, Dict]:
    """Extract features for all subjects in parallel."""
    log.info("Extracting features for %d subjects (parallel)...", len(subjects_data))

    def _extract(sid):
        feat_df = extract_features_for_subject(subjects_data[sid]["windows"])
        return sid, feat_df

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_extract)(sid) for sid in tqdm(subjects_data.keys(), desc="Feature extraction")
    )

    features = {}
    for sid, feat_df in results:
        features[sid] = {"X": feat_df, "y": subjects_data[sid]["labels"]}
        log.info("  S%02d: %d windows, %d features", sid, len(feat_df), feat_df.shape[1])
    return features


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIERS & HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════════════════════════
def get_classifiers() -> Dict[str, Any]:
    clfs = {
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
        "LogisticReg": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=SEED),
        "MLP": MLPClassifier(max_iter=1000, early_stopping=True, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
    }
    if HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=SEED,
                                         n_jobs=1, verbosity=0)
    return clfs


def get_param_grids() -> Dict[str, Dict]:
    grids = {
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", "log2"],
        },
        "LogisticReg": {"C": [0.01, 0.1, 1, 10]},
        "SVM": {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
        "MLP": {
            "hidden_layer_sizes": [(100, 50), (200, 100), (100, 50, 25)],
            "alpha": [0.0001, 0.001],
        },
        "AdaBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    }
    if HAS_XGB:
        grids["XGBoost"] = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [3, 5, 7],
        }
    return grids


# ══════════════════════════════════════════════════════════════════════════════
# LOSO EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def prepare_fold(features: Dict, test_sid: int):
    X_trains, y_trains = [], []
    for sid, data in features.items():
        if sid == test_sid:
            continue
        X_trains.append(data["X"])
        y_trains.append(data["y"])
    X_train = pd.concat(X_trains, ignore_index=True)
    y_train = np.concatenate(y_trains)
    X_test = features[test_sid]["X"].copy()
    y_test = features[test_sid]["y"].copy()
    return X_train, y_train, X_test, y_test


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: np.ndarray, k: int = K_FEATURES):
    """Clean, scale, select features."""
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Remove constant features
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = pd.DataFrame(vt.fit_transform(X_train), columns=X_train.columns[vt.get_support()])
    X_test_vt = pd.DataFrame(vt.transform(X_test), columns=X_train_vt.columns)

    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_vt), columns=X_train_vt.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_vt), columns=X_train_vt.columns)

    # Scale
    scaler = RobustScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test_imp), columns=X_train_imp.columns)

    # Feature selection
    k_actual = min(k, X_train_sc.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k_actual)
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_test_sel = selector.transform(X_test_sc)

    selected_cols = X_train_sc.columns[selector.get_support()].tolist()
    return X_train_sel, X_test_sel, selected_cols, {"vt": vt, "imputer": imputer, "scaler": scaler, "selector": selector}


def train_and_evaluate_classifier(clf_name, clf, param_grid, X_train, y_train,
                                   X_test, y_test):
    """Train one classifier with hyperparameter tuning, optimize threshold on validation set."""
    from sklearn.model_selection import train_test_split

    # Split training into train/val for threshold optimization (no test leakage)
    has_both_train = len(np.unique(y_train)) > 1
    if has_both_train and len(y_train) > 20:
        X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
    else:
        X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = X_train, X_train, y_train, y_train

    # SMOTE on train-fit portion only
    X_tr_sm, y_tr_sm = X_tr_fit, y_tr_fit
    if HAS_SMOTE and len(np.unique(y_tr_fit)) > 1 and np.sum(y_tr_fit == 1) >= 6:
        try:
            k_neighbors = min(5, np.sum(y_tr_fit == 1) - 1)
            if k_neighbors >= 1:
                sm = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
        except Exception:
            pass

    inner_cv = StratifiedKFold(n_splits=N_INNER_CV, shuffle=True, random_state=SEED)
    n_iter = min(N_SEARCH_ITER, max(1, np.prod([len(v) for v in param_grid.values()])))
    try:
        search = RandomizedSearchCV(
            clf, param_grid, n_iter=n_iter, cv=inner_cv, scoring="f1",
            random_state=SEED, n_jobs=-1, error_score=0.0,
        )
        search.fit(X_tr_sm, y_tr_sm)
        best_clf = search.best_estimator_
        best_params = search.best_params_
    except Exception as e:
        log.warning("  %s search failed (%s), using defaults", clf_name, e)
        clf.fit(X_tr_sm, y_tr_sm)
        best_clf = clf
        best_params = {}

    # Optimize threshold on VALIDATION set (not test)
    if hasattr(best_clf, "predict_proba"):
        y_val_prob = best_clf.predict_proba(X_tr_val)[:, 1]
        y_prob = best_clf.predict_proba(X_test)[:, 1]
    elif hasattr(best_clf, "decision_function"):
        y_val_prob = best_clf.decision_function(X_tr_val)
        y_prob = best_clf.decision_function(X_test)
    else:
        y_val_prob = best_clf.predict(X_tr_val).astype(float)
        y_prob = best_clf.predict(X_test).astype(float)

    threshold = youden_threshold(y_tr_val, y_val_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = threshold
    metrics["best_params"] = best_params
    return metrics


def run_loso_evaluation(features: Dict):
    """Run full LOSO evaluation for all classifiers."""
    classifiers = get_classifiers()
    param_grids = get_param_grids()
    subjects = sorted(features.keys())
    all_results = {name: [] for name in classifiers}

    for test_sid in tqdm(subjects, desc="LOSO folds"):
        X_train, y_train, X_test, y_test = prepare_fold(features, test_sid)
        has_both = len(np.unique(y_test)) == 2
        n_fog = int(np.sum(y_test == 1))

        log.info("Fold S%02d: test=%d (%d FoG), train=%d, both_classes=%s",
                 test_sid, len(y_test), n_fog, len(y_train), has_both)

        if len(y_test) == 0:
            continue

        X_train_p, X_test_p, sel_cols, pipes = preprocess_features(X_train, X_test, y_train)

        def _train_clf(clf_name):
            clf = get_classifiers()[clf_name]
            grid = param_grids[clf_name]
            m = train_and_evaluate_classifier(clf_name, clf, grid, X_train_p, y_train,
                                               X_test_p, y_test)
            m["subject"] = test_sid
            m["has_both_classes"] = has_both
            return clf_name, m

        fold_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_train_clf)(name) for name in classifiers
        )

        for clf_name, m in fold_results:
            all_results[clf_name].append(m)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# FUSION TECHNIQUES
# ══════════════════════════════════════════════════════════════════════════════
def get_sensor_group_features(X: pd.DataFrame, group: str) -> pd.DataFrame:
    """Get feature columns for accelerometer or gyroscope group."""
    # Channels: acc_ml=ch0, acc_ap=ch1, acc_si=ch2, gyr_ml=ch3, gyr_ap=ch4, gyr_si=ch5
    # acc_mag=ch6, gyr_mag=ch7, jerk_mag=ch8
    group_map = {
        "acc": [0, 1, 2, 6],     # acc channels + acc_mag
        "gyr": [3, 4, 5, 7],     # gyr channels + gyr_mag
        "all": list(range(9)),    # all channels
    }
    ch_ids = group_map.get(group, list(range(9)))
    cols = [c for c in X.columns if any(f"ch{ch}_" in c for ch in ch_ids)]
    return X[cols] if cols else X


def _build_group_model(X_tr, y_tr, X_te):
    """Train a per-group RF model. Returns (test_probs, train_oof_probs)."""
    from sklearn.model_selection import StratifiedKFold as SKF

    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    # Generate OOF predictions for stacking (no leakage)
    oof_probs = np.zeros(len(y_tr))
    inner_cv = SKF(n_splits=3, shuffle=True, random_state=SEED)

    for tr_idx, val_idx in inner_cv.split(X_tr, y_tr):
        X_f, y_f = X_tr[tr_idx], y_tr[tr_idx]
        X_v = X_tr[val_idx]

        if HAS_SMOTE and np.sum(y_f == 1) >= 6:
            try:
                k_n = min(5, int(np.sum(y_f == 1)) - 1)
                sm = SMOTE(random_state=SEED, k_neighbors=k_n)
                X_f, y_f = sm.fit_resample(X_f, y_f)
            except Exception:
                pass

        clf_inner = RandomForestClassifier(n_estimators=200, max_depth=20,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1)
        clf_inner.fit(X_f, y_f)
        oof_probs[val_idx] = clf_inner.predict_proba(X_v)[:, 1]

    # Train final model on full training set for test predictions
    X_tr_sm, y_tr_sm = X_tr, y_tr
    if HAS_SMOTE and np.sum(y_tr == 1) >= 6:
        try:
            k_n = min(5, int(np.sum(y_tr == 1)) - 1)
            sm = SMOTE(random_state=SEED, k_neighbors=k_n)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass

    clf_full = RandomForestClassifier(n_estimators=200, max_depth=20,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf_full.fit(X_tr_sm, y_tr_sm)
    test_probs = clf_full.predict_proba(X_te)[:, 1]

    return test_probs, oof_probs


def run_fusion_evaluation(features: Dict):
    """Run 3 fusion techniques evaluation. No test-set leakage."""
    from sklearn.model_selection import train_test_split
    subjects = sorted(features.keys())
    fusion_results = {"feature_level": [], "stacking": [], "weighted_voting": []}

    for test_sid in tqdm(subjects, desc="Fusion LOSO"):
        X_train, y_train, X_test, y_test = prepare_fold(features, test_sid)
        has_both = len(np.unique(y_test)) == 2

        if len(y_test) == 0:
            continue

        # ── Fusion 1: Feature-Level ──
        X_tr_p, X_te_p, _, _ = preprocess_features(X_train, X_test, y_train, k=K_FEATURES)

        X_tr_sm, y_tr_sm = X_tr_p, y_train
        if HAS_SMOTE and np.sum(y_train == 1) >= 6:
            try:
                k_n = min(5, np.sum(y_train == 1) - 1)
                sm = SMOTE(random_state=SEED, k_neighbors=k_n)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_p, y_train)
            except Exception:
                pass

        clf_fl = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)
        clf_fl.fit(X_tr_sm, y_tr_sm)
        y_prob_fl = clf_fl.predict_proba(X_te_p)[:, 1]

        # Threshold on validation split (not test)
        if len(y_train) > 20 and len(np.unique(y_train)) > 1:
            _, X_val_fl, _, y_val_fl = train_test_split(X_tr_p, y_train, test_size=0.2,
                                                         stratify=y_train, random_state=SEED)
            y_val_prob_fl = clf_fl.predict_proba(X_val_fl)[:, 1]
            thr_fl = youden_threshold(y_val_fl, y_val_prob_fl)
        else:
            thr_fl = 0.5

        y_pred_fl = (y_prob_fl >= thr_fl).astype(int)
        m_fl = compute_metrics(y_test, y_pred_fl, y_prob_fl)
        m_fl["subject"] = test_sid
        m_fl["has_both_classes"] = has_both
        fusion_results["feature_level"].append(m_fl)

        # ── Per-sensor-group models with OOF predictions ──
        group_oof = {}
        group_test = {}

        for group_name in ["acc", "gyr", "all"]:
            X_tr_g = get_sensor_group_features(X_train, group_name)
            X_te_g = get_sensor_group_features(X_test, group_name)

            if X_tr_g.shape[1] == 0:
                continue

            X_tr_g = X_tr_g.replace([np.inf, -np.inf], np.nan)
            X_te_g = X_te_g.replace([np.inf, -np.inf], np.nan)

            vt = VarianceThreshold(threshold=0.0)
            try:
                X_tr_g_vt = pd.DataFrame(vt.fit_transform(X_tr_g))
                X_te_g_vt = pd.DataFrame(vt.transform(X_te_g))
            except Exception:
                continue
            if X_tr_g_vt.shape[1] == 0:
                continue

            imp = KNNImputer(n_neighbors=5)
            X_tr_g_i = imp.fit_transform(X_tr_g_vt)
            X_te_g_i = imp.transform(X_te_g_vt)

            sc = RobustScaler()
            X_tr_g_sc = sc.fit_transform(X_tr_g_i)
            X_te_g_sc = sc.transform(X_te_g_i)

            k_g = min(30, X_tr_g_sc.shape[1])
            sel = SelectKBest(mutual_info_classif, k=k_g)
            X_tr_g_sel = sel.fit_transform(X_tr_g_sc, y_train)
            X_te_g_sel = sel.transform(X_te_g_sc)

            test_probs, oof_probs = _build_group_model(X_tr_g_sel, y_train, X_te_g_sel)
            group_test[group_name] = test_probs
            group_oof[group_name] = oof_probs

        if len(group_test) < 2:
            continue

        group_names = list(group_test.keys())
        P_test = np.column_stack([group_test[g] for g in group_names])
        P_oof = np.column_stack([group_oof[g] for g in group_names])

        # ── Fusion 2: Stacking with OOF predictions (no leakage) ──
        meta_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        meta_clf.fit(P_oof, y_train)
        y_prob_stack = meta_clf.predict_proba(P_test)[:, 1]

        y_oof_stack = meta_clf.predict_proba(P_oof)[:, 1]
        thr_st = youden_threshold(y_train, y_oof_stack)
        y_pred_stack = (y_prob_stack >= thr_st).astype(int)
        m_st = compute_metrics(y_test, y_pred_stack, y_prob_stack)
        m_st["subject"] = test_sid
        m_st["has_both_classes"] = has_both
        fusion_results["stacking"].append(m_st)

        # ── Fusion 3: Weighted Voting — weights optimized on OOF (not test) ──
        def neg_f1_weighted(w):
            w = np.abs(w)
            w = w / (w.sum() + 1e-12)
            fused = P_oof @ w
            preds = (fused >= 0.5).astype(int)
            return -f1_score(y_train, preds, zero_division=0)

        w0 = np.ones(len(group_names)) / len(group_names)
        try:
            res = minimize(neg_f1_weighted, w0, method="Nelder-Mead", options={"maxiter": 200})
            w_opt = np.abs(res.x)
            w_opt = w_opt / (w_opt.sum() + 1e-12)
        except Exception:
            w_opt = w0

        y_prob_wv = P_test @ w_opt
        y_oof_wv = P_oof @ w_opt
        thr_wv = youden_threshold(y_train, y_oof_wv)
        y_pred_wv = (y_prob_wv >= thr_wv).astype(int)
        m_wv = compute_metrics(y_test, y_pred_wv, y_prob_wv)
        m_wv["subject"] = test_sid
        m_wv["has_both_classes"] = has_both
        m_wv["weights"] = {g: round(w, 3) for g, w in zip(group_names, w_opt)}
        fusion_results["weighted_voting"].append(m_wv)

    return fusion_results


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_results(results_list: List[Dict], name: str = "") -> Dict:
    evaluable = [r for r in results_list if r.get("has_both_classes", True)]
    all_folds = results_list

    cm_total = np.zeros((2, 2), dtype=int)
    for r in all_folds:
        cm_total += np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    tn, fp, fn, tp = cm_total.ravel()
    total = tn + fp + fn + tp

    agg = {
        "name": name,
        "n_folds_total": len(all_folds),
        "n_folds_evaluable": len(evaluable),
        "accuracy_agg": (tp + tn) / total if total > 0 else 0,
        "precision_agg": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall_agg": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity_agg": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1_agg": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "cm_total": cm_total.tolist(),
    }
    if evaluable:
        for metric in ["f1", "recall", "precision", "specificity", "balanced_acc"]:
            vals = [r[metric] for r in evaluable if metric in r]
            agg[f"{metric}_mean"] = np.mean(vals) if vals else 0
        auc_vals = [r["auc"] for r in evaluable if "auc" in r]
        agg["auc_mean"] = np.mean(auc_vals) if auc_vals else 0
    return agg


def print_results_table(all_results: Dict, title: str = "CLASSIFIER COMPARISON"):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Classifier':<18} {'F1(agg)':>8} {'F1(mean)':>9} {'Recall':>8} {'Prec':>8} "
          f"{'Spec':>8} {'BalAcc':>8} {'AUC':>8} {'Folds':>6}")
    print("-" * 90)

    rows = []
    for name, results in all_results.items():
        agg = aggregate_results(results, name)
        rows.append(agg)
    rows.sort(key=lambda x: x.get("f1_agg", 0), reverse=True)

    for agg in rows:
        print(f"{agg['name']:<18} {agg['f1_agg']:>8.4f} {agg.get('f1_mean', 0):>9.4f} "
              f"{agg.get('recall_mean', 0):>8.4f} {agg.get('precision_mean', 0):>8.4f} "
              f"{agg.get('specificity_mean', 0):>8.4f} {agg.get('balanced_acc_mean', 0):>8.4f} "
              f"{agg.get('auc_mean', 0):>8.4f} {agg['n_folds_evaluable']:>3}/{agg['n_folds_total']}")

    print("-" * 90)
    print(f"\nTOP 3 CLASSIFIERS (by aggregated F1):")
    for i, agg in enumerate(rows[:3]):
        print(f"  {i+1}. {agg['name']} — F1={agg['f1_agg']:.4f}, AUC={agg.get('auc_mean', 0):.4f}")
    return rows


def print_fusion_results(fusion_results: Dict):
    print(f"\n{'='*90}")
    print(f"  FUSION TECHNIQUE COMPARISON")
    print(f"{'='*90}")

    fusion_names = {"feature_level": "Feature-Level Fusion",
                    "stacking": "Stacking (Meta-LR)",
                    "weighted_voting": "Weighted Soft Voting"}
    rows = []
    for key, results in fusion_results.items():
        if results:
            agg = aggregate_results(results, fusion_names.get(key, key))
            rows.append(agg)
    rows.sort(key=lambda x: x.get("f1_agg", 0), reverse=True)

    print(f"{'Fusion Method':<25} {'F1(agg)':>8} {'F1(mean)':>9} {'Recall':>8} {'Prec':>8} "
          f"{'Spec':>8} {'AUC':>8} {'Folds':>6}")
    print("-" * 90)

    for agg in rows:
        print(f"{agg['name']:<25} {agg['f1_agg']:>8.4f} {agg.get('f1_mean', 0):>9.4f} "
              f"{agg.get('recall_mean', 0):>8.4f} {agg.get('precision_mean', 0):>8.4f} "
              f"{agg.get('specificity_mean', 0):>8.4f} {agg.get('auc_mean', 0):>8.4f} "
              f"{agg['n_folds_evaluable']:>3}/{agg['n_folds_total']}")

    print("-" * 90)
    print(f"\nTOP 3 FUSION TECHNIQUES (by aggregated F1):")
    for i, agg in enumerate(rows[:3]):
        print(f"  {i+1}. {agg['name']} — F1={agg['f1_agg']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  FIGSHARE FoG DETECTION — IMPROVED PIPELINE")
    print("=" * 90)
    print(f"  Sampling rate: {FS} Hz")
    print(f"  Window: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"  Bandpass: {BP_LOW}-{BP_HIGH} Hz")
    print(f"  Label threshold: {LABEL_THRESH} (majority voting)")
    print(f"  Feature selection: top {K_FEATURES} (mutual information)")
    print(f"  Derived signals: acc_mag, gyr_mag, jerk_mag")
    print(f"  Threshold optimisation: Youden's J")
    print()

    # Step 1: Load & preprocess
    subjects_data = load_and_preprocess()

    # Step 2: Extract features
    features = extract_all_features(subjects_data)

    # Step 3: LOSO evaluation
    log.info("Running LOSO evaluation with %d classifiers...", len(get_classifiers()))
    all_results = run_loso_evaluation(features)

    # Step 4: Print results
    clf_rows = print_results_table(all_results)

    # Step 5: Fusion evaluation
    log.info("Running fusion evaluation...")
    fusion_results = run_fusion_evaluation(features)
    print_fusion_results(fusion_results)

    # Step 6: Save results
    save_data = {}
    for name, results in all_results.items():
        save_data[name] = aggregate_results(results, name)
    save_data["fusion"] = {}
    for key, results in fusion_results.items():
        if results:
            save_data["fusion"][key] = aggregate_results(results, key)

    results_file = OUTPUT_DIR / "results_summary.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2, default=_convert)
    log.info("Results saved to %s", results_file)

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
