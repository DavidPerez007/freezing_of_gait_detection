"""
Shared utility functions for Daphnet and Figshare improved pipelines.
=====================================================================
Contains signal processing, ML helpers, and results reporting functions
that are common across both FoG detection pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize

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


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
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


def bandpass_filter(data: np.ndarray, fs: int, low: float = 0.5,
                    high: float = 20.0, order: int = 4) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════════════════════════
# ML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold via Youden's J statistic."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute classification metrics."""
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


def get_classifiers(seed: int = 42) -> Dict[str, Any]:
    """Get dictionary of classifiers."""
    clfs = {
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
        "LogisticReg": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=seed),
        "MLP": MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
    }
    if HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed,
                                         n_jobs=1, verbosity=0)
    return clfs


def get_param_grids() -> Dict[str, Dict]:
    """Get hyperparameter grids for all classifiers."""
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


def prepare_fold(features: Dict, test_sid: int):
    """Prepare train/test split for one LOSO fold."""
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
                        y_train: np.ndarray, k: int = 60):
    """Clean, scale, select features. Returns processed arrays + selector."""
    # Replace inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Variance threshold (remove constants)
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
                                   X_test, y_test, seed: int = 42,
                                   n_inner_cv: int = 3, n_search_iter: int = 20,
                                   fold_info=""):
    """Train one classifier with hyperparameter tuning, optimize threshold on validation set."""
    from sklearn.model_selection import train_test_split

    # Split training into train/val for threshold optimization (stratified)
    has_both_train = len(np.unique(y_train)) > 1
    if has_both_train and len(y_train) > 20:
        X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    else:
        X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = X_train, X_train, y_train, y_train

    # SMOTE on train-fit portion only
    X_tr_sm, y_tr_sm = X_tr_fit, y_tr_fit
    if HAS_SMOTE and len(np.unique(y_tr_fit)) > 1 and np.sum(y_tr_fit == 1) >= 6:
        try:
            k_neighbors = min(5, np.sum(y_tr_fit == 1) - 1)
            if k_neighbors >= 1:
                sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
        except Exception:
            pass

    # Hyperparameter search
    inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=seed)
    n_iter = min(n_search_iter, max(1, np.prod([len(v) for v in param_grid.values()])))
    try:
        search = RandomizedSearchCV(
            clf, param_grid, n_iter=n_iter, cv=inner_cv, scoring="f1",
            random_state=seed, n_jobs=-1, error_score=0.0,
        )
        search.fit(X_tr_sm, y_tr_sm)
        best_clf = search.best_estimator_
        best_params = search.best_params_
    except Exception:
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


def build_base_model(X_tr, y_tr, X_te, seed: int = 42):
    """Train a per-group/per-sensor RF model. Returns (test_probs, train_oof_probs).
    Used by both Daphnet (per-sensor) and Figshare (per-group) fusion pipelines."""
    from sklearn.model_selection import StratifiedKFold as SKF

    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    # Generate OOF predictions for stacking (no leakage)
    oof_probs = np.zeros(len(y_tr))
    inner_cv = SKF(n_splits=3, shuffle=True, random_state=seed)

    for tr_idx, val_idx in inner_cv.split(X_tr, y_tr):
        X_f, y_f = X_tr[tr_idx], y_tr[tr_idx]
        X_v = X_tr[val_idx]

        # SMOTE on inner train
        if HAS_SMOTE and np.sum(y_f == 1) >= 6:
            try:
                k_n = min(5, int(np.sum(y_f == 1)) - 1)
                sm = SMOTE(random_state=seed, k_neighbors=k_n)
                X_f, y_f = sm.fit_resample(X_f, y_f)
            except Exception:
                pass

        clf_inner = RandomForestClassifier(n_estimators=200, max_depth=20,
                                           class_weight="balanced", random_state=seed, n_jobs=-1)
        clf_inner.fit(X_f, y_f)
        oof_probs[val_idx] = clf_inner.predict_proba(X_v)[:, 1]

    # Train final model on full training set for test predictions
    X_tr_sm, y_tr_sm = X_tr, y_tr
    if HAS_SMOTE and np.sum(y_tr == 1) >= 6:
        try:
            k_n = min(5, int(np.sum(y_tr == 1)) - 1)
            sm = SMOTE(random_state=seed, k_neighbors=k_n)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass

    clf_full = RandomForestClassifier(n_estimators=200, max_depth=20,
                                       class_weight="balanced", random_state=seed, n_jobs=-1)
    clf_full.fit(X_tr_sm, y_tr_sm)
    test_probs = clf_full.predict_proba(X_te)[:, 1]

    return test_probs, oof_probs


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_results(results_list: List[Dict], name: str = "") -> Dict:
    """Aggregate LOSO fold results, excluding monoclase folds for F1/recall."""
    evaluable = [r for r in results_list if r.get("has_both_classes", True)]
    all_folds = results_list

    # Aggregate confusion matrix across ALL folds
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

    # Mean of evaluable folds
    if evaluable:
        for metric in ["f1", "recall", "precision", "specificity", "balanced_acc"]:
            vals = [r[metric] for r in evaluable if metric in r]
            agg[f"{metric}_mean"] = np.mean(vals) if vals else 0
        auc_vals = [r["auc"] for r in evaluable if "auc" in r]
        agg["auc_mean"] = np.mean(auc_vals) if auc_vals else 0

    return agg


def print_results_table(all_results: Dict, title: str = "CLASSIFIER COMPARISON"):
    """Print formatted results table."""
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
    """Print fusion comparison table."""
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
