"""
Visualization utilities for Daphnet FoG Dataset.

This module provides plotting functions for data exploration and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from utils.constants import LABEL_NAMES_BINARY, LABEL_NAMES_MULTICLASS, SAMPLING_RATE


def plot_window_example(
    window: np.ndarray,
    label: int,
    sampling_rate: int = SAMPLING_RATE,
    label_names: dict = None,
    title: Optional[str] = None
) -> None:
    """
    Plot an example window showing all channels.

    Parameters
    ----------
    window : np.ndarray
        Window data of shape (n_samples, n_channels)
    label : int
        Window label
    sampling_rate : int, optional
        Sampling rate in Hz (default: 64)
    label_names : dict, optional
        Mapping of labels to names (default: None)
    title : str, optional
        Plot title (default: auto-generated)

    Examples
    --------
    >>> plot_window_example(window, label=1)
    """
    n_samples, n_channels = window.shape
    time_axis = np.arange(n_samples) / sampling_rate

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        axes[ch].plot(time_axis, window[:, ch], linewidth=1.5)
        axes[ch].set_ylabel(f'Channel {ch}')
        axes[ch].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)')

    if title is None:
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        title = f'Window Example - Label: {label_name}'

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_fog_distribution_per_fold(
    loso_splits: List[dict],
    label_type: str = 'binary'
) -> None:
    """
    Plot FoG distribution across LOSO folds.

    Parameters
    ----------
    loso_splits : List[dict]
        List of LOSO splits
    label_type : str, optional
        Type of labels: 'binary' or 'multiclass' (default: 'binary')

    Examples
    --------
    >>> plot_fog_distribution_per_fold(loso_splits)
    """
    n_folds = len(loso_splits)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train/test sizes
    train_sizes = [split['train_dist'].sum() if hasattr(split.get('train_dist', []), 'sum')
                   else len(split.get('y_train', [])) for split in loso_splits]
    test_sizes = [split['test_dist'].sum() if hasattr(split.get('test_dist', []), 'sum')
                  else len(split.get('y_test', [])) for split in loso_splits]
    test_subjects = [split['test_subject'] for split in loso_splits]

    x = np.arange(n_folds)
    width = 0.35

    axes[0].bar(x - width/2, train_sizes, width, label='Train', color='steelblue')
    axes[0].bar(x + width/2, test_sizes, width, label='Test', color='coral')
    axes[0].set_xlabel('Fold (Test Subject)')
    axes[0].set_ylabel('Number of Windows')
    axes[0].set_title('Train/Test Size per Fold', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"F{i}\n{s}" for i, s in enumerate(test_subjects)])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Class distribution in test set
    for i, split in enumerate(loso_splits):
        test_dist = split.get('test_dist', np.array([]))
        if len(test_dist) == 0:
            continue

        bottom = 0
        colors = ['steelblue', 'coral', 'gold']
        labels = list(LABEL_NAMES_MULTICLASS.values()) if label_type == 'multiclass' else list(LABEL_NAMES_BINARY.values())

        for cls_idx, count in enumerate(test_dist):
            axes[1].bar(i, count, bottom=bottom, color=colors[cls_idx % len(colors)],
                       label=labels[cls_idx] if i == 0 and cls_idx < len(labels) else "")
            bottom += count

    axes[1].set_xlabel('Fold (Test Subject)')
    axes[1].set_ylabel('Number of Windows')
    axes[1].set_title(f'{label_type.capitalize()} Distribution in Test Set', fontweight='bold')
    axes[1].set_xticks(range(n_folds))
    axes[1].set_xticklabels([f"F{i}\n{s}" for i, s in enumerate(test_subjects)])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_correlation(
    features_df: pd.DataFrame,
    method: str = 'pearson',
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot correlation heatmap of features.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'pearson')
    figsize : tuple, optional
        Figure size (default: (12, 10))

    Examples
    --------
    >>> plot_feature_correlation(features_df)
    """
    plt.figure(figsize=figsize)

    corr = features_df.corr(method=method)

    sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.title(f'Feature Correlation Matrix ({method.capitalize()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_label_distribution(
    labels: pd.Series | np.ndarray,
    label_type: str = 'binary',
    title: Optional[str] = None
) -> None:
    """
    Plot distribution of labels.

    Parameters
    ----------
    labels : pd.Series | np.ndarray
        Label series or array
    label_type : str, optional
        Type of labels: 'binary' or 'multiclass' (default: 'binary')
    title : str, optional
        Plot title (default: auto-generated)

    Examples
    --------
    >>> plot_label_distribution(multiclass_labels, 'multiclass')
    """
    if isinstance(labels, pd.Series):
        labels = labels.values

    unique, counts = np.unique(labels, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    label_names = LABEL_NAMES_MULTICLASS if label_type == 'multiclass' else LABEL_NAMES_BINARY
    names = [label_names.get(u, str(u)) for u in unique]
    colors = ['steelblue', 'coral', 'gold']

    ax1.bar(names, counts, color=colors[:len(names)])
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution - Counts', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (name, count) in enumerate(zip(names, counts)):
        percentage = count / len(labels) * 100
        ax1.text(i, count, f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=names, autopct='%1.1f%%', colors=colors[:len(names)])
    ax2.set_title('Label Distribution - Proportions', fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_subject_distribution(
    df: pd.DataFrame,
    subject_col: str = 'subject',
    label_col: str = 'annotation'
) -> None:
    """
    Plot distribution of data per subject.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    subject_col : str, optional
        Name of subject column (default: 'subject')
    label_col : str, optional
        Name of label column (default: 'annotation')

    Examples
    --------
    >>> plot_subject_distribution(df)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Samples per subject
    subject_counts = df[subject_col].value_counts().sort_index()
    axes[0].bar(range(len(subject_counts)), subject_counts.values, color='steelblue')
    axes[0].set_xlabel('Subject')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Number of Samples per Subject', fontweight='bold')
    axes[0].set_xticks(range(len(subject_counts)))
    axes[0].set_xticklabels(subject_counts.index)
    axes[0].grid(axis='y', alpha=0.3)

    # FoG samples per subject
    fog_by_subject = df[df[label_col] == 2].groupby(subject_col).size()
    axes[1].bar(range(len(fog_by_subject)), fog_by_subject.values, color='coral')
    axes[1].set_xlabel('Subject')
    axes[1].set_ylabel('Number of FoG Samples')
    axes[1].set_title('FoG Episodes per Subject', fontweight='bold')
    axes[1].set_xticks(range(len(fog_by_subject)))
    axes[1].set_xticklabels(fog_by_subject.index)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
