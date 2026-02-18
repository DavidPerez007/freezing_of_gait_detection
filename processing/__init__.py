"""
Processing module for Daphnet FoG Dataset.

This module provides signal cleaning, labeling, and windowing functionality
for preprocessing time-series sensor data.
"""

from .signal_cleaning import SignalCleaner
from .labeling import LabelCreator
from .windowing import WindowCreator

__all__ = [
    'SignalCleaner',
    'LabelCreator',
    'WindowCreator',
]
