"""
FoG Dataset Loaders Package

Este paquete contiene loaders estandarizados para múltiples datasets
de Freezing of Gait (FoG) en pacientes con Parkinson.
"""

from .BaseDatasetLoader import BaseDatasetLoader, BaseFileReader, load_dataset
from .DaphnetReader import DaphnetDatasetLoader, DaphnetFileReader
from .FigshareReader import FigshareDatasetLoader, FigshareFileReader
from .ChariteReader import ChariteDatasetLoader, ChariteFileReader
from .MendelayReader import MendelayDatasetLoader, MendelayFileReader
from .KaggleReader import KaggleDatasetLoader, KaggleFileReader

__all__ = [
    'BaseDatasetLoader',
    'BaseFileReader',
    'load_dataset',
    'DaphnetDatasetLoader',
    'DaphnetFileReader',
    'FigshareDatasetLoader',
    'FigshareFileReader',
    'ChariteDatasetLoader',
    'ChariteFileReader',
    'MendelayDatasetLoader',
    'MendelayFileReader',
    'KaggleDatasetLoader',
    'KaggleFileReader',
]

__version__ = '1.0.0'
