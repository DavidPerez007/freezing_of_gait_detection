"""
Charité-Universitätsmedizin Berlin Dataset Loader

Dataset Information:
- Source: Charité - Real-Time Detection of Freezing Motions in Parkinson's Patients
- Published: 2021, Frontiers in Neurology
- Sensors: 3D accelerometer + 3D gyroscope @ 200 Hz
- Placement: Both feet (left and right), worn simultaneously
- Subjects: 16 subjects (S1-S16)
- Trials: 2 trials per subject

Column naming convention: {sensor}_{axis}_{body_position}
  e.g. acc_x_left_foot, gyr_z_right_foot
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader


# Raw CSV columns (in order as they appear in the file)
_RAW_COLUMNS = ['time_s', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'fog_label']

# Sensor channel names (without body position suffix)
_SENSOR_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']


class ChariteFileReader(BaseFileReader):
    """Lee archivos individuales del dataset Charité."""

    def read_file(self, file_path: Path, foot: str = None) -> pd.DataFrame:
        """
        Lee un archivo CSV del dataset Charité y renombra columnas según la posición del pie.

        Args:
            file_path: Ruta al archivo .csv
            foot: 'left' o 'right'. Si es None, se infiere del nombre del archivo.

        Returns:
            DataFrame con columnas renombradas: acc_x_{foot}_foot, gyr_z_{foot}_foot, etc.
        """
        try:
            df = pd.read_csv(file_path)
            df.columns = _RAW_COLUMNS

            if foot is None:
                foot = self._parse_filename(file_path.name)['foot']

            rename_map = {ch: f'{ch}_{foot}_foot' for ch in _SENSOR_CHANNELS}
            df = df.rename(columns=rename_map)

            return df

        except Exception as e:
            raise Exception(f"Error leyendo archivo {file_path}: {str(e)}")

    @staticmethod
    def _parse_filename(filename: str) -> Dict[str, any]:
        """
        Extrae información del nombre del archivo.
        Formato: S<subject>_<foot>_foot_trial_<trial>.csv
        Ejemplo: S1_left_foot_trial_1.csv
        """
        name = filename.replace('.csv', '')
        parts = name.split('_')
        return {
            'subject_id': int(parts[0].replace('S', '')),
            'foot': parts[1],
            'trial_id': int(parts[4])
        }


class ChariteDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Charité-Universitätsmedizin Berlin.

    Fusiona los archivos de pie izquierdo y derecho de cada trial en un único
    DataFrame con 12 columnas de sensores (ambos pies llevados simultáneamente):

        acc_x_left_foot,  acc_y_left_foot,  acc_z_left_foot,
        gyr_x_left_foot,  gyr_y_left_foot,  gyr_z_left_foot,
        acc_x_right_foot, acc_y_right_foot, acc_z_right_foot,
        gyr_x_right_foot, gyr_y_right_foot, gyr_z_right_foot
    """

    FEATURE_COLUMNS = [
        'acc_x_left_foot', 'acc_y_left_foot', 'acc_z_left_foot',
        'gyr_x_left_foot', 'gyr_y_left_foot', 'gyr_z_left_foot',
        'acc_x_right_foot', 'acc_y_right_foot', 'acc_z_right_foot',
        'gyr_x_right_foot', 'gyr_y_right_foot', 'gyr_z_right_foot',
    ]

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Ruta a la carpeta 'data' del dataset Charité
        """
        super().__init__(dataset_path)
        self.file_reader = ChariteFileReader()

    def _read_trial(self, subject_id: int, trial_id: int) -> pd.DataFrame:
        """
        Lee y fusiona los archivos de ambos pies para un trial específico.

        Args:
            subject_id: ID del sujeto
            trial_id: ID del trial (1 o 2)

        Returns:
            DataFrame con 12 columnas de sensores fusionadas por timestamp
        """
        subject_folder = self.dataset_path / f"S{subject_id}"

        left_file = subject_folder / f"S{subject_id}_left_foot_trial_{trial_id}.csv"
        right_file = subject_folder / f"S{subject_id}_right_foot_trial_{trial_id}.csv"

        df_left = self.file_reader.read_file(left_file, foot='left')
        df_right = self.file_reader.read_file(right_file, foot='right')

        # Los timestamps son idénticos → combinar columnas directamente
        right_sensor_cols = [f'{ch}_right_foot' for ch in _SENSOR_CHANNELS]
        df = df_left.copy()
        for col in right_sensor_cols:
            df[col] = df_right[col].values

        df['subject_id'] = subject_id
        df['trial_id'] = trial_id
        df['fog_label_text'] = df['fog_label'].map({0: 'No FoG', 1: 'FoG Active'})

        return df

    def get_file_list(self, **kwargs) -> List[Path]:
        """Obtiene lista de todos los archivos CSV del dataset."""
        files = sorted(self.dataset_path.glob('S*/*_foot_trial_*.csv'))
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {self.dataset_path}")
        return files

    def _get_trials(self) -> List[tuple]:
        """Retorna lista de (subject_id, trial_id) únicos disponibles."""
        files = self.get_file_list()
        trials = set()
        for f in files:
            info = self.file_reader._parse_filename(f.name)
            trials.add((info['subject_id'], info['trial_id']))
        return sorted(trials)

    def load_all_data(self, verbose: bool = True, **kwargs) -> pd.DataFrame:
        """
        Carga todos los trials del dataset fusionando ambos pies por timestamp.

        Returns:
            DataFrame con 12 columnas de sensores (6 por pie)
        """
        trials = self._get_trials()

        if verbose:
            print(f"📁 Encontrados {len(trials)} trials")
            print(f"📊 Cargando datos del dataset Charité-Universitätsmedizin...\n")

        dataframes = []
        iterator = tqdm(trials, desc="Cargando trials") if verbose else trials

        for subject_id, trial_id in iterator:
            try:
                df = self._read_trial(subject_id, trial_id)
                dataframes.append(df)
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Error en S{subject_id} trial {trial_id}: {str(e)}")
                continue

        if not dataframes:
            raise ValueError("No se pudieron cargar datos de ningún trial")

        self.data = pd.concat(dataframes, ignore_index=True)

        if verbose:
            print(f"\n✅ Dataset cargado exitosamente")
            self.print_summary()

        return self.data

    def load_subject_data(self, subject_id: int, **kwargs) -> pd.DataFrame:
        """
        Carga todos los trials de un sujeto específico.

        Args:
            subject_id: ID del sujeto (1-16)

        Returns:
            DataFrame con los datos del sujeto
        """
        subject_folder = self.dataset_path / f"S{subject_id}"
        if not subject_folder.exists():
            raise FileNotFoundError(f"No se encontró la carpeta para el sujeto {subject_id}")

        left_files = sorted(subject_folder.glob(f"S{subject_id}_left_foot_trial_*.csv"))
        trial_ids = [self.file_reader._parse_filename(f.name)['trial_id'] for f in left_files]

        dataframes = [self._read_trial(subject_id, tid) for tid in trial_ids]
        return pd.concat(dataframes, ignore_index=True)

    def get_summary_by_subject(self) -> pd.DataFrame:
        """Genera un resumen estadístico por sujeto."""
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")

        summary = self.data.groupby('subject_id').agg(
            duration_s=('time_s', 'max'),
            fog_count=('fog_label', 'sum'),
            total_samples=('fog_label', 'count'),
            n_trials=('trial_id', 'nunique'),
        ).round(2)

        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)

        return summary

    def get_summary_by_trial(self) -> pd.DataFrame:
        """Genera un resumen por trial (1 vs 2)."""
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")

        summary = self.data.groupby('trial_id').agg(
            n_subjects=('subject_id', 'nunique'),
            duration_s=('time_s', 'max'),
            fog_count=('fog_label', 'sum'),
            total_samples=('fog_label', 'count'),
        ).round(2)

        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)

        return summary


if __name__ == "__main__":
    DATASET_PATH = r'Datasets/Charité–Universitätsmedizin Berlin/Data Sheet 2/data'

    loader = ChariteDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True)

    print("\n" + "="*60)
    print("RESUMEN POR SUJETO")
    print("="*60)
    print(loader.get_summary_by_subject().head())

    print("\n" + "="*60)
    print("RESUMEN POR TRIAL")
    print("="*60)
    print(loader.get_summary_by_trial())

    print("\n" + "="*60)
    print("COLUMNAS DE SENSORES")
    print("="*60)
    print(ChariteDatasetLoader.FEATURE_COLUMNS)
