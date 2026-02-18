"""
Charité-Universitätsmedizin Berlin Dataset Loader

Dataset Information:
- Source: Charité - Real-Time Detection of Freezing Motions in Parkinson's Patients
- Published: 2021, Frontiers in Neurology
- Sensors: 3D accelerometer + 3D gyroscope @ 200 Hz
- Placement: Both feet (left and right)
- Subjects: 16 subjects (S1-S16)
- Trials: 2 trials per subject
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader


class ChariteFileReader(BaseFileReader):
    """Lee archivos individuales del dataset Charité."""
    
    # Columnas esperadas en los archivos CSV
    COLUMN_NAMES = [
        'time_s',
        'acc_x_m_s2',    # Aceleración eje X [m/s²]
        'acc_y_m_s2',    # Aceleración eje Y [m/s²]
        'acc_z_m_s2',    # Aceleración eje Z [m/s²]
        'gyr_x_deg_s',   # Velocidad angular eje X [deg/s]
        'gyr_y_deg_s',   # Velocidad angular eje Y [deg/s]
        'gyr_z_deg_s',   # Velocidad angular eje Z [deg/s]
        'fog_label'      # FoG label (1 = FoG activo, 0 = no FoG)
    ]
    
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo CSV del dataset Charité.
        
        Args:
            file_path: Ruta al archivo .csv
            
        Returns:
            DataFrame con los datos del archivo
        """
        try:
            # Leer archivo CSV con encabezados
            df = pd.read_csv(file_path)
            
            # Renombrar columnas a nuestro formato estándar
            df.columns = self.COLUMN_NAMES
            
            # Extraer información del nombre del archivo
            file_info = self._parse_filename(file_path.name)
            
            # Agregar metadatos
            df['subject_id'] = file_info['subject_id']
            df['trial_id'] = file_info['trial_id']
            df['foot'] = file_info['foot']
            df['filename'] = file_path.name
            
            # Crear etiqueta descriptiva del FoG
            df['fog_label_text'] = df['fog_label'].map({
                0: 'No FoG',
                1: 'FoG Active'
            })
            
            return df
            
        except Exception as e:
            raise Exception(f"Error leyendo archivo {file_path}: {str(e)}")
    
    @staticmethod
    def _parse_filename(filename: str) -> Dict[str, any]:
        """
        Extrae información del nombre del archivo.
        Formato: S<subject>_<foot>_foot_trial_<trial>.csv
        Ejemplo: S1_left_foot_trial_1.csv
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Diccionario con subject_id, trial_id y foot
        """
        # Remover extensión
        name = filename.replace('.csv', '')
        
        # Dividir por underscore
        parts = name.split('_')
        
        # Extraer información
        # S1 -> 1
        subject_id = int(parts[0].replace('S', ''))
        
        # left o right
        foot = parts[1]  # 'left' o 'right'
        
        # trial_1 -> 1
        trial_id = int(parts[4])
        
        return {
            'subject_id': subject_id,
            'trial_id': trial_id,
            'foot': foot
        }


class ChariteDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Charité-Universitätsmedizin Berlin."""
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta a la carpeta 'data' del dataset Charité
        """
        super().__init__(dataset_path)
        self.file_reader = ChariteFileReader()
    
    def get_file_list(self, foot: Optional[str] = None, **kwargs) -> List[Path]:
        """
        Obtiene lista de todos los archivos CSV en el dataset.
        
        Args:
            foot: Filtrar por pie ('left', 'right', None=todos)
        
        Returns:
            Lista de rutas a archivos CSV
        """
        # Buscar en todas las carpetas de sujetos
        files = sorted(self.dataset_path.glob('S*/*_foot_trial_*.csv'))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {self.dataset_path}"
            )
        
        # Filtrar por pie si se especifica
        if foot:
            foot_pattern = f'_{foot}_foot_'
            files = [f for f in files if foot_pattern in f.name]
        
        return files
    
    def load_all_data(self, verbose: bool = True, foot: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga todos los archivos CSV del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar barra de progreso
            foot: Filtrar por pie ('left', 'right', None=todos)
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        files = self.get_file_list(foot=foot)
        
        if verbose:
            print(f"📁 Encontrados {len(files)} archivos CSV")
            if foot:
                print(f"🦶 Filtrando por pie: {foot}")
            print(f"📊 Cargando datos del dataset Charité-Universitätsmedizin...\n")
        
        # Leer todos los archivos
        dataframes = []
        iterator = tqdm(files, desc="Cargando archivos") if verbose else files
        
        for file_path in iterator:
            try:
                df = self.file_reader.read_file(file_path)
                dataframes.append(df)
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Error en {file_path.name}: {str(e)}")
                continue
        
        # Concatenar todos los DataFrames
        if not dataframes:
            raise ValueError("No se pudieron cargar datos de ningún archivo")
        
        self.data = pd.concat(dataframes, ignore_index=True)
        
        if verbose:
            print(f"\n✅ Dataset cargado exitosamente")
            self.print_summary()
        
        return self.data
    
    def load_subject_data(self, subject_id: int, foot: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga datos de un sujeto específico.
        
        Args:
            subject_id: ID del sujeto (1-16)
            foot: Filtrar por pie ('left', 'right', None=todos)
            **kwargs: Parámetros adicionales (no utilizados)
            
        Returns:
            DataFrame con los datos del sujeto
        """
        # Buscar carpeta del sujeto
        subject_folder = self.dataset_path / f"S{subject_id}"
        
        if not subject_folder.exists():
            raise FileNotFoundError(f"No se encontró la carpeta para el sujeto {subject_id}")
        
        # Obtener archivos del sujeto
        pattern = f"S{subject_id}_*_foot_trial_*.csv"
        files = sorted(subject_folder.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para el sujeto {subject_id}")
        
        # Filtrar por pie si se especifica
        if foot:
            foot_pattern = f'_{foot}_foot_'
            files = [f for f in files if foot_pattern in f.name]
        
        # Leer archivos
        dataframes = []
        for file_path in files:
            df = self.file_reader.read_file(file_path)
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)
    
    def get_summary_by_subject(self) -> pd.DataFrame:
        """
        Genera un resumen estadístico por sujeto.
        
        Returns:
            DataFrame con estadísticas por sujeto
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        # Agrupar por sujeto
        summary = self.data.groupby('subject_id').agg({
            'time_s': 'max',  # Duración total
            'fog_label': ['sum', 'count'],  # Total de FoG y muestras
            'trial_id': 'nunique',  # Número de trials
            'filename': 'nunique'  # Número de archivos
        }).round(2)
        
        # Renombrar columnas
        summary.columns = ['duration_s', 'fog_count', 'total_samples', 
                          'n_trials', 'n_files']
        
        # Calcular porcentaje de FoG
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_foot(self) -> pd.DataFrame:
        """
        Genera un resumen comparativo entre pie izquierdo y derecho.
        
        Returns:
            DataFrame con estadísticas por pie
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('foot').agg({
            'subject_id': 'nunique',
            'filename': 'nunique',
            'time_s': 'max',
            'fog_label': ['sum', 'count']
        }).round(2)
        
        summary.columns = ['n_subjects', 'n_files', 'total_duration_s', 
                          'fog_count', 'total_samples']
        
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_trial(self) -> pd.DataFrame:
        """
        Genera un resumen por trial (1 vs 2).
        
        Returns:
            DataFrame con estadísticas por trial
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('trial_id').agg({
            'subject_id': 'nunique',
            'filename': 'nunique',
            'time_s': 'max',
            'fog_label': ['sum', 'count']
        }).round(2)
        
        summary.columns = ['n_subjects', 'n_files', 'total_duration_s', 
                          'fog_count', 'total_samples']
        
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary


if __name__ == "__main__":
    # Ejemplo de uso
    DATASET_PATH = r'Datasets\Charité–Universitätsmedizin Berlin\Data Sheet 2\data'
    
    loader = ChariteDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True)
    
    print("\n" + "="*60)
    print("RESUMEN POR SUJETO")
    print("="*60)
    print(loader.get_summary_by_subject().head())
    
    print("\n" + "="*60)
    print("RESUMEN POR PIE")
    print("="*60)
    print(loader.get_summary_by_foot())
    
    print("\n" + "="*60)
    print("RESUMEN POR TRIAL")
    print("="*60)
    print(loader.get_summary_by_trial())
    
    # Guardar dataset (descomenta para ejecutar)
    # loader.save_dataset('charite_complete_dataset', formats=['csv', 'parquet'])
