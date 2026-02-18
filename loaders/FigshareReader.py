"""
Figshare Public Dataset Loader for Parkinson's Disease and Freezing of Gait

Dataset Information:
- Source: Figshare - A public dataset of Parkinson's disease with FoG
- Sensors: IMU data (accelerometer + gyroscope)
- Subjects: 35 subjects
- Trial types: Walking and Standing
- Metadata: PDFEinfo_cleaned.csv con información demográfica y clínica
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader


class FigshareFileReader(BaseFileReader):
    """Lee archivos individuales del dataset Figshare."""
    
    # Columnas esperadas en los archivos IMU
    COLUMN_NAMES = [
        'frame',
        'time_s',
        'acc_ml_g',      # ACC ML [g] - Medio-Lateral
        'acc_ap_g',      # ACC AP [g] - Antero-Posterior
        'acc_si_g',      # ACC SI [g] - Superior-Inferior
        'gyr_ml_deg_s',  # GYR ML [deg/s]
        'gyr_ap_deg_s',  # GYR AP [deg/s]
        'gyr_si_deg_s',  # GYR SI [deg/s]
        'freezing_flag'  # Freezing event [flag]
    ]
    
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo TXT del dataset Figshare.
        
        Args:
            file_path: Ruta al archivo .txt
            
        Returns:
            DataFrame con los datos del archivo
        """
        try:
            # Leer archivo con tabuladores como separador
            df = pd.read_csv(
                file_path,
                sep='\t',
                skiprows=1,  # Saltar la primera línea con encabezados
                names=self.COLUMN_NAMES,
                engine='python'
            )
            
            # Extraer información del nombre del archivo
            file_info = self._parse_filename(file_path.name)
            
            # Agregar metadatos
            df['subject_id'] = file_info['subject_id']
            df['session_id'] = file_info['session_id']
            df['trial_type'] = file_info['trial_type']
            df['filename'] = file_path.name
            
            # Crear etiqueta descriptiva del freezing
            df['freezing_label'] = df['freezing_flag'].map({
                0: 'No Freeze',
                1: 'Freeze'
            })
            
            return df
            
        except Exception as e:
            raise Exception(f"Error leyendo archivo {file_path}: {str(e)}")
    
    @staticmethod
    def _parse_filename(filename: str) -> Dict[str, any]:
        """
        Extrae información del nombre del archivo.
        Formato: SUB<subject>_<session>.txt (ej: SUB01_1.txt, SUB01_standing.txt)
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Diccionario con subject_id, session_id y trial_type
        """
        # Remover extensión
        name = filename.replace('.txt', '').replace('.csv', '')
        
        # Dividir por underscore
        parts = name.split('_')
        
        # Extraer subject (SUB01 -> 1)
        subject_id = int(parts[0].replace('SUB', ''))
        
        # Determinar tipo de trial y sesión
        if 'standing' in parts[1].lower():
            trial_type = 'standing'
            session_id = 0  # Standing no tiene número de sesión
        else:
            trial_type = 'walking'
            session_id = int(parts[1])
        
        return {
            'subject_id': subject_id,
            'session_id': session_id,
            'trial_type': trial_type
        }


class FigshareDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Figshare de Parkinson's y FoG."""
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta a la carpeta 'Figshare a public dataset'
        """
        super().__init__(dataset_path)
        self.imu_path = self.dataset_path / '2 - IMU'
        self.metadata_path = self.dataset_path / 'PDFEinfo_cleaned.csv'
        self.file_reader = FigshareFileReader()
        self.metadata: Optional[pd.DataFrame] = None
        
        # Validar ruta IMU
        if not self.imu_path.exists():
            raise FileNotFoundError(f"La ruta {self.imu_path} no existe")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Carga el archivo de metadata (PDFEinfo_cleaned.csv).
        
        Returns:
            DataFrame con la información demográfica y clínica de los pacientes
        """
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            return self.metadata
        except Exception as e:
            raise Exception(f"Error cargando metadata: {str(e)}")
    
    def get_file_list(self, trial_type: Optional[str] = None, **kwargs) -> List[Path]:
        """
        Obtiene lista de todos los archivos .txt en el dataset.
        
        Args:
            trial_type: Filtrar por tipo de trial ('walking', 'standing', None=todos)
        
        Returns:
            Lista de rutas a archivos .txt
        """
        files = sorted(self.imu_path.glob('SUB*.txt'))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos .txt en {self.imu_path}"
            )
        
        # Filtrar por tipo de trial si se especifica
        if trial_type:
            if trial_type.lower() == 'walking':
                files = [f for f in files if 'standing' not in f.name.lower()]
            elif trial_type.lower() == 'standing':
                files = [f for f in files if 'standing' in f.name.lower()]
        
        return files
    
    def load_all_data(self, verbose: bool = True, trial_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga todos los archivos TXT del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar barra de progreso
            trial_type: Filtrar por tipo de trial ('walking', 'standing', None=todos)
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        files = self.get_file_list(trial_type=trial_type)
        
        if verbose:
            print(f"📁 Encontrados {len(files)} archivos TXT")
            if trial_type:
                print(f"🔍 Filtrando por tipo: {trial_type}")
            print(f"📊 Cargando datos del dataset Figshare...\n")
        
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
    
    def load_subject_data(self, subject_id: int, trial_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga datos de un sujeto específico.
        
        Args:
            subject_id: ID del sujeto (1-35)
            trial_type: Filtrar por tipo de trial ('walking', 'standing', None=todos)
            **kwargs: Parámetros adicionales (no utilizados)
            
        Returns:
            DataFrame con los datos del sujeto
        """
        # Obtener archivos del sujeto
        pattern = f"SUB{subject_id:02d}_*.txt"
        files = sorted(self.imu_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para el sujeto {subject_id}")
        
        # Filtrar por tipo de trial si se especifica
        if trial_type:
            if trial_type.lower() == 'walking':
                files = [f for f in files if 'standing' not in f.name.lower()]
            elif trial_type.lower() == 'standing':
                files = [f for f in files if 'standing' in f.name.lower()]
        
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
            'freezing_flag': ['sum', 'count'],  # Total de freezes y muestras
            'session_id': 'nunique',  # Número de sesiones
            'filename': 'nunique'  # Número de archivos
        }).round(2)
        
        # Renombrar columnas
        summary.columns = ['duration_s', 'freeze_count', 'total_samples', 
                          'n_sessions', 'n_files']
        
        # Calcular porcentaje de freeze
        summary['freeze_percentage'] = (
            (summary['freeze_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_trial_type(self) -> pd.DataFrame:
        """
        Genera un resumen por tipo de trial (walking vs standing).
        
        Returns:
            DataFrame con estadísticas por tipo de trial
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('trial_type').agg({
            'subject_id': 'nunique',
            'filename': 'nunique',
            'time_s': 'max',
            'freezing_flag': ['sum', 'count']
        }).round(2)
        
        summary.columns = ['n_subjects', 'n_files', 'total_duration_s', 
                          'freeze_count', 'total_samples']
        
        summary['freeze_percentage'] = (
            (summary['freeze_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary


if __name__ == "__main__":
    # Ejemplo de uso
    DATASET_PATH = r'Datasets\Figshare a public dataset'
    
    loader = FigshareDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True)
    
    print("\n" + "="*60)
    print("RESUMEN POR SUJETO")
    print("="*60)
    print(loader.get_summary_by_subject().head())
    
    print("\n" + "="*60)
    print("RESUMEN POR TIPO DE TRIAL")
    print("="*60)
    print(loader.get_summary_by_trial_type())
    
    # Guardar dataset (descomenta para ejecutar)
    # loader.save_dataset('figshare_complete_dataset', formats=['csv', 'parquet'])
