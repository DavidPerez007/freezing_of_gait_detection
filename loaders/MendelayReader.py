"""
Mendeley Data - Multimodal Dataset of Freezing of Gait in Parkinson's Disease Loader

Dataset Information:
- Source: Mendeley Data - Multimodal Dataset of Freezing of Gait
- Subjects: 12 patients (001-012)
- Tasks: 4 tasks per subject
- Sensors: Multiple IMU sensors with accelerometer and gyroscope data
- Format: TXT files with comma-separated values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader


class MendelayFileReader(BaseFileReader):
    """Lee archivos individuales del dataset Mendeley FoG."""
    
    # Nombres de columnas basados en la estructura del archivo
    # El dataset tiene 61 columnas con datos de sensores IMU
    COLUMN_NAMES = [
        'frame_id',
        'timestamp',
        # Primeras 24 columnas parecen ser datos de sensores (acelerómetro/giroscopio)
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
        'sensor_19', 'sensor_20', 'sensor_21', 'sensor_22', 'sensor_23', 'sensor_24',
        # Siguientes columnas parecen ser datos adicionales de sensores
        'raw_1', 'raw_2', 'raw_3', 'raw_4', 'raw_5', 'raw_6',
        'raw_7', 'raw_8', 'raw_9', 'raw_10', 'raw_11', 'raw_12', 'raw_13',
        'raw_14', 'raw_15', 'raw_16', 'raw_17', 'raw_18', 'raw_19', 'raw_20',
        'raw_21', 'raw_22', 'raw_23', 'raw_24', 'raw_25', 'raw_26', 'raw_27',
        'raw_28', 'raw_29', 'raw_30', 'raw_31', 'raw_32', 'raw_33',
        'fog_label'  # Última columna: etiqueta de FoG (0 o 1)
    ]
    
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo TXT del dataset Mendeley.
        
        Args:
            file_path: Ruta al archivo .txt
            
        Returns:
            DataFrame con los datos del archivo
        """
        try:
            # Leer archivo CSV (separado por comas)
            df = pd.read_csv(
                file_path,
                header=None,
                names=self.COLUMN_NAMES
            )
            
            # Extraer información del nombre del archivo
            file_info = self._parse_filename(file_path)
            
            # Agregar metadatos
            df['subject_id'] = file_info['subject_id']
            df['task_id'] = file_info['task_id']
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
    def _parse_filename(file_path: Path) -> Dict[str, any]:
        """
        Extrae información del path del archivo.
        Formato: .../001/task_1.txt
        
        Args:
            file_path: Path completo al archivo
            
        Returns:
            Diccionario con subject_id y task_id
        """
        # Obtener el nombre del directorio padre (subject ID)
        subject_folder = file_path.parent.name  # '001', '002', etc.
        subject_id = int(subject_folder)
        
        # Obtener el task del nombre del archivo
        filename = file_path.stem  # 'task_1', 'task_2', etc.
        task_id = int(filename.split('_')[1])
        
        return {
            'subject_id': subject_id,
            'task_id': task_id
        }


class MendelayDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Mendeley Multimodal FoG."""
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta a la carpeta 'Filtered' del dataset Mendeley
        """
        super().__init__(dataset_path)
        self.file_reader = MendelayFileReader()
    
    def get_file_list(self, **kwargs) -> List[Path]:
        """
        Obtiene lista de todos los archivos TXT en el dataset.
        
        Returns:
            Lista de rutas a archivos TXT
        """
        # Buscar en todas las carpetas de sujetos
        files = sorted(self.dataset_path.glob('*/task_*.txt'))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos TXT en {self.dataset_path}"
            )
        
        return files
    
    def load_all_data(self, verbose: bool = True, **kwargs) -> pd.DataFrame:
        """
        Carga todos los archivos TXT del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar barra de progreso
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        files = self.get_file_list()
        
        if verbose:
            print(f"📁 Encontrados {len(files)} archivos TXT")
            print(f"📊 Cargando datos del dataset Mendeley FoG...\n")
        
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
    
    def load_subject_data(self, subject_id: int, **kwargs) -> pd.DataFrame:
        """
        Carga datos de un sujeto específico.
        
        Args:
            subject_id: ID del sujeto (1-12)
            **kwargs: Parámetros adicionales (no utilizados)
            
        Returns:
            DataFrame con los datos del sujeto
        """
        # Buscar carpeta del sujeto
        subject_folder = self.dataset_path / f"{subject_id:03d}"
        
        if not subject_folder.exists():
            raise FileNotFoundError(f"No se encontró la carpeta para el sujeto {subject_id}")
        
        # Obtener archivos del sujeto
        files = sorted(subject_folder.glob("task_*.txt"))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para el sujeto {subject_id}")
        
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
            'fog_label': ['sum', 'count'],
            'task_id': 'nunique',
            'filename': 'nunique'
        }).round(2)
        
        # Renombrar columnas
        summary.columns = ['fog_count', 'total_samples', 'n_tasks', 'n_files']
        
        # Calcular porcentaje de FoG
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_task(self) -> pd.DataFrame:
        """
        Genera un resumen por tarea.
        
        Returns:
            DataFrame con estadísticas por tarea
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('task_id').agg({
            'subject_id': 'nunique',
            'filename': 'nunique',
            'fog_label': ['sum', 'count']
        }).round(2)
        
        summary.columns = ['n_subjects', 'n_files', 'fog_count', 'total_samples']
        
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary


if __name__ == "__main__":
    """
    Ejemplos de uso del MendelayDatasetLoader
    
    Dataset: 11 sujetos, 43 archivos, 4+ millones de muestras
    Formatos de salida: 'csv', 'parquet', 'pickle'
    """
    
    DATASET_PATH = r"Datasets\FOG - Mendeley Data Raw Data Multimodal Dataset of Freezing of Gait in Parkinson's Disease\Filtered"
    
    # === EJEMPLO 1: Cargar dataset completo ===
    print("\n" + "="*70)
    print(" CARGANDO DATASET MENDELEY COMPLETO")
    print("="*70)
    
    loader = MendelayDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True)
    
    print("\n" + "="*60)
    print("RESUMEN POR SUJETO")
    print("="*60)
    print(loader.get_summary_by_subject())
    
    print("\n" + "="*60)
    print("RESUMEN POR TAREA")
    print("="*60)
    print(loader.get_summary_by_task())
    
    # Guardar dataset (descomenta la línea siguiente para ejecutar)
    # print("\n💾 Guardando dataset...")
    # loader.save_dataset('mendeley_complete_dataset', formats=['csv'])
    # print("✅ Dataset guardado: mendeley_complete_dataset.csv")
    
    
    # === EJEMPLO 2: Cargar solo un sujeto (comentado por defecto) ===
    # print("\n" + "="*70)
    # print(" EJEMPLO 2: Cargando solo sujeto 1")
    # print("="*70)
    # 
    # df_subject = loader.load_subject_data(subject_id=1)
    # print(f"Sujeto 1: {df_subject.shape}")
    # print(f"Tasks: {df_subject['task_id'].unique()}")
    # print(f"FoG episodes: {df_subject['fog_label'].sum()}")    
    # 
    # # Guardar solo este sujeto
    # df_subject.to_csv('mendeley_subject_01.csv', index=False)
