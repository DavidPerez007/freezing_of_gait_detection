"""
Daphnet Freezing of Gait Dataset Loader

Dataset Information:
- Source: Laboratory for Gait and Neurodynamics, Tel Aviv Sourasky Medical Center
- Sensors: 3 acelerómetros 3D a 64 Hz
- Ubicación: Tobillo (shank), Muslo (thigh), Tronco (trunk)
- Sujetos: 10 pacientes con Parkinson's (S01-S10)
- Anotaciones: 0=No experimento, 1=Caminar sin freeze, 2=Freeze
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader

# Importación absoluta desde el root del proyecto
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.daphnetConfig import DaphnetConfig


class DaphnetFileReader(BaseFileReader):
    """Clase para leer archivos individuales del dataset Daphnet."""
    
    def __init__(self, config: DaphnetConfig = DaphnetConfig()):
        """
        Inicializa el lector de archivos.
        
        Args:
            config: Configuración del dataset
        """
        self.config = config
    
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo del dataset Daphnet.
        
        Args:
            file_path: Ruta al archivo .txt
            
        Returns:
            DataFrame con los datos del archivo
        """
        try:
            # Leer archivo sin encabezado, separado por espacios
            df = pd.read_csv(
                file_path,
                sep='\s+',
                header=None,
                names=self.config.COLUMN_NAMES,
                engine='python'
            )
            
            # Extraer información del nombre del archivo
            file_info = self._parse_filename(file_path.name)
            
            # Agregar metadatos
            df['subject_id'] = file_info['subject_id']
            df['run_id'] = file_info['run_id']
            df['filename'] = file_path.name
            
            # Crear etiqueta descriptiva de la anotación
            df['label'] = df['annotation']
            
            # Convertir tiempo a segundos
            df['time_s'] = df['time_ms'] / 1000.0
            
            return df
            
        except Exception as e:
            raise Exception(f"Error leyendo archivo {file_path}: {str(e)}")
    
    @staticmethod
    def _parse_filename(filename: str) -> Dict[str, int]:
        """
        Extrae información del nombre del archivo.
        Formato: S<subject>R<run>.txt (ej: S01R01.txt)
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Diccionario con subject_id y run_id
        """
        # Remover extensión
        name = filename.replace('.txt', '')
        
        # Extraer subject y run
        subject_id = int(name[1:3])  # S01 -> 1
        run_id = int(name[4:6])      # R01 -> 1
        
        return {
            'subject_id': subject_id,
            'run_id': run_id
        }
        
class DaphnetDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Daphnet Freezing of Gait."""
    
    def __init__(self, dataset_path: str, config: DaphnetConfig = None):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta a la carpeta 'dataset' del Daphnet
            config: Configuración del dataset (opcional)
        """
        super().__init__(dataset_path)
        self.config = config or DaphnetConfig()
        self.file_reader = DaphnetFileReader(self.config)
    
    def get_file_list(self, **kwargs) -> List[Path]:
        """
        Obtiene lista de todos los archivos .txt en el dataset.
        
        Returns:
            Lista de rutas a archivos .txt
        """
        files = sorted(self.dataset_path.glob('S*.txt'))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos .txt en {self.dataset_path}"
            )
        
        return files
    
    def load_all_data(self, verbose: bool = True, **kwargs) -> pd.DataFrame:
        """
        Carga todos los archivos del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar barra de progreso
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        files = self.get_file_list()
        
        if verbose:
            print(f"📁 Encontrados {len(files)} archivos")
            print(f"📊 Cargando datos del dataset Daphnet...\n")
        
        # Leer todos los archivos
        dataframes = []
        iterator = tqdm(files, desc="Cargando archivos") if verbose else files
        
        for file_path in iterator:
            df = self.file_reader.read_file(file_path)
            dataframes.append(df)
        
        # Concatenar todos los DataFrames
        self.data = pd.concat(dataframes, ignore_index=True)
        
        if verbose:
            print(f"\n✅ Dataset cargado exitosamente")
            self.print_summary()
        
        return self.data
    
    def load_subject_data(self, subject_id: int, **kwargs) -> pd.DataFrame:
        """
        Carga datos de un sujeto específico (todas sus corridas).
        
        Args:
            subject_id: ID del sujeto (1-10)
            **kwargs: Parámetros adicionales (no utilizados)
            
        Returns:
            DataFrame con datos del sujeto
        """
        files = [f for f in self.get_file_list() 
                if f.name.startswith(f'S{subject_id:02d}')]
        
        if not files:
            raise ValueError(f"No se encontraron archivos para el sujeto {subject_id}")
        
        dataframes = [self.file_reader.read_file(f) for f in files]
        return pd.concat(dataframes, ignore_index=True)
    
    def get_summary_by_subject(self) -> pd.DataFrame:
        """
        Obtiene un resumen estadístico por sujeto.
        
        Returns:
            DataFrame con estadísticas por sujeto
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('subject_id').agg({
            'run_id': 'nunique',
            'time_s': 'max',
            'annotation': lambda x: (x == 2).sum(),  # Contar freezes
            'filename': 'count'
        }).rename(columns={
            'run_id': 'num_runs',
            'time_s': 'duration_s',
            'annotation': 'freeze_samples',
            'filename': 'total_samples'
        })
        
        summary['freeze_percentage'] = (
            (summary['freeze_samples'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
        
        summary['freeze_percentage'] = (
            (summary['freeze_samples'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary


if __name__ == "__main__":
    # Ejemplo de uso
    DATASET_PATH = r'Datasets\Daphnet fog\dataset'
    
    loader = DaphnetDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True)
    
    print("\n" + "="*60)
    print("RESUMEN POR SUJETO")
    print("="*60)
    print(loader.get_summary_by_subject())
    
    # Guardar dataset (descomenta para ejecutar)
    # loader.save_dataset('daphnet_complete_dataset', formats=['csv', 'parquet'])