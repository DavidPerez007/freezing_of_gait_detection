"""
Base classes for loading and processing Freezing of Gait (FoG) datasets.

This module provides abstract base classes and common utilities for loading
various FoG datasets in a standardized way, promoting code reusability and
consistency across different dataset implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración global de visualización
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class BaseFileReader(ABC):
    """
    Clase abstracta base para leer archivos individuales de datasets.
    
    Cada dataset específico debe implementar sus propios métodos de lectura
    heredando de esta clase.
    """
    
    @abstractmethod
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo individual del dataset.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            DataFrame con los datos del archivo, incluyendo metadatos
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _parse_filename(filename: str) -> Dict[str, any]:
        """
        Extrae información del nombre del archivo.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Diccionario con metadatos extraídos del nombre
        """
        pass


class BaseDatasetLoader(ABC):
    """
    Clase abstracta base para cargar datasets completos de FoG.
    
    Proporciona funcionalidad común para todos los loaders de datasets,
    incluyendo carga de datos, generación de resúmenes y guardado.
    """
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta al directorio del dataset
        """
        self.dataset_path = Path(dataset_path)
        self.data: Optional[pd.DataFrame] = None
        
        # Validar que la ruta existe
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"La ruta {self.dataset_path} no existe")
    
    @abstractmethod
    def get_file_list(self, **kwargs) -> List[Path]:
        """
        Obtiene la lista de archivos a procesar del dataset.
        
        Returns:
            Lista de rutas a archivos
        """
        pass
    
    @abstractmethod
    def load_all_data(self, verbose: bool = True, **kwargs) -> pd.DataFrame:
        """
        Carga todos los archivos del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar información de progreso
            **kwargs: Parámetros adicionales específicos del dataset
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        pass
    
    def load_subject_data(self, subject_id: Union[int, str], **kwargs) -> pd.DataFrame:
        """
        Carga datos de un sujeto específico.
        
        Args:
            subject_id: ID del sujeto (puede ser int o string dependiendo del dataset)
            **kwargs: Parámetros adicionales específicos del dataset
            
        Returns:
            DataFrame con los datos del sujeto
        """
        # Implementación por defecto: cargar todo y filtrar
        if self.data is None:
            self.load_all_data(verbose=False, **kwargs)
        
        # Intentar filtrar por subject_id si existe la columna
        if 'subject_id' in self.data.columns:
            return self.data[self.data['subject_id'] == subject_id].copy()
        else:
            raise NotImplementedError(
                "Este dataset no tiene columna 'subject_id' o requiere "
                "implementación específica de load_subject_data()"
            )
    
    def read_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Lee un dataset previamente guardado desde un archivo.
        Soporta formatos CSV, Parquet y Pickle.
        
        Args:
            file_path: Ruta al archivo (con extensión)
            
        Returns:
            DataFrame con los datos cargados
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
            
        Example:
            loader = DaphnetDatasetLoader('path')
            df = loader.read_dataset('outputs/daphnet_dataset.csv')
        """
        file_path = Path(file_path)
        
        # Verificar que el archivo existe
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        # Detectar formato por extensión
        extension = file_path.suffix.lower()
        
        print(f"📂 Cargando dataset desde: {file_path.name}")
        
        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif extension in ['.pkl', '.pickle']:
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(
                f"Formato {extension} no soportado. "
                f"Usa: .csv, .parquet, .pkl o .pickle"
            )
        
        # Actualizar self.data con los datos cargados
        self.data = df
        
        print(f"✅ Dataset cargado exitosamente")
        print(f"   📊 Dimensiones: {df.shape}")
        print(f"   💽 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def save_dataset(self,
                     output_path: str = None,
                     format: str = 'csv') -> pd.DataFrame:
        """
        Save dataset to disk and return DataFrame.

        Args:
            output_path: Base path (without extension)
            format: 'csv', 'parquet', or 'pickle'

        Returns:
            pd.DataFrame: The dataset
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_all_data() first")

        if output_path is None:
            output_path = f"{self.__class__.__name__.replace('Loader', '').lower()}_dataset"

        # Save in specified format
        if format == 'csv':
            file_path = f"{output_path}.csv"
            self.data.to_csv(file_path, index=False)
        elif format == 'parquet':
            file_path = f"{output_path}.parquet"
            self.data.to_parquet(file_path, index=False)
        elif format == 'pickle':
            file_path = f"{output_path}.pkl"
            self.data.to_pickle(file_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        file_size = Path(file_path).stat().st_size / 1024**2
        print(f"Saved: {file_path} ({file_size:.1f} MB)")
        
        return self.data
    
    def get_basic_info(self) -> Dict[str, any]:
        """
        Obtiene información básica del dataset cargado.
        
        Returns:
            Diccionario con información del dataset
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        info = {
            'shape': self.data.shape,
            'n_rows': len(self.data),
            'n_cols': len(self.data.columns),
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.data.columns)
        }
        
        # Agregar información de sujetos si está disponible
        if 'subject_id' in self.data.columns:
            info['n_subjects'] = self.data['subject_id'].nunique()
        
        # Agregar información de archivos si está disponible
        if 'filename' in self.data.columns:
            info['n_files'] = self.data['filename'].nunique()
        
        return info
    
    def get_subjects(self) -> List[Union[int, str]]:
        """
        Obtiene la lista de sujetos únicos en el dataset.
        
        Returns:
            Lista de IDs de sujetos
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        if 'subject_id' not in self.data.columns:
            raise ValueError("El dataset no tiene columna 'subject_id'")
        
        return sorted(self.data['subject_id'].unique().tolist())
    
    def get_fog_label_column(self) -> str:
        """
        Identifica la columna de etiquetas de FoG en el dataset.
        
        Returns:
            Nombre de la columna de etiquetas
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        # Buscar columna de etiquetas en orden de prioridad
        label_candidates = ['fog_label', 'freezing_flag', 'annotation', 'fog_any']
        
        for col in label_candidates:
            if col in self.data.columns:
                return col
        
        raise ValueError(
            f"No se encontró columna de etiqueta de FoG. "
            f"Buscado: {label_candidates}"
        )
    
    def get_summary_by_subject(self) -> pd.DataFrame:
        """
        Genera resumen estadístico por sujeto.
        Debe ser implementado o sobrescrito por clases hijas si necesitan
        lógica específica.
        
        Returns:
            DataFrame con estadísticas por sujeto
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        if 'subject_id' not in self.data.columns:
            raise ValueError("El dataset no tiene columna 'subject_id'")
        
        # Obtener columna de etiqueta usando método estándar
        label_col = self.get_fog_label_column()
        
        # Generar resumen básico
        agg_dict = {label_col: ['sum', 'count']}
        
        # Agregar filename si existe
        if 'filename' in self.data.columns:
            agg_dict['filename'] = 'nunique'
        
        summary = self.data.groupby('subject_id').agg(agg_dict).round(2)
        
        # Renombrar columnas
        if 'filename' in self.data.columns:
            summary.columns = ['fog_count', 'total_samples', 'n_files']
        else:
            summary.columns = ['fog_count', 'total_samples']
        
        summary['fog_percentage'] = (
            (summary['fog_count'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def plot_fog_distribution(self):
        """
        Genera visualización básica de la distribución de FoG.
        Puede ser sobrescrita por clases hijas para visualizaciones específicas.
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        # Obtener columna de etiqueta usando método estándar
        label_col = self.get_fog_label_column()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribución general
        if f'{label_col}_text' in self.data.columns:
            counts = self.data[f'{label_col}_text'].value_counts()
        else:
            counts = self.data[label_col].value_counts()
        
        axes[0].bar(range(len(counts)), counts.values, color=['green', 'red'])
        axes[0].set_xticks(range(len(counts)))
        axes[0].set_xticklabels(counts.index, rotation=45)
        axes[0].set_xlabel('Estado')
        axes[0].set_ylabel('Número de Muestras')
        axes[0].set_title('Distribución General de FoG')
        
        # Por sujeto
        if 'subject_id' in self.data.columns:
            summary = self.get_summary_by_subject()
            axes[1].bar(summary.index, summary['fog_percentage'], 
                       color='steelblue', edgecolor='black')
            axes[1].set_xlabel('Subject ID')
            axes[1].set_ylabel('Porcentaje de FoG (%)')
            axes[1].set_title('Porcentaje de FoG por Sujeto')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """
        Imprime un resumen completo del dataset cargado.
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        print("\n" + "="*60)
        print(f"RESUMEN DEL DATASET: {self.__class__.__name__}")
        print("="*60)
        
        info = self.get_basic_info()
        print(f"\n📊 Información General:")
        print(f"   Dimensiones: {info['shape']}")
        print(f"   Filas: {info['n_rows']:,}")
        print(f"   Columnas: {info['n_cols']}")
        print(f"   Memoria: {info['memory_mb']:.2f} MB")
        
        if 'subject_id' in self.data.columns:
            print(f"\n👥 Sujetos: {self.data['subject_id'].nunique()}")
        
        if 'filename' in self.data.columns:
            print(f"📁 Archivos: {self.data['filename'].nunique()}")
        
        # Información de FoG
        try:
            label_col = self.get_fog_label_column()
            
            # Para datasets con múltiples clases (como Daphnet), contar solo FoG positivos
            if label_col == 'annotation':
                # En Daphnet: 0=No experimento, 1=Sin freeze, 2=Freeze
                fog_count = (self.data[label_col] == 2).sum()
            else:
                fog_count = self.data[label_col].sum()
            
            fog_pct = (fog_count / len(self.data)) * 100
            print(f"\n🚨 Episodios de FoG:")
            print(f"   Muestras con FoG: {fog_count:,} ({fog_pct:.2f}%)")
            print(f"   Muestras sin FoG: {len(self.data) - fog_count:,} ({100-fog_pct:.2f}%)")
        except ValueError:
            pass  # No hay columna de FoG, skip
        
        print("="*60 + "\n")


def load_dataset(dataset_type: str, dataset_path: str, **kwargs) -> BaseDatasetLoader:
    """
    Función factory para crear el loader apropiado según el tipo de dataset.
    
    Args:
        dataset_type: Tipo de dataset ('daphnet', 'figshare', 'charite', 'mendeley', 'kaggle')
        dataset_path: Ruta al dataset
        **kwargs: Argumentos adicionales para el loader
        
    Returns:
        Instancia del loader apropiado
        
    Example:
        loader = load_dataset('daphnet', 'path/to/daphnet')
        df = loader.load_all_data()
    """
    from .DaphnetReader import DaphnetDatasetLoader
    from .FigshareReader import FigshareDatasetLoader
    from .ChariteReader import ChariteDatasetLoader
    from .MendelayReader import MendelayDatasetLoader
    from .KaggleReader import KaggleDatasetLoader
    
    loaders = {
        'daphnet': DaphnetDatasetLoader,
        'figshare': FigshareDatasetLoader,
        'charite': ChariteDatasetLoader,
        'mendeley': MendelayDatasetLoader,
        'kaggle': KaggleDatasetLoader,
    }
    
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower not in loaders:
        raise ValueError(f"Tipo de dataset '{dataset_type}' no soportado. "
                        f"Opciones: {list(loaders.keys())}")
    
    return loaders[dataset_type_lower](dataset_path, **kwargs)


if __name__ == "__main__":
    print("BaseDatasetLoader - Módulo de clases base para carga de datasets FoG")
    print("Este módulo no está diseñado para ejecutarse directamente.")
    print("\nUso:")
    print("  from BaseDatasetLoader import BaseDatasetLoader")
    print("  class MyDatasetLoader(BaseDatasetLoader):")
    print("      # Implementar métodos abstractos")
