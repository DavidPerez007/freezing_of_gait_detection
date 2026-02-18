"""
Kaggle Michael J Fox Foundation - FoG Dataset Loader

Dataset Information:
- Source: Kaggle - Freezing of Gait Prediction Competition
- Subjects: Múltiples pacientes con Parkinson's
- Sensors: 3D accelerometer (AccV, AccML, AccAP)
- Tasks: defog, tdcsfog, notype
- Etiquetas: StartHesitation, Turn, Walking
- Format: CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal
from tqdm import tqdm

from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader


class KaggleFileReader(BaseFileReader):
    """Lee archivos CSV del dataset Kaggle Michael J Fox Foundation."""
    
    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Lee un archivo CSV del dataset Kaggle.
        
        Args:
            file_path: Ruta al archivo .csv
            
        Returns:
            DataFrame con los datos del archivo
        """
        try:
            # Leer archivo CSV
            df = pd.read_csv(file_path)
            
            # Extraer información del archivo
            file_info = self._parse_filename(file_path)
            
            # Agregar metadatos
            df['file_id'] = file_info['file_id']
            df['dataset_type'] = file_info['dataset_type']
            df['subset'] = file_info['subset']
            df['filename'] = file_path.name
            
            # Crear columna de FoG combinada (cualquier tipo de FoG)
            df['fog_any'] = ((df['StartHesitation'] == 1) | 
                            (df['Turn'] == 1) | 
                            (df['Walking'] == 1)).astype(int)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error leyendo archivo {file_path}: {str(e)}")
    
    @staticmethod
    def _parse_filename(file_path: Path) -> Dict[str, any]:
        """
        Extrae información del path del archivo.
        Formato: .../train/defog/02ea782681.csv
        
        Args:
            file_path: Path completo al archivo
            
        Returns:
            Diccionario con file_id, dataset_type, subset
        """
        # Obtener ID del archivo (nombre sin extensión)
        file_id = file_path.stem
        
        # Obtener tipo de dataset (defog, tdcsfog, notype)
        dataset_type = file_path.parent.name
        
        # Obtener subset (train, test, unlabeled)
        subset = file_path.parent.parent.name
        
        return {
            'file_id': file_id,
            'dataset_type': dataset_type,
            'subset': subset
        }


class KaggleDatasetLoader(BaseDatasetLoader):
    """Loader para el dataset Kaggle Michael J Fox Foundation FoG."""
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador del dataset.
        
        Args:
            dataset_path: Ruta a la carpeta raíz del dataset Kaggle
        """
        super().__init__(dataset_path)
        self.file_reader = KaggleFileReader()
        
        # Cargar metadatos si existen
        self.subjects_metadata = self._load_metadata('subjects.csv')
        self.tasks_metadata = self._load_metadata('tasks.csv')
        self.defog_metadata = self._load_metadata('defog_metadata.csv')
        self.tdcsfog_metadata = self._load_metadata('tdcsfog_metadata.csv')
    
    def _load_metadata(self, filename: str) -> Optional[pd.DataFrame]:
        """Carga un archivo de metadata si existe."""
        metadata_path = self.dataset_path / filename
        if metadata_path.exists():
            return pd.read_csv(metadata_path)
        return None
    
    def get_file_list(
        self, 
        subset: Literal['train', 'test', 'unlabeled'] = 'train',
        dataset_type: Optional[Literal['defog', 'tdcsfog', 'notype']] = None
    ) -> List[Path]:
        """
        Obtiene lista de archivos CSV del dataset.
        
        Args:
            subset: Subset a cargar ('train', 'test', 'unlabeled')
            dataset_type: Tipo de dataset ('defog', 'tdcsfog', 'notype', None para todos)
            
        Returns:
            Lista de rutas a archivos CSV
        """
        files = []
        subset_path = self.dataset_path / subset
        
        if not subset_path.exists():
            raise FileNotFoundError(f"No se encontró el subset: {subset_path}")
        
        # Si se especifica un tipo de dataset específico
        if dataset_type:
            type_path = subset_path / dataset_type
            if type_path.exists():
                files.extend(sorted(type_path.glob('*.csv')))
        else:
            # Cargar todos los tipos
            for subdir in subset_path.iterdir():
                if subdir.is_dir():
                    files.extend(sorted(subdir.glob('*.csv')))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {subset_path}"
            )
        
        return files
    
    def load_all_data(
        self, 
        verbose: bool = True,
        subset: Literal['train', 'test', 'unlabeled'] = 'train',
        dataset_type: Optional[Literal['defog', 'tdcsfog', 'notype']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Carga todos los archivos CSV del dataset en un único DataFrame.
        
        Args:
            verbose: Mostrar barra de progreso
            subset: Subset a cargar ('train', 'test', 'unlabeled')
            dataset_type: Tipo de dataset a cargar (None para todos)
            
        Returns:
            DataFrame con todos los datos del dataset
        """
        files = self.get_file_list(subset=subset, dataset_type=dataset_type)
        
        if verbose:
            print(f"📁 Encontrados {len(files)} archivos CSV")
            print(f"📊 Subset: {subset}")
            if dataset_type:
                print(f"🔖 Tipo: {dataset_type}")
            print(f"📊 Cargando datos del dataset Kaggle FoG...\n")
        
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
    
    def load_file_data(self, file_id: str, **kwargs) -> pd.DataFrame:
        """
        Carga datos de un archivo específico por su ID.
        Alias: load_subject_data (para compatibilidad con base class)
        
        Args:
            file_id: ID del archivo (sin extensión)
            **kwargs: Parámetros adicionales (no utilizados)
            
        Returns:
            DataFrame con los datos del archivo
        """
        # Buscar el archivo en todos los subsets y tipos
        for subset in ['train', 'test', 'unlabeled']:
            subset_path = self.dataset_path / subset
            if not subset_path.exists():
                continue
            
            # Buscar en todos los subdirectorios
            for type_dir in subset_path.iterdir():
                if type_dir.is_dir():
                    file_path = type_dir / f"{file_id}.csv"
                    if file_path.exists():
                        return self.file_reader.read_file(file_path)
        
        raise FileNotFoundError(f"No se encontró el archivo con ID: {file_id}")
    
    # Alias para compatibilidad con la interfaz base
    def load_subject_data(self, subject_id: str, **kwargs) -> pd.DataFrame:
        """
        Alias de load_file_data para mantener compatibilidad con BaseDatasetLoader.
        
        Args:
            subject_id: ID del archivo (en Kaggle no hay sujetos, sino file_ids)
            **kwargs: Parámetros adicionales
            
        Returns:
            DataFrame con los datos del archivo
        """
        return self.load_file_data(subject_id, **kwargs)
    
    def get_summary_by_subject(self) -> pd.DataFrame:
        """
        Genera un resumen estadístico por archivo.
        
        Returns:
            DataFrame con estadísticas por archivo
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        # Agrupar por file_id
        summary = self.data.groupby('file_id').agg({
            'fog_any': ['sum', 'count'],
            'StartHesitation': 'sum',
            'Turn': 'sum',
            'Walking': 'sum',
            'Valid': 'sum',
            'dataset_type': 'first',
            'subset': 'first'
        }).round(2)
        
        # Renombrar columnas
        summary.columns = [
            'fog_total', 'total_samples', 'hesitation_count', 
            'turn_count', 'walking_count', 'valid_count',
            'dataset_type', 'subset'
        ]
        
        # Calcular porcentaje de FoG
        summary['fog_percentage'] = (
            (summary['fog_total'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_dataset_type(self) -> pd.DataFrame:
        """
        Genera un resumen por tipo de dataset (defog, tdcsfog, notype).
        
        Returns:
            DataFrame con estadísticas por tipo
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        summary = self.data.groupby('dataset_type').agg({
            'file_id': 'nunique',
            'fog_any': ['sum', 'count'],
            'StartHesitation': 'sum',
            'Turn': 'sum',
            'Walking': 'sum'
        }).round(2)
        
        summary.columns = [
            'n_files', 'fog_total', 'total_samples',
            'hesitation_count', 'turn_count', 'walking_count'
        ]
        
        summary['fog_percentage'] = (
            (summary['fog_total'] / summary['total_samples']) * 100
        ).round(2)
        
        return summary
    
    def get_summary_by_fog_type(self) -> pd.DataFrame:
        """
        Genera un resumen por tipo de FoG (StartHesitation, Turn, Walking).
        
        Returns:
            DataFrame con conteos por tipo de FoG
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_all_data()")
        
        fog_types = {
            'StartHesitation': self.data['StartHesitation'].sum(),
            'Turn': self.data['Turn'].sum(),
            'Walking': self.data['Walking'].sum(),
            'Any_FoG': self.data['fog_any'].sum(),
            'No_FoG': (self.data['fog_any'] == 0).sum()
        }
        
        total = len(self.data)
        
        summary_data = []
        for fog_type, count in fog_types.items():
            summary_data.append({
                'fog_type': fog_type,
                'count': int(count),
                'percentage': (count / total * 100)
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    """
    Ejemplos de uso del KaggleDatasetLoader
    
    Opciones de configuración:
    - subset: 'train', 'test', 'unlabeled'
    - dataset_type: None (todos), 'defog', 'tdcsfog', 'notype'
    - formats: ['csv'], ['parquet'], ['pickle'], o combinaciones
    """
    
    DATASET_PATH = r'Datasets\Kaggle Michael J Fox Foundation Dataset'
    
    # === EJEMPLO 1: Cargar solo train/defog (más rápido para pruebas) ===
    print("\n" + "="*70)
    print(" EJEMPLO 1: Cargando solo TRAIN/DEFOG (91 archivos)")
    print("="*70)
    
    loader = KaggleDatasetLoader(DATASET_PATH)
    df = loader.load_all_data(verbose=True, subset='train', dataset_type='defog')
    
    print("\n" + "="*60)
    print("RESUMEN POR TIPO DE FOG")
    print("="*60)
    print(loader.get_summary_by_fog_type())
    
    # Guardar dataset
    print("\n💾 Guardando dataset...")
    loader.save_dataset('kaggle_train_defog', formats=['csv'])
    print("✅ Dataset guardado: kaggle_train_defog.csv")
    
    
    # === EJEMPLO 2: Cargar TODO el train (comentado por defecto) ===
    # print("\n" + "="*70)
    # print(" EJEMPLO 2: Cargando TODO TRAIN (970 archivos)")
    # print("="*70)
    # 
    # loader_full = KaggleDatasetLoader(DATASET_PATH)
    # df_full = loader_full.load_all_data(verbose=True, subset='train')
    # 
    # print("\n" + "="*60)
    # print("RESUMEN POR TIPO DE DATASET")
    # print("="*60)
    # print(loader_full.get_summary_by_dataset_type())
    # 
    # print("\n" + "="*60)
    # print("RESUMEN POR TIPO DE FOG")
    # print("="*60)
    # print(loader_full.get_summary_by_fog_type())
    # 
    # # Guardar en múltiples formatos
    # loader_full.save_dataset('kaggle_train_complete', formats=['csv', 'parquet'])
    
    
    # === EJEMPLO 3: Cargar un archivo específico ===
    # print("\n" + "="*70)
    # print(" EJEMPLO 3: Cargar archivo específico por ID")
    # print("="*70)
    # 
    # df_single = loader.load_subject_data('02ea782681')
    # print(f"Archivo cargado: {df_single.shape}")
    # print(df_single.head())

