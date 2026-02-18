# Interfaz Estándar de Loaders

Este documento describe la interfaz común que todos los loaders de datasets implementan.

## Arquitectura

Todos los loaders siguen una arquitectura de dos niveles:

1. **BaseFileReader**: Clase abstracta para leer archivos individuales
2. **BaseDatasetLoader**: Clase abstracta para cargar datasets completos

```
BaseFileReader (ABC)
├── DaphnetFileReader
├── FigshareFileReader
├── ChariteFileReader
├── MendelayFileReader
└── KaggleFileReader

BaseDatasetLoader (ABC)
├── DaphnetDatasetLoader
├── FigshareDatasetLoader
├── ChariteDatasetLoader
├── MendelayDatasetLoader
└── KaggleDatasetLoader
```

## Métodos Obligatorios (Abstractos)

### BaseFileReader

Todos los FileReaders deben implementar:

```python
@abstractmethod
def read_file(self, file_path: Path) -> pd.DataFrame:
    """Lee un archivo individual del dataset."""
    pass

@staticmethod
@abstractmethod
def _parse_filename(filename: str) -> Dict[str, any]:
    """Extrae información del nombre del archivo."""
    pass
```

### BaseDatasetLoader

Todos los DatasetLoaders deben implementar:

```python
@abstractmethod
def get_file_list(self, **kwargs) -> List[Path]:
    """Obtiene la lista de archivos a procesar."""
    pass

@abstractmethod
def load_all_data(self, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Carga todos los archivos del dataset."""
    pass
```

## Métodos Comunes (Con Implementación Base)

Todos los loaders heredan estos métodos de `BaseDatasetLoader`:

### Carga de Datos

```python
def load_subject_data(self, subject_id: Union[int, str], **kwargs) -> pd.DataFrame:
    """
    Carga datos de un sujeto específico.
    
    Implementación por defecto: carga todo y filtra por subject_id.
    Los loaders pueden sobrescribir este método para cargas más eficientes.
    """
```

### Información y Resúmenes

```python
def get_basic_info(self) -> Dict[str, any]:
    """Obtiene información básica del dataset: dimensiones, memoria, columnas."""

def get_subjects(self) -> List[Union[int, str]]:
    """Obtiene lista de sujetos únicos en el dataset."""

def get_fog_label_column(self) -> str:
    """Identifica la columna de etiquetas de FoG en el dataset."""

def get_summary_by_subject(self) -> pd.DataFrame:
    """
    Genera resumen estadístico por sujeto.
    
    Implementación por defecto disponible. Los loaders pueden sobrescribir
    para agregar información específica del dataset.
    """

def print_summary(self):
    """Imprime un resumen completo del dataset cargado."""
```

### Persistencia

```python
def save_dataset(self, 
                 output_path: str = None,
                 formats: List[str] = ['csv', 'parquet', 'pickle']) -> List[str]:
    """Guarda el dataset en uno o más formatos."""

def read_dataset(self, file_path: str) -> pd.DataFrame:
    """Lee un dataset previamente guardado desde un archivo (CSV, Parquet o Pickle)."""
```

### Visualización

```python
def plot_fog_distribution(self):
    """
    Genera visualización básica de la distribución de FoG.
    
    Puede ser sobrescrita para visualizaciones específicas del dataset.
    """
```

## Factory Function

Para facilitar la creación de loaders sin importar clases específicas:

```python
def load_dataset(dataset_type: str, dataset_path: str, **kwargs) -> BaseDatasetLoader:
    """
    Función factory para crear el loader apropiado.
    
    Args:
        dataset_type: 'daphnet', 'figshare', 'charite', 'mendeley', 'kaggle'
        dataset_path: Ruta al dataset
        **kwargs: Argumentos adicionales para el loader
    
    Returns:
        Instancia del loader apropiado
    
    Example:
        loader = load_dataset('daphnet', 'path/to/daphnet')
        df = loader.load_all_data()
    """
```

## Ejemplos de Uso

### Uso Básico

```python
from loaders import DaphnetDatasetLoader

# Crear loader
loader = DaphnetDatasetLoader('Datasets/Daphnet fog/dataset')

# Cargar todos los datos
df = loader.load_all_data(verbose=True)

# Ver información
loader.print_summary()
info = loader.get_basic_info()

# Obtener lista de sujetos
subjects = loader.get_subjects()

# Cargar un sujeto específico
subject_1_data = loader.load_subject_data(subject_id=1)

# Guardar dataset
loader.save_dataset('output/daphnet_dataset', formats=['csv', 'parquet'])

# Leer dataset previamente guardado
loader_new = DaphnetDatasetLoader('Datasets/Daphnet fog/dataset')
df = loader_new.read_dataset('output/daphnet_dataset.csv')
```

### Uso con Factory Function

```python
from loaders import load_dataset

# Crear loader usando factory
loader = load_dataset('daphnet', 'Datasets/Daphnet fog/dataset')

# Resto del código igual
df = loader.load_all_data()
```

### Cargar Múltiples Datasets

```python
from loaders import load_dataset

datasets = {
    'daphnet': 'Datasets/Daphnet fog/dataset',
    'figshare': 'Datasets/Figshare a public dataset',
    'charite': 'Datasets/Charité–Universitätsmedizin Berlin',
    'mendeley': 'Datasets/FOG - Mendeley Data Raw Data/Filtered',
    'kaggle': 'Datasets/Kaggle Michael J Fox Foundation Dataset'
}

# Cargar todos los datasets
loaded_data = {}
for name, path in datasets.items():
    print(f"\n=== Cargando {name.upper()} ===")
    loader = load_dataset(name, path)
    loaded_data[name] = loader.load_all_data(verbose=True)
```

## Extensibilidad

Para agregar un nuevo dataset, simplemente:

1. Crear una clase que herede de `BaseFileReader` e implementar:
   - `read_file()`
   - `_parse_filename()`

2. Crear una clase que herede de `BaseDatasetLoader` e implementar:
   - `get_file_list()`
   - `load_all_data()`
   - Opcionalmente sobrescribir otros métodos para funcionalidad específica

3. Registrar el nuevo loader en la factory function `load_dataset()`

## Métodos Específicos por Dataset

Además de la interfaz común, algunos loaders tienen métodos adicionales:

### FigshareDatasetLoader

```python
def load_metadata(self) -> pd.DataFrame:
    """Carga el archivo de metadata (PDFEinfo_cleaned.csv)."""

def get_summary_by_trial_type(self) -> pd.DataFrame:
    """Genera resumen por tipo de trial (walking vs standing)."""
```

### ChariteDatasetLoader

```python
def get_summary_by_foot(self) -> pd.DataFrame:
    """Genera resumen por pie (left vs right)."""
```

### MendelayDatasetLoader

```python
def get_summary_by_task(self) -> pd.DataFrame:
    """Genera resumen por tarea."""
```

### KaggleDatasetLoader

```python
def load_file_data(self, file_id: str) -> pd.DataFrame:
    """Carga datos de un archivo específico por su ID."""

def get_summary_by_dataset_type(self) -> pd.DataFrame:
    """Genera resumen por tipo de dataset (defog, tdcsfog, notype)."""

def get_summary_by_fog_type(self) -> pd.DataFrame:
    """Genera resumen por tipo de FoG (StartHesitation, Turn, Walking)."""
```

## Convenciones de Nombres de Columnas

Para mantener consistencia, los datasets usan estos nombres de columnas comunes:

- `subject_id`: Identificador del sujeto/paciente
- `filename`: Nombre del archivo fuente
- `fog_label` o `freezing_flag` o `annotation`: Etiqueta de FoG
- `*_label_text`: Versión legible de las etiquetas

El método `get_fog_label_column()` detecta automáticamente la columna correcta.
