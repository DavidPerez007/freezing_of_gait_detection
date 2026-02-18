# FoG Dataset Loaders

Sistema profesional y escalable para cargar y procesar múltiples datasets de Freezing of Gait (FoG) en pacientes con Parkinson.

## 🎯 Características

- **Interfaz estandarizada** - Todos los loaders implementan los mismos métodos
- **Herencia y polimorfismo** - Arquitectura basada en clases abstractas (ABC)
- **5 datasets soportados** con API uniforme
- **Factory pattern** para creación dinámica de loaders
- **Funcionalidad común** (guardado, resúmenes, visualizaciones)
- **Código limpio** siguiendo principios SOLID y DRY
- **Fácil de extender** para nuevos datasets

## 🏗️ Arquitectura

Todos los loaders heredan de clases base abstractas que definen una interfaz común:

```
BaseFileReader (ABC)          BaseDatasetLoader (ABC)
     ├── read_file()               ├── get_file_list()
     └── _parse_filename()         ├── load_all_data()
                                   ├── load_subject_data()
                                   ├── get_basic_info()
                                   ├── get_subjects()
                                   ├── get_fog_label_column()
                                   ├── get_summary_by_subject()
                                   ├── save_dataset()
                                   ├── plot_fog_distribution()
                                   └── print_summary()
```

**Resultado:** Todos los loaders pueden usarse de manera intercambiable con la misma API.

## 📁 Estructura del Proyecto

```
.
├── loaders/                    # Paquete de loaders estandarizados
│   ├── __init__.py            # Exporta todos los loaders
│   ├── BaseDatasetLoader.py   # Clases base abstractas (ABC)
│   ├── INTERFACE.md           # Documentación de la interfaz
│   ├── DaphnetReader.py       # Loader Daphnet
│   ├── FigshareReader.py      # Loader Figshare
│   ├── ChariteReader.py       # Loader Charité
│   ├── MendelayReader.py      # Loader Mendeley
│   └── KaggleReader.py        # Loader Kaggle
├── configs/                    # Configuraciones
│   └── daphnetConfig.py       # Config Daphnet
├── examples/                   # Ejemplos de uso y notebooks
│   ├── usage_examples.py      # Ejemplos completos
│   ├── daphnet_dataset/       # Análisis Daphnet
│   ├── figshare_dataset/      # Análisis Figshare
│   └── ...
├── outputs/                    # Datasets procesados (CSV, Parquet)
├── Datasets/                   # Datasets originales
├── load_dataset.py            # Script CLI principal
├── test_interface.py          # Tests de interfaz estándar
├── unified_loader_example.py  # Demo de polimorfismo
├── requirements.txt           # Dependencias
└── README.md                  # Esta documentación
```

## 🚀 Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## 📖 Uso Rápido

### Opción 1: CLI - Línea de Comandos (Más Simple)

```bash
# Cargar y ver resumen
python load_dataset.py daphnet

# Cargar y guardar
python load_dataset.py kaggle --save

# Con opciones específicas
python load_dataset.py kaggle --subset train --type defog --save --format csv parquet

# Ayuda
python load_dataset.py --help
```

### Opción 2: Python Script con Factory Pattern

```python
from loaders import load_dataset

# Cargar cualquier dataset
loader = load_dataset('daphnet', r'Datasets\Daphnet fog\dataset')
df = loader.load_all_data(verbose=True)

# Guardar
loader.save_dataset('daphnet_data', formats=['csv', 'parquet'])
```

### Opción 3: Importación Directa de Loaders

```python
from loaders import DaphnetDatasetLoader, KaggleDatasetLoader

# Daphnet
loader = DaphnetDatasetLoader(r'Datasets\Daphnet fog\dataset')
df = loader.load_all_data()

# Kaggle con opciones
kaggle = KaggleDatasetLoader(r'Datasets\Kaggle Michael J Fox Foundation Dataset')
df_train = kaggle.load_all_data(subset='train', dataset_type='defog')
```

## 📚 Datasets Soportados

### 1. Daphnet Freezing of Gait
- **Sujetos**: 10 pacientes (S01-S10)
- **Sensores**: 3 acelerómetros 3D @ 64 Hz (tobillo, muslo, tronco)
- **Anotaciones**: 0=No experimento, 1=Caminar, 2=Freeze
- **Formato**: TXT

```python
from loaders import DaphnetDatasetLoader

loader = DaphnetDatasetLoader('Datasets/Daphnet fog/dataset')
df = loader.load_all_data()
loader.save_dataset('daphnet_complete')
```

### 2. Figshare Public Dataset
- **Sujetos**: 35 pacientes
- **Sensores**: IMU (acelerómetro + giroscopio)
- **Tipos**: Walking y Standing trials
- **Metadata**: Información demográfica y clínica
- **Formato**: TXT + CSV metadata

```python
from loaders import FigshareDatasetLoader

loader = FigshareDatasetLoader('Datasets/Figshare a public dataset')

# Cargar todo
df = loader.load_all_data()

# Solo trials de caminata
df_walking = loader.load_all_data(trial_type='walking')

# Metadata
metadata = loader.load_metadata()
```

### 3. Charité-Universitätsmedizin Berlin
- **Sujetos**: 16 pacientes
- **Sensores**: 3D accel + 3D gyro @ 200 Hz
- **Ubicación**: Ambos pies
- **Trials**: 2 trials por sujeto
- **Formato**: CSV

```python
from loaders import ChariteDatasetLoader

loader = ChariteDatasetLoader('Datasets/Charité–Universitätsmedizin Berlin/Data Sheet 2/data')

# Todo el dataset
df = loader.load_all_data()

# Solo pie izquierdo
df_left = loader.load_all_data(foot='left')

# Resúmenes específicos
print(loader.get_summary_by_foot())
print(loader.get_summary_by_trial())
```

### 4. Mendeley Data - Multimodal Dataset
- **Sujetos**: 11 pacientes
- **Sensores**: Múltiples IMU (61 columnas)
- **Tareas**: 4-6 tareas por sujeto
- **Muestras**: 4+ millones de registros
- **Formato**: TXT (CSV)

```python
from loaders import MendelayDatasetLoader

loader = MendelayDatasetLoader('Datasets/FOG - Mendeley Data.../Filtered')
df = loader.load_all_data()

# Un sujeto específico
df_subject = loader.load_subject_data(subject_id=1)

# Resúmenes
print(loader.get_summary_by_subject())
print(loader.get_summary_by_task())
```

### 5. Kaggle Michael J Fox Foundation
- **Sujetos**: Múltiples pacientes
- **Sensores**: 3D accelerometer (AccV, AccML, AccAP)
- **Archivos**: 970 en train (91 defog, 833 tdcsfog, 46 notype)
- **Etiquetas**: StartHesitation, Turn, Walking
- **Formato**: CSV

```python
from loaders import KaggleDatasetLoader

loader = KaggleDatasetLoader('Datasets/Kaggle Michael J Fox Foundation Dataset')

# Todo el train
df = loader.load_all_data(subset='train')

# Solo defog
df_defog = loader.load_all_data(subset='train', dataset_type='defog')

# Resúmenes
print(loader.get_summary_by_dataset_type())
print(loader.get_summary_by_fog_type())
```

## 🔧 Interfaz Estándar

Todos los loaders implementan la misma interfaz, permitiendo código reutilizable y polimorfismo:

### Métodos Comunes

```python
# 1. Cargar todos los datos
df = loader.load_all_data(verbose=True, **kwargs)

# 2. Cargar datos de un sujeto específico
df_subject = loader.load_subject_data(subject_id=1, **kwargs)

# 3. Obtener información básica
info = loader.get_basic_info()  # {'shape', 'n_rows', 'memory_mb', ...}

# 4. Obtener lista de sujetos
subjects = loader.get_subjects()  # [1, 2, 3, ...]

# 5. Identificar columna de FoG
fog_col = loader.get_fog_label_column()  # 'fog_label', 'annotation', etc.

# 6. Resumen estadístico por sujeto
summary = loader.get_summary_by_subject()

# 7. Guardar dataset en múltiples formatos
loader.save_dataset('output_name', formats=['csv', 'parquet', 'pickle'])

# 8. Leer dataset previamente guardado
df = loader.read_dataset('output_name.csv')

# 9. Imprimir resumen completo
loader.print_summary()

# 10. Visualizar distribución de FoG
loader.plot_fog_distribution()
```

### Polimorfismo en Acción

Gracias a la interfaz común, puedes escribir código genérico que funcione con cualquier dataset:

```python
def analyze_any_dataset(loader):
    """Esta función funciona con CUALQUIER loader."""
    # Cargar datos
    df = loader.load_all_data(verbose=False)
    
    # Obtener información
    info = loader.get_basic_info()
    print(f"Filas: {info['n_rows']:,}")
    print(f"Memoria: {info['memory_mb']:.2f} MB")
    
    # Obtener sujetos
    subjects = loader.get_subjects()
    print(f"Sujetos: {len(subjects)}")
    
    # Resumen por sujeto
    summary = loader.get_summary_by_subject()
    print(summary)
    
    return df

# Usar con cualquier dataset
from loaders import load_dataset

daphnet_loader = load_dataset('daphnet', 'Datasets/Daphnet fog/dataset')
figshare_loader = load_dataset('figshare', 'Datasets/Figshare a public dataset')

# ¡Mismo código, diferentes datasets!
daphnet_data = analyze_any_dataset(daphnet_loader)
figshare_data = analyze_any_dataset(figshare_loader)
```

### Comparación de Datasets

Ejemplo de uso de la interfaz estándar para comparar todos los datasets:

```python
# Ejecutar el script de ejemplo unificado
python unified_loader_example.py

# O especificar un dataset
python unified_loader_example.py daphnet
```

### Validar la Interfaz

Verifica que todos los loaders implementan correctamente la interfaz:

```python
# Ejecutar tests de interfaz
python test_interface.py
```

Ver documentación completa en [`loaders/INTERFACE.md`](loaders/INTERFACE.md)
```

## 🆕 Agregar Nuevo Dataset

1. Crea `loaders/NewDatasetReader.py`:

```python
from .BaseDatasetLoader import BaseFileReader, BaseDatasetLoader

class NewFileReader(BaseFileReader):
    def read_file(self, file_path):
        # Tu lógica de lectura
        pass
    
    @staticmethod
    def _parse_filename(file_path):
        # Extraer metadata del nombre
        pass

class NewDatasetLoader(BaseDatasetLoader):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.file_reader = NewFileReader()
    
    def get_file_list(self, **kwargs):
        # Listar archivos
        pass
    
    def load_all_data(self, verbose=True, **kwargs):
        # Cargar todos los datos
        pass
```

2. Agregar a `loaders/__init__.py`:

```python
from .NewDatasetReader import NewDatasetLoader
__all__.append('NewDatasetLoader')
```

3. Actualizar factory en `BaseDatasetLoader.py`:

```python
loaders = {
    ...
    'newdataset': NewDatasetLoader,
}
```

## 📊 Formatos de Salida

- **CSV**: Compatible, legible, grande
- **Parquet**: Eficiente, comprimido, rápido
- **Pickle**: Nativo de pandas, preserva tipos

```python
# Guardar en múltiples formatos
loader.save_dataset('dataset_name', formats=['csv', 'parquet', 'pickle'])

# Solo CSV (más rápido)
loader.save_dataset('dataset_name', formats=['csv'])
```

## 🎓 Ejemplos Completos

Ver `examples/usage_examples.py` para ejemplos detallados de:
- Uso de factory pattern
- Instanciación directa
- Filtrado de datos
- Análisis y resúmenes
- Visualizaciones

## 📦 Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🏗️ Arquitectura

```
BaseDatasetLoader (Abstract)
    ├── BaseFileReader (Abstract)
    ├── load_all_data() [Abstract]
    ├── get_file_list() [Abstract]
    ├── save_dataset() [Implementado]
    ├── get_basic_info() [Implementado]
    └── print_summary() [Implementado]
        ↓ inherits
    DaphnetDatasetLoader
    FigshareDatasetLoader
    ChariteDatasetLoader
    MendelayDatasetLoader
    KaggleDatasetLoader
```

## 📝 Principios de Diseño

- **SOLID**: Responsabilidad única, abierto/cerrado, sustitución de Liskov
- **DRY**: Funcionalidad común en clase base (~40% reducción de código)
- **Factory Pattern**: Creación dinámica de loaders
- **Template Method**: Algoritmo en base, detalles en hijos

## 🤝 Contribuir

Para agregar un nuevo dataset:
1. Crea el loader siguiendo el patrón establecido
2. Agrega tests si es posible
3. Actualiza documentación
4. Envía pull request

## 📄 Licencia

MIT License

## ✨ Autor

Proyecto desarrollado para análisis de Freezing of Gait en pacientes con Parkinson.

---

**Versión**: 1.0.0  
**Última actualización**: Febrero 2026
