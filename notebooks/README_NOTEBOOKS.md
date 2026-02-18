# Examples - Jupyter Notebooks por Dataset

Esta carpeta contiene análisis individuales para cada dataset de Freezing of Gait (FoG), organizados en subcarpetas.

## Estructura

```
examples/
├── daphnet_dataset/
│   └── daphnet_analysis.ipynb      # Análisis del dataset Daphnet FoG
├── figshare_dataset/
│   └── figshare_analysis.ipynb     # Análisis del dataset Figshare Public
├── charite_dataset/
│   └── charite_analysis.ipynb      # Análisis del dataset Charité Berlin
├── mendeley_dataset/
│   └── mendeley_analysis.ipynb     # Análisis del dataset Mendeley Multimodal
├── kaggle_dataset/
│   └── kaggle_analysis.ipynb       # Análisis del dataset Kaggle MJFF
├── usage_examples.py                # Ejemplos de uso en Python
└── daphnet_data_loader.ipynb       # Notebook original de Daphnet
```

## Notebooks Disponibles

### 1. Daphnet FoG Dataset
**Archivo:** `daphnet_dataset/daphnet_analysis.ipynb`

**Contenido:**
- Carga del dataset Daphnet (10 sujetos, 3 acelerómetros)
- Análisis de distribución de clases (No experimento, Caminar, Freeze)
- Visualización de señales de aceleración por sensor (ankle, thigh, trunk)
- Estadísticas por sujeto y trial
- Ejemplos de visualización de señales temporales

**Características del dataset:**
- 10 pacientes con Parkinson's (S01-S10)
- 3 acelerómetros 3D @ 64 Hz
- Anotaciones: 0=No experimento, 1=Caminar sin freeze, 2=Freeze

---

### 2. Figshare Public Dataset
**Archivo:** `figshare_dataset/figshare_analysis.ipynb`

**Contenido:**
- Carga del dataset Figshare (35 sujetos, IMU)
- Análisis de trials walking vs standing
- Visualización de acelerómetro + giroscopio
- Distribución de muestras por sujeto
- Matriz de correlación entre sensores

**Características del dataset:**
- 35 pacientes
- IMU: Acelerómetro (x, y, z) + Giroscopio (x, y, z)
- Tipos de trial: Walking y Standing

---

### 3. Charité-Universitätsmedizin Berlin Dataset
**Archivo:** `charite_dataset/charite_analysis.ipynb`

**Contenido:**
- Carga del dataset Charité (16 sujetos, ambos pies)
- Análisis por pie (izquierdo vs derecho)
- Visualización de señales IMU @ 200 Hz
- Comparación entre pies
- Análisis de correlación

**Características del dataset:**
- 16 pacientes
- IMU en ambos pies @ 200 Hz
- Acelerómetro (3 ejes) + Giroscopio (3 ejes) por pie

---

### 4. Mendeley Multimodal Dataset
**Archivo:** `mendeley_dataset/mendeley_analysis.ipynb`

**Contenido:**
- Carga del dataset completo (4.1M+ muestras)
- Análisis de distribución de FoG (35.55% FoG)
- Resumen por sujeto y por tarea
- Visualización de señales multimodales
- Análisis de correlación con FoG

**Características del dataset:**
- 11 sujetos con Parkinson's
- 4,179,822 muestras
- 61 columnas (sensores multimodales)
- 35.55% de episodios FoG

**⚠️ Nota:** Este dataset es muy grande y puede tardar varios minutos en cargar.

---

### 5. Kaggle Michael J Fox Foundation Dataset
**Archivo:** `kaggle_dataset/kaggle_analysis.ipynb`

**Contenido:**
- Carga de subsets (train/test/unlabeled) y tipos (defog/tdcsfog/notype)
- Análisis de 3 tipos de eventos FoG: StartHesitation, Turn, Walking
- Visualización de componentes de aceleración (AccV, AccML, AccAP)
- Comparación FoG vs No FoG
- Estadísticas por archivo

**Características del dataset:**
- 970 archivos de entrenamiento
- 3 tipos: defog (91), tdcsfog (833), notype (46)
- 3 etiquetas FoG: StartHesitation, Turn, Walking
- Aceleración en 3 ejes: Vertical, Medio-Lateral, Antero-Posterior

**⚠️ Nota:** Carga solo defog por defecto (más pequeño). Cambiar `type='tdcsfog'` para dataset completo.

---

## Cómo Usar los Notebooks

### 1. Asegúrate de tener Jupyter instalado:
```bash
pip install jupyter notebook
# o
pip install jupyterlab
```

### 2. Inicia Jupyter:
```bash
# Opción 1: Jupyter Notebook
jupyter notebook

# Opción 2: JupyterLab
jupyter lab
```

### 3. Navega a la carpeta del dataset deseado y abre el notebook

### 4. Ejecuta las celdas secuencialmente
Cada notebook está diseñado para ejecutarse celda por celda, con explicaciones en cada paso.

---

## Requisitos

Todos los notebooks requieren las dependencias especificadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Paquetes principales:**
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- pathlib

---

## Personalización

Cada notebook puede personalizarse modificando los parámetros de carga:

### Daphnet
```python
loader = DaphnetDatasetLoader()
df = loader.load_data()
```

### Figshare
```python
# Opciones: trial_type='walking', 'standing', o None (ambos)
loader = FigshareDatasetLoader(trial_type='walking')
```

### Charité
```python
# Opciones: foot='left', 'right', o None (ambos)
loader = ChariteDatasetLoader(foot='left')
```

### Mendeley
```python
loader = MendelayDatasetLoader()
df = loader.load_data()  # Puede tardar varios minutos
```

### Kaggle
```python
# subset: 'train', 'test', 'unlabeled'
# type: 'defog', 'tdcsfog', 'notype', None (todos)
loader = KaggleDatasetLoader(subset='train', type='defog')
```

---

## Estructura de cada Notebook

Todos los notebooks siguen una estructura similar:

1. **Imports y configuración**: Importar librerías y configurar visualizaciones
2. **Carga de datos**: Cargar el dataset usando el loader correspondiente
3. **Información básica**: Resumen y estadísticas del dataset
4. **Visualización de primeras filas**: Ver estructura de datos
5. **Análisis de distribución**: Distribución de clases, sujetos, etc.
6. **Visualizaciones de señales**: Gráficas de señales temporales
7. **Análisis estadístico**: Correlaciones, estadísticas descriptivas
8. **Análisis específico**: Análisis únicos para cada dataset

---

## Guardando Resultados

Para guardar datasets procesados:

```python
# Método 1: Usar el loader
loader.save_dataset(df, 'nombre_archivo', formats=['csv', 'parquet'])

# Método 2: Manual
df.to_csv('outputs/mi_dataset.csv', index=False)
df.to_parquet('outputs/mi_dataset.parquet', index=False)
```

---

## Troubleshooting

### Error: "Module not found"
Asegúrate de que el path al proyecto esté configurado correctamente:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
```

### Error: "Dataset not found"
Verifica que los datos originales estén en la carpeta `Datasets/` con la estructura correcta.

### Memoria insuficiente (Mendeley o Kaggle)
- Mendeley: Reduce la cantidad de archivos cargados modificando el loader
- Kaggle: Carga solo un subset específico (`type='defog'` es más pequeño)

---

## Recursos Adicionales

- **CLI Tool**: `load_dataset.py` en la raíz del proyecto
- **Python Examples**: `usage_examples.py` en esta carpeta
- **Documentation**: Ver `README.md` en la raíz del proyecto
- **API Reference**: Revisa los docstrings en cada loader (`loaders/`)

---

## Contribuir

Si deseas agregar análisis adicionales:
1. Crea una nueva celda en el notebook correspondiente
2. Documenta tu análisis con markdown
3. Asegúrate de que el código sea reproducible
4. Comparte tus hallazgos!
