"""
Ejemplos de uso de los loaders de FoG datasets

Este notebook muestra cómo usar cada uno de los loaders disponibles.
"""

# Agregar el directorio raíz al path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from loaders import (
    load_dataset,
    DaphnetDatasetLoader,
    FigshareDatasetLoader,
    ChariteDatasetLoader,
    MendelayDatasetLoader,
    KaggleDatasetLoader
)

# ==================================================================================
# MÉTODO 1: Usar la función factory load_dataset() - MÁS SIMPLE
# ==================================================================================

print("="*70)
print("MÉTODO 1: Usando load_dataset() - Factory Pattern")
print("="*70 + "\n")

# Ejemplo 1: Daphnet
loader_daphnet = load_dataset('daphnet', r'Datasets\Daphnet fog\dataset')
df_daphnet = loader_daphnet.load_all_data(verbose=True)

# Ejemplo 2: Kaggle (con parámetros específicos)
loader_kaggle = load_dataset('kaggle', r'Datasets\Kaggle Michael J Fox Foundation Dataset')
df_kaggle = loader_kaggle.load_all_data(verbose=True, subset='train', dataset_type='defog')

# ==================================================================================
# MÉTODO 2: Instanciar loaders directamente - MÁS CONTROL
# ==================================================================================

print("\n" + "="*70)
print("MÉTODO 2: Instanciando loaders directamente")
print("="*70 + "\n")

# Ejemplo 3: Figshare
figshare_path = r'Datasets\Figshare a public dataset'
loader_figshare = FigshareDatasetLoader(figshare_path)

# Cargar solo trials de caminata
df_walking = loader_figshare.load_all_data(trial_type='walking')

# Cargar metadata
metadata = loader_figshare.load_metadata()
print(f"Metadata shape: {metadata.shape}")

# Ejemplo 4: Charité
charite_path = r'Datasets\Charité–Universitätsmedizin Berlin\Data Sheet 2\data'
loader_charite = ChariteDatasetLoader(charite_path)

# Cargar solo pie izquierdo
df_left = loader_charite.load_all_data(foot='left')

# Obtener resúmenes específicos
print("\nResumen por pie:")
print(loader_charite.get_summary_by_foot())

# ==================================================================================
# GUARDAR DATASETS
# ==================================================================================

print("\n" + "="*70)
print("GUARDANDO DATASETS")
print("="*70 + "\n")

# Guardar en múltiples formatos
loader_daphnet.save_dataset(
    'daphnet_complete',
    formats=['csv', 'parquet', 'pickle']
)

# Guardar solo en CSV (más rápido)
loader_kaggle.save_dataset(
    'kaggle_train_defog',
    formats=['csv']
)

# ==================================================================================
# ANÁLISIS Y RESÚMENES
# ==================================================================================

print("\n" + "="*70)
print("RESÚMENES Y ANÁLISIS")
print("="*70 + "\n")

# Información básica
info = loader_daphnet.get_basic_info()
print(f"Dataset shape: {info['shape']}")
print(f"Memory usage: {info['memory_mb']:.2f} MB")

# Resumen por sujeto
summary = loader_daphnet.get_summary_by_subject()
print("\nResumen por sujeto (Daphnet):")
print(summary)

# Visualización de distribución de FoG
loader_daphnet.plot_fog_distribution()

print("\n✅ Ejemplos completados!")
