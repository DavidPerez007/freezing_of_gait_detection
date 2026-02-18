"""
Ejemplo Unificado de Uso de Loaders con Interfaz Estándar

Este script demuestra cómo usar todos los loaders de manera consistente
gracias a la interfaz estandarizada.
"""

import pandas as pd
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from loaders import load_dataset


def process_dataset_with_standard_interface(dataset_type: str, dataset_path: str):
    """
    Procesa cualquier dataset usando la interfaz estándar.
    Esta función funciona igual para todos los datasets.
    
    Args:
        dataset_type: Tipo de dataset ('daphnet', 'figshare', etc.)
        dataset_path: Ruta al dataset
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO: {dataset_type.upper()}")
    print(f"{'='*70}\n")
    
    # 1. Crear loader usando factory function
    loader = load_dataset(dataset_type, dataset_path)
    
    # 2. Cargar todos los datos
    print("📊 Cargando datos...")
    df = loader.load_all_data(verbose=True)
    
    # 3. Obtener información básica (interfaz estándar)
    print("\n📋 Información Básica:")
    info = loader.get_basic_info()
    print(f"   - Dimensiones: {info['shape']}")
    print(f"   - Filas: {info['n_rows']:,}")
    print(f"   - Columnas: {info['n_cols']}")
    print(f"   - Memoria: {info['memory_mb']:.2f} MB")
    
    if 'n_subjects' in info:
        print(f"   - Sujetos: {info['n_subjects']}")
    if 'n_files' in info:
        print(f"   - Archivos: {info['n_files']}")
    
    # 4. Obtener lista de sujetos (interfaz estándar)
    try:
        subjects = loader.get_subjects()
        print(f"\n👥 Sujetos disponibles: {subjects}")
    except ValueError as e:
        print(f"\n👥 Sujetos: {str(e)}")
    
    # 5. Identificar columna de FoG (interfaz estándar)
    try:
        fog_col = loader.get_fog_label_column()
        print(f"\n🏷️  Columna de FoG: '{fog_col}'")
    except ValueError as e:
        print(f"\n🏷️  Columna de FoG: {str(e)}")
    
    # 6. Obtener resumen por sujeto (interfaz estándar)
    try:
        summary = loader.get_summary_by_subject()
        print(f"\n📈 Resumen por Sujeto (primeros 5):")
        print(summary.head())
    except (ValueError, Exception) as e:
        print(f"\n📈 Resumen por Sujeto: {str(e)}")
    
    # 7. Cargar datos de un sujeto específico (interfaz estándar)
    try:
        subjects = loader.get_subjects()
        if subjects:
            first_subject = subjects[0]
            print(f"\n🔍 Cargando datos del sujeto {first_subject}...")
            subject_data = loader.load_subject_data(first_subject)
            print(f"   - Muestras del sujeto: {len(subject_data):,}")
    except (ValueError, Exception) as e:
        print(f"\n🔍 Carga de sujeto individual: {str(e)}")
    
    # 8. Guardar dataset (interfaz estándar)
    output_path = f"outputs/{dataset_type}_unified"
    print(f"\n💾 Guardando dataset en: {output_path}")
    saved_files = loader.save_dataset(output_path, formats=['csv', 'parquet'])
    
    print(f"\n✅ Dataset {dataset_type.upper()} procesado exitosamente")
    return loader


def compare_all_datasets():
    """
    Compara todos los datasets usando la interfaz estándar.
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║     COMPARACIÓN DE DATASETS USANDO INTERFAZ ESTÁNDAR               ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Definir todos los datasets
    datasets = {
        'daphnet': r'Datasets\Daphnet fog\dataset',
        'figshare': r'Datasets\Figshare a public dataset',
        'charite': r'Datasets\Charité–Universitätsmedizin Berlin',
        'mendeley': r'Datasets\FOG - Mendeley Data Raw Data\Filtered',
        'kaggle': r'Datasets\Kaggle Michael J Fox Foundation Dataset'
    }
    
    # Tabla comparativa
    comparison = []
    
    for dataset_type, dataset_path in datasets.items():
        # Verificar que la ruta existe
        if not Path(dataset_path).exists():
            print(f"⚠️  SALTADO: {dataset_type} - ruta no encontrada")
            continue
        
        try:
            # Usar interfaz estándar para obtener información
            loader = load_dataset(dataset_type, dataset_path)
            df = loader.load_all_data(verbose=False)
            
            info = loader.get_basic_info()
            fog_col = loader.get_fog_label_column()
            
            # Calcular porcentaje de FoG
            if fog_col == 'annotation':  # Daphnet tiene clases múltiples
                fog_samples = (df[fog_col] == 2).sum()
            else:
                fog_samples = df[fog_col].sum()
            
            fog_pct = (fog_samples / len(df)) * 100
            
            comparison.append({
                'Dataset': dataset_type.title(),
                'Sujetos': info.get('n_subjects', 'N/A'),
                'Archivos': info.get('n_files', 'N/A'),
                'Muestras': info['n_rows'],
                '% FoG': f"{fog_pct:.2f}%",
                'Memoria (MB)': f"{info['memory_mb']:.2f}"
            })
            
        except Exception as e:
            print(f"❌ Error procesando {dataset_type}: {str(e)}")
            continue
    
    # Mostrar tabla comparativa
    if comparison:
        comparison_df = pd.DataFrame(comparison)
        print("\n" + "="*80)
        print("TABLA COMPARATIVA DE DATASETS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80 + "\n")
    
    return comparison


def demo_polymorphism():
    """
    Demuestra el polimorfismo: todos los loaders pueden tratarse igual.
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║     DEMOSTRACIÓN DE POLIMORFISMO                                   ║
║     Todos los loaders responden a los mismos métodos               ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    def analyze_dataset(loader, dataset_name):
        """
        Función genérica que funciona con cualquier loader.
        """
        print(f"\n--- Analizando {dataset_name} ---")
        
        # Estos métodos funcionan para TODOS los loaders
        info = loader.get_basic_info()
        print(f"Filas: {info['n_rows']:,}")
        
        try:
            subjects = loader.get_subjects()
            print(f"Sujetos: {len(subjects)}")
        except:
            print("Sujetos: N/A")
        
        try:
            fog_col = loader.get_fog_label_column()
            print(f"Columna FoG: {fog_col}")
        except:
            print("Columna FoG: N/A")
    
    # Esta es la misma función, usada con diferentes loaders
    datasets = [
        ('daphnet', r'Datasets\Daphnet fog\dataset'),
        ('figshare', r'Datasets\Figshare a public dataset'),
    ]
    
    for dataset_type, path in datasets:
        if Path(path).exists():
            loader = load_dataset(dataset_type, path)
            loader.load_all_data(verbose=False)
            analyze_dataset(loader, dataset_type.title())


def main():
    """
    Ejecuta ejemplos de uso de la interfaz estándar.
    """
    import sys
    
    # Si se proporciona un argumento, procesar solo ese dataset
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        
        # Mapeo de tipos a rutas
        dataset_paths = {
            'daphnet': r'Datasets\Daphnet fog\dataset',
            'figshare': r'Datasets\Figshare a public dataset',
            'charite': r'Datasets\Charité–Universitätsmedizin Berlin',
            'mendeley': r'Datasets\FOG - Mendeley Data Raw Data\Filtered',
            'kaggle': r'Datasets\Kaggle Michael J Fox Foundation Dataset'
        }
        
        if dataset_type in dataset_paths:
            process_dataset_with_standard_interface(dataset_type, dataset_paths[dataset_type])
        else:
            print(f"❌ Dataset desconocido: {dataset_type}")
            print(f"Opciones: {', '.join(dataset_paths.keys())}")
    else:
        # Sin argumentos: ejecutar demos
        print("Ejecutando demostraciones de la interfaz estándar...\n")
        
        # 1. Demostración de polimorfismo
        demo_polymorphism()
        
        # 2. Comparación de todos los datasets
        # compare_all_datasets()  # Descomenta para comparar todos
        
        print("\n" + "="*70)
        print("💡 TIP: Ejecuta este script con un argumento para procesar un dataset:")
        print("   python unified_loader_example.py daphnet")
        print("   python unified_loader_example.py figshare")
        print("="*70)


if __name__ == "__main__":
    main()

