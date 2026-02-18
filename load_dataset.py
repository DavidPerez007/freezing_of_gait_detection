"""
Script principal para cargar y procesar cualquier dataset de FoG

Uso:
    python load_dataset.py <dataset_name> [options]
    
Ejemplos:
    python load_dataset.py daphnet
    python load_dataset.py kaggle --subset train --type defog
    python load_dataset.py mendeley --save
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio raíz al path para importar loaders
sys.path.insert(0, str(Path(__file__).parent))

from loaders import load_dataset

# Configuración de rutas de datasets
DATASET_PATHS = {
    'daphnet': r'Datasets\Daphnet fog\dataset',
    'figshare': r'Datasets\Figshare a public dataset',
    'charite': r'Datasets\Charité–Universitätsmedizin Berlin\Data Sheet 2\data',
    'mendeley': r'Datasets\FOG - Mendeley Data Raw Data Multimodal Dataset of Freezing of Gait in Parkinson\'s Disease\Filtered',
    'kaggle': r'Datasets\Kaggle Michael J Fox Foundation Dataset'
}


def main():
    parser = argparse.ArgumentParser(
        description='Cargar y procesar datasets de Freezing of Gait',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'dataset',
        choices=DATASET_PATHS.keys(),
        help='Nombre del dataset a cargar'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Guardar el dataset cargado en CSV'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Nombre del archivo de salida (sin extensión)'
    )
    
    parser.add_argument(
        '--format',
        nargs='+',
        choices=['csv', 'parquet', 'pickle'],
        default=['csv'],
        help='Formato(s) de salida (default: csv)'
    )
    
    # Opciones específicas para Kaggle
    parser.add_argument(
        '--subset',
        type=str,
        choices=['train', 'test', 'unlabeled'],
        default='train',
        help='[Kaggle] Subset a cargar (default: train)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['defog', 'tdcsfog', 'notype'],
        default=None,
        help='[Kaggle] Tipo de dataset (default: todos)'
    )
    
    # Opciones específicas para Figshare
    parser.add_argument(
        '--trial',
        type=str,
        choices=['walking', 'standing'],
        default=None,
        help='[Figshare] Tipo de trial (default: todos)'
    )
    
    # Opciones específicas para Charité
    parser.add_argument(
        '--foot',
        type=str,
        choices=['left', 'right'],
        default=None,
        help='[Charité] Pie a cargar (default: ambos)'
    )
    
    args = parser.parse_args()
    
    # Obtener ruta del dataset
    dataset_path = DATASET_PATHS[args.dataset]
    
    print("\n" + "="*70)
    print(f" CARGANDO DATASET: {args.dataset.upper()}")
    print("="*70 + "\n")
    
    # Crear loader
    loader = load_dataset(args.dataset, dataset_path)
    
    # Preparar argumentos específicos del dataset
    load_kwargs = {}
    
    if args.dataset == 'kaggle':
        load_kwargs['subset'] = args.subset
        if args.type:
            load_kwargs['dataset_type'] = args.type
    elif args.dataset == 'figshare' and args.trial:
        load_kwargs['trial_type'] = args.trial
    elif args.dataset == 'charite' and args.foot:
        load_kwargs['foot'] = args.foot
    
    # Cargar datos
    df = loader.load_all_data(verbose=True, **load_kwargs)
    
    # Mostrar resumen
    print("\n" + "="*70)
    print(" RESUMEN DEL DATASET")
    print("="*70)
    print(f"\nDimensiones: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Guardar si se solicita
    if args.save:
        output_name = args.output or f"{args.dataset}_dataset"
        
        print("\n" + "="*70)
        print(" GUARDANDO DATASET")
        print("="*70 + "\n")
        
        saved_files = loader.save_dataset(output_name, formats=args.format)
        
        print("\n" + "="*70)
        print(" ✅ PROCESO COMPLETADO")
        print("="*70)
        print(f"\nArchivos guardados: {len(saved_files)}")
        for file in saved_files:
            print(f"  - {Path(file).name}")
    else:
        print("\n💡 Usa --save para guardar el dataset")
    
    print()


if __name__ == "__main__":
    main()
