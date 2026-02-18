"""
Diagrama Visual de la Arquitectura Estandarizada

Este script genera una visualización de la arquitectura de loaders.
"""

def print_architecture_diagram():
    """Imprime un diagrama ASCII de la arquitectura."""
    
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ARQUITECTURA DE LOADERS ESTANDARIZADA                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASES BASE ABSTRACTAS (ABC)                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐         ┌────────────────────────────────┐
    │   BaseFileReader (ABC)   │         │  BaseDatasetLoader (ABC)       │
    │                          │         │                                │
    │  @abstractmethod         │         │  @abstractmethod               │
    │  • read_file()           │         │  • get_file_list()             │
    │  • _parse_filename()     │         │  • load_all_data()             │
    └────────────┬─────────────┘         │                                │
                 │                       │  Métodos concretos:            │
                 │                       │  • load_subject_data()         │
                 │                       │  • get_basic_info()            │
                 │                       │  • get_subjects()              │
                 │                       │  • get_fog_label_column()      │
                 │                       │  • get_summary_by_subject()    │
                 │                       │  • save_dataset()              │
                 │                       │  • plot_fog_distribution()     │
                 │                       │  • print_summary()             │
                 │                       └────────────┬───────────────────┘
                 │                                    │
                 │                                    │
┌────────────────┴────────────────────────────────────┴────────────────────────┐
│                          IMPLEMENTACIONES CONCRETAS                          │
└──────────────────────────────────────────────────────────────────────────────┘

    FileReader                           DatasetLoader
    ─────────────                        ──────────────────

    DaphnetFileReader    ───────────►    DaphnetDatasetLoader
    • read_file()                        • get_file_list()
    • _parse_filename()                  • load_all_data()
                                         • load_subject_data(subject_id, **kwargs)
                                         + get_summary_by_subject() [override]

    FigshareFileReader   ───────────►    FigshareDatasetLoader
    • read_file()                        • get_file_list(trial_type)
    • _parse_filename()                  • load_all_data(trial_type)
                                         • load_subject_data(subject_id, trial_type, **kwargs)
                                         + load_metadata() [específico]
                                         + get_summary_by_trial_type() [específico]

    ChariteFileReader    ───────────►    ChariteDatasetLoader
    • read_file()                        • get_file_list(foot)
    • _parse_filename()                  • load_all_data(foot)
                                         • load_subject_data(subject_id, foot, **kwargs)
                                         + get_summary_by_foot() [específico]

    MendelayFileReader   ───────────►    MendelayDatasetLoader
    • read_file()                        • get_file_list()
    • _parse_filename()                  • load_all_data()
                                         • load_subject_data(subject_id, **kwargs)
                                         + get_summary_by_task() [específico]

    KaggleFileReader     ───────────►    KaggleDatasetLoader
    • read_file()                        • get_file_list(subset, dataset_type)
    • _parse_filename()                  • load_all_data(subset, dataset_type)
                                         • load_subject_data(subject_id, **kwargs) [alias]
                                         + load_file_data(file_id) [específico]
                                         + get_summary_by_dataset_type() [específico]
                                         + get_summary_by_fog_type() [específico]

┌──────────────────────────────────────────────────────────────────────────────┐
│                            FACTORY FUNCTION                                  │
└──────────────────────────────────────────────────────────────────────────────┘

    load_dataset(dataset_type: str, dataset_path: str, **kwargs)
         │
         ├─► 'daphnet'   → DaphnetDatasetLoader(dataset_path)
         ├─► 'figshare'  → FigshareDatasetLoader(dataset_path)
         ├─► 'charite'   → ChariteDatasetLoader(dataset_path)
         ├─► 'mendeley'  → MendelayDatasetLoader(dataset_path)
         └─► 'kaggle'    → KaggleDatasetLoader(dataset_path)

┌──────────────────────────────────────────────────────────────────────────────┐
│                         VENTAJAS DE LA ESTANDARIZACIÓN                       │
└──────────────────────────────────────────────────────────────────────────────┘

    ✅ POLIMORFISMO
       Todos los loaders pueden usarse intercambiablemente:
       
       def process_dataset(loader):
           df = loader.load_all_data()
           loader.print_summary()
       
       # Funciona con CUALQUIER loader

    ✅ CÓDIGO REUTILIZABLE
       Métodos comunes heredados automáticamente

    ✅ INTERFAZ CONSISTENTE
       Misma firma de métodos en todos los loaders

    ✅ FÁCIL EXTENSIÓN
       Para agregar nuevo dataset:
       1. Heredar BaseFileReader → implementar 2 métodos
       2. Heredar BaseDatasetLoader → implementar 2 métodos
       3. ¡Listo! Resto heredado automáticamente

    ✅ MANTENIMIENTO SIMPLE
       Cambios en clase base se propagan a todos

┌──────────────────────────────────────────────────────────────────────────────┐
│                          EJEMPLO DE USO UNIFICADO                            │
└──────────────────────────────────────────────────────────────────────────────┘

    from loaders import load_dataset
    
    # Cargar cualquier dataset con la misma interfaz
    datasets = ['daphnet', 'figshare', 'charite', 'mendeley', 'kaggle']
    
    for dataset_type in datasets:
        loader = load_dataset(dataset_type, f'path/to/{dataset_type}')
        
        # Mismos métodos para todos
        df = loader.load_all_data()
        info = loader.get_basic_info()
        subjects = loader.get_subjects()
        summary = loader.get_summary_by_subject()
        loader.save_dataset(f'output/{dataset_type}')

╔══════════════════════════════════════════════════════════════════════════════╗
║                              FIN DEL DIAGRAMA                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    print(diagram)


def print_interface_table():
    """Imprime una tabla de la interfaz común."""
    
    table = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TABLA DE MÉTODOS DE INTERFAZ COMÚN                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────┬──────────────┬──────────────────────────────────┐
│ Método                   │ Tipo         │ Descripción                      │
├──────────────────────────┼──────────────┼──────────────────────────────────┤
│ get_file_list()          │ Abstracto    │ Lista de archivos a procesar     │
│ load_all_data()          │ Abstracto    │ Carga todos los datos            │
│ load_subject_data()      │ Concreto     │ Carga datos de un sujeto         │
│ get_basic_info()         │ Concreto     │ Info: shape, memoria, columnas   │
│ get_subjects()           │ Concreto     │ Lista de sujetos únicos          │
│ get_fog_label_column()   │ Concreto     │ Identifica columna de FoG        │
│ get_summary_by_subject() │ Concreto     │ Resumen estadístico por sujeto   │
│ save_dataset()           │ Concreto     │ Guarda en CSV/Parquet/Pickle     │
│ read_dataset()           │ Concreto     │ Lee dataset desde archivo        │
│ plot_fog_distribution()  │ Concreto     │ Visualización de distribución    │
│ print_summary()          │ Concreto     │ Imprime resumen completo         │
└──────────────────────────┴──────────────┴──────────────────────────────────┘

Leyenda:
  • Abstracto: Debe ser implementado por cada loader
  • Concreto: Implementado en la clase base, puede ser sobrescrito

╔══════════════════════════════════════════════════════════════════════════════╗
║                    MÉTODOS ESPECÍFICOS POR DATASET                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────┬──────────────────────────────────────────────────────────────┐
│ Loader       │ Métodos Adicionales                                          │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Daphnet      │ (ninguno adicional - usa solo interfaz estándar)             │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Figshare     │ • load_metadata()                                            │
│              │ • get_summary_by_trial_type()                                │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Charite      │ • get_summary_by_foot()                                      │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Mendeley     │ • get_summary_by_task()                                      │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Kaggle       │ • load_file_data()                                           │
│              │ • get_summary_by_dataset_type()                              │
│              │ • get_summary_by_fog_type()                                  │
└──────────────┴──────────────────────────────────────────────────────────────┘
    """
    
    print(table)


if __name__ == "__main__":
    print_architecture_diagram()
    print("\n\n")
    print_interface_table()

