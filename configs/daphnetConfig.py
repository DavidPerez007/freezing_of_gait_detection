
from dataclasses import dataclass


@dataclass
class DaphnetConfig:
    """Configuración del dataset Daphnet."""
    
    # Nombres de las columnas según la documentación
    COLUMN_NAMES = [
        'time_ms',                    # Tiempo en milisegundos
        'ankle_acc_forward',          # Aceleración tobillo - horizontal forward [mg]
        'ankle_acc_vertical',         # Aceleración tobillo - vertical [mg]
        'ankle_acc_lateral',          # Aceleración tobillo - horizontal lateral [mg]
        'thigh_acc_forward',          # Aceleración muslo - horizontal forward [mg]
        'thigh_acc_vertical',         # Aceleración muslo - vertical [mg]
        'thigh_acc_lateral',          # Aceleración muslo - horizontal lateral [mg]
        'trunk_acc_forward',          # Aceleración tronco - horizontal forward [mg]
        'trunk_acc_vertical',         # Aceleración tronco - vertical [mg]
        'trunk_acc_lateral',          # Aceleración tronco - horizontal lateral [mg]
        'annotation'                  # Anotación: 0=No experimento, 1=No freeze, 2=Freeze
    ]
    

    # Frecuencia de muestreo
    SAMPLING_RATE = 64  # Hz
    
    # Unidades
    ACCELERATION_UNIT = 'mg'  # mili-g