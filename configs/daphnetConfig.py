
from dataclasses import dataclass


@dataclass
class DaphnetConfig:
    """Configuración del dataset Daphnet."""
    
    # Nombres de las columnas según la documentación
    # Convención: {sensor}_{eje}_{posición_corporal}
    COLUMN_NAMES = [
        'time_ms',                    # Tiempo en milisegundos
        'acc_forward_ankle',          # Aceleración tobillo - horizontal forward [mg]
        'acc_vertical_ankle',         # Aceleración tobillo - vertical [mg]
        'acc_lateral_ankle',          # Aceleración tobillo - horizontal lateral [mg]
        'acc_forward_thigh',          # Aceleración muslo - horizontal forward [mg]
        'acc_vertical_thigh',         # Aceleración muslo - vertical [mg]
        'acc_lateral_thigh',          # Aceleración muslo - horizontal lateral [mg]
        'acc_forward_trunk',          # Aceleración tronco - horizontal forward [mg]
        'acc_vertical_trunk',         # Aceleración tronco - vertical [mg]
        'acc_lateral_trunk',          # Aceleración tronco - horizontal lateral [mg]
        'annotation'                  # Anotación: 0=No experimento, 1=No freeze, 2=Freeze
    ]
    

    # Frecuencia de muestreo
    SAMPLING_RATE = 64  # Hz
    
    # Unidades
    ACCELERATION_UNIT = 'mg'  # mili-g