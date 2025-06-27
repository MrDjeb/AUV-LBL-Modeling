import json
import numpy as np
from pathlib import Path


def load_config_from_file(filename):
    """Загружает конфигурацию из JSON файла"""
    with open(filename) as f:
        loaded_config = json.load(f)
    
    # Конвертируем строковые ключи обратно в float
    loaded_config['course_changes'] = {
        float(k): v for k, v in loaded_config['course_changes'].items()
    }
    loaded_config['beacons'] = np.array(loaded_config['beacons'], dtype=np.float64)
    
    return loaded_config

def load_init_config(filename='init.json'):
    """Загружает начальные параметры из JSON файла"""
    if not Path(filename).exists():
        raise FileNotFoundError(f"Файл {filename} не найден")
    
    with open(filename) as f:
        config = json.load(f)
    
    # Проверка обязательных полей
    required_keys = ['meander_points', 'velocity', 'dt', 'beacons', 'system_period', 'distance_noise']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Отсутствует обязательный ключ: {key}")
    
    config['meander_points'] = np.array(config['meander_points'], dtype=np.float64)
    config['beacons'] = np.array(config.get('beacons', []), dtype=np.float64)
    config['velocity'] = float(config['velocity'])
    config['dt'] = float(config['dt'])
    config['system_period'] = float(config['system_period'])
    config['ssp_period'] = float(config['ssp_period'])
    config['distance_noise'] = float(config['distance_noise'])
    
    return config

def save_config_to_file(config, filename):
    """Сохраняет конфигурацию симуляции в JSON файл с конвертацией numpy-типов"""
    
    def convert_numpy_types(obj):
        """Рекурсивно конвертирует numpy-типы в стандартные Python-типы"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    # Создаем копию конфига с конвертированными типами
    config_to_save = convert_numpy_types(config.copy())

    # Дополнительная конвертация ключей времени в строки
    config_to_save['course_changes'] = {
        str(k): v for k, v in config_to_save['course_changes'].items()
    }
    

    with open(filename, 'w') as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
    print(f"Конфиг успешно сохранен в {filename}")

# Модифицируем generate_course_schedule для совместимости
def generate_course_schedule(init_config):
    """
    Генерирует полный конфиг для симуляции движения по маршруту
    
    Параметры:
    points - массив точек маршрута [[x,y,z], ...]
    velocity - скорость движения (м/с)
    dt - шаг времени (сек)
    
    Возвращает:
    simulation_config - словарь с параметрами для simulate_navigation
    """

    points=init_config['meander_points']
    velocity=init_config['velocity']
    beacons=init_config['beacons']
    
    if len(points) < 2:
        raise ValueError("Нужно минимум 2 точки для построения маршрута")

    if beacons is None:
        beacons = np.empty((0,3))  # Пустой массив по умолчани

    # Извлекаем начальные координаты из первой точки
    x0, y0, _ = points[0]
    
    # Рассчитываем параметры сегментов маршрута
    segments = []
    for i in range(len(points)-1):
        start = points[i]
        end = points[i+1]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Рассчитываем курс и длину сегмента
        heading = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)
        time = length / velocity if velocity > 0 else 0
        
        segments.append((heading, length, time))
    
    # Рассчитываем общее время и циклограмму
    total_time = sum(seg[2] for seg in segments)
    course_changes = {}
    cumulative_time = 0.0
    
    # Первый курс
    initial_heading = segments[0][0]
    
    # Формируем моменты смены курса
    for i in range(1, len(segments)):
        cumulative_time += segments[i-1][2]
        course_changes[round(cumulative_time, 2)] = segments[i][0]

    # Формируем итоговый конфиг
    return {
        'x0': np.float64(x0),
        'y0': np.float64(y0),
        'initial_heading': np.float64(initial_heading),
        'initial_velocity': np.float64(velocity),
        'course_changes': {
            np.float64(k): np.float64(v) for k, v in course_changes.items()
        },
        'total_time': np.float64(total_time),
        'dt': np.float64(init_config['dt']),
        'beacons': beacons.copy(),
        'system_period': np.float64(init_config['system_period']),
        'distance_noise': np.float64(init_config['distance_noise'])
    }

# Пример использования
if __name__ == "__main__":
    # Загрузка параметров из файла
    init_config = load_init_config('config/init.json')
    
    # Генерация конфига для симуляции
    simulation_config = generate_course_schedule(init_config)
    
    # Сохранение в файл
    save_config_to_file(simulation_config, 'config/simulation_config.json')
    
    # Загрузка из файла в другой программе
    loaded_config = load_config_from_file('config/simulation_config.json')
    
    # Проверка целостности данных
    print("\nПроверка загруженного конфига:")
    for key, value in loaded_config.items():
        if key == 'course_changes':
            print(f"{key}: {len(value)} записей")
        else:
            print(f"{key}: {value}")