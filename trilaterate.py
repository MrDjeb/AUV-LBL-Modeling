import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

def trilaterate(points, distances, weights=None, initial_guess=None):
    """
    Реализация алгоритма трилатерации с использованием взвешенного нелинейного МНК
    
    Параметры:
    points - массив опорных точек в формате [[x1, y1], [x2, y2], ..., [xn, yn]]
    distances - массив измеренных расстояний [d1, d2, ..., dn]
    weights - массив весовых коэффициентов (по умолчанию все 1)
    initial_guess - начальное предположение для позиции [x0, y0]
    
    Возвращает:
    Оптимальные координаты (x, y) в виде numpy массива
    """
    # Проверка входных данных
    points = np.asarray(points)
    if len(points) < 2:
        raise ValueError("Нужно минимум 2 опорные точки")
    if len(points) != len(distances):
        raise ValueError("Количество точек и расстояний должно совпадать")
        
    # Установка весов по умолчанию
    if weights is None:
        weights = np.ones_like(distances)
    else:
        weights = np.asarray(weights)
    
    # Начальное предположение (центроид точек)
    if initial_guess is None:
        initial_guess = np.mean(points, axis=0)
    
    # Функция невязок с учетом весов
    def residuals(params):
        calculated_dists = np.linalg.norm(points - params, axis=1)
        return np.sqrt(weights) * (calculated_dists - distances)
    
    # Минимизация невязок
    result = least_squares(residuals, 
                          initial_guess, 
                          method='lm')  # Используем метод Левенберга-Марквардта
    
    if not result.success:
        raise RuntimeError("Оптимизация не удалась: " + result.message)
    
    return result.x

# Пример использования
if __name__ == "__main__":
    # Опорные точки (x, y)
    # anchors = np.array([
    #     [0, 0],
    #     [10, 0],
    #     [5, 10]
    # ])
    BEACON_DEPTH = -100

    beacons = np.array([
        [300, 800, BEACON_DEPTH],  # Маяк 2
        [1000, 0, BEACON_DEPTH],   # Маяк 3
        [1000, 800, BEACON_DEPTH],  # Маяк 4
        #[500, 700, BEACON_DEPTH],  # Маяк 4
        #[400, 600, BEACON_DEPTH],  # Маяк 4
    ])
    P = np.array([360.01354331, 707.5011592,  582.06981526])
    
    # Истинная позиция
    true_position = np.array([5, 5])
    
    # Рассчитаем точные расстояния
    # exact_dists = np.linalg.norm(anchors - true_position, axis=1)
    
    # Добавим шум и создадим веса
    # np.random.seed(42)
    # noisy_dists = exact_dists + np.random.normal(0, 0.3, size=exact_dists.shape)
    # weights = 1/np.array([0.1, 0.2, 0.3])  # Пример весов (обратно пропорциональны дисперсии)
    
    # Выполним трилатерацию
    estimated_pos = trilaterate(beacons[:, :2], P)
    
    print(f"Истинная позиция: {true_position}")
    print(f"Расчетная позиция: {estimated_pos}")
    print(f"Ошибка: {np.linalg.norm(true_position - estimated_pos):.4f}")