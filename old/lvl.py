"""
Модуль для моделирования ГАНС с ДБ на основе МНК
Автор: Панькин Глеб (2025)
Версия: 1.0
"""

import numpy as np
from scipy.optimize import least_squares

from trilaterate import trilaterate, averaged_position
from trajectory import interpolate_trajectory, generate_measurements
from draw import draw_graphs, draw_errors, draw_show

# Глобальные константы
SPEED_OF_SOUND = 1500  # Скорость звука в воде (м/с)


NOISE = 5 #5c = 7.5m
WEIGHTS = 1 / (NOISE ** 2)

UPDATE_RATE = 5 #2сек
V_MARCH = 0.8 #0.5м/с

STEP = V_MARCH * UPDATE_RATE 
BEACON_DEPTH = -100     # Глубина установки маяков (м)

# Опорные точки меандровой траектории (X, Y, Z)
meander_points = np.array([
    [500, 300, -50],      # Точка 1
    [500, 500, -50],      # Точка 2
    [550, 500, -50],      # Точка 3
    [550, 300, -50],      # Точка 4
    [600, 300, -50],      # Точка 5
    [600, 500, -50],      # Точка 6
    [650, 500, -50],      # Точка 7
    [650, 300, -50],      # Точка 8
    [700, 300, -50],      # Точка 9
    [700, 500, -50],      # Точка 10
    [750, 500, -50],      # Точка 11
    [750, 300, -50],      # Точка 12
    [900, 300, -50]      # Точка 13
])


# Координаты маяков (X, Y, Z) в метрах
beacons = np.array([
    #[300, 0, BEACON_DEPTH],    # Маяк 1
    #[300, 800, BEACON_DEPTH],  # Маяк 2
    #[1000, 0, BEACON_DEPTH],   # Маяк 3
    #[1000, 800, BEACON_DEPTH],  # Маяк 4
    [400, 200, BEACON_DEPTH],  
    [400, 600, BEACON_DEPTH],
    [1000, 200, BEACON_DEPTH],  
    [1000, 600, BEACON_DEPTH],  
])


def simulate_lbl(trajectory, measurements):
    """
    Основной цикл моделирования навигационной системы
    """
    estimated_positions = []
    current_estimate = trajectory[0].copy()
    
    for k in range(0, len(trajectory)):

        projection = np.array([np.sqrt(measurements[k][i]**2 - (trajectory[k][2] - beacons[i][2])**2) 
                     for i in range(len(beacons))])

        current_estimate = averaged_position(beacons, projection)
        estimated_positions.append(current_estimate.copy())
    
    return np.array(estimated_positions)

def simulate_lbl_MNK(trajectory, measurements):
    """
    Основной цикл моделирования навигационной системы
    """
    estimated_positions = []
    errors = []
    current_estimate = trajectory[0].copy()
    
    for k in range(0, len(trajectory)):

        projection = np.array([np.sqrt(measurements[k][i]**2 - (trajectory[k][2] - beacons[i][2])**2) 
                     for i in range(len(beacons))])

        current_estimate = trilaterate(beacons[:, :2], projection, WEIGHTS)
        #print(f"Ошибка: {np.linalg.norm(trajectory[k][:2] - current_estimate):.4f}")
        if np.linalg.norm(trajectory[k][:2] - current_estimate) < STEP:
            estimated_positions.append(current_estimate.copy())
            errors.append(np.linalg.norm(trajectory[k][:2] - current_estimate))
    
    return np.array(estimated_positions), np.array(errors)

def calculate_distances(position):
    """
    Вычисление расстояний от текущей позиции до всех маяков
    """
    return np.linalg.norm(beacons - position, axis=1)

def residuals(position, measured_distances):
    return calculate_distances(position) - measured_distances

def simulate_lbl_MNK2(trajectory, measurements):
    """Оценка позиций на основе сохраненных измерений"""
    estimated_positions = []
    current_estimate = trajectory[0].copy()
    
    for measured_dists in measurements:
        result = least_squares(
            residuals,
            current_estimate,
            args=(measured_dists,),
            bounds=([0, 0, -200], [1500, 1500, 0])
        )
        current_estimate = result.x
        estimated_positions.append(current_estimate)
    
    return np.array(estimated_positions)


# Визуализация результатов
if __name__ == "__main__":
    trajectory = interpolate_trajectory(meander_points, step=STEP)
    noisy_dists, weights = generate_measurements(trajectory, beacons, noise_std=NOISE)


    estimated_positions = simulate_lbl(trajectory, noisy_dists)
    estimated_positions_MNK, errors = simulate_lbl_MNK(trajectory, noisy_dists)
    estimated_positions_MNK2 = simulate_lbl_MNK2(trajectory, noisy_dists)

    
    draw_graphs(estimated_positions, beacons, 'Исходная траектория')
    draw_graphs(estimated_positions_MNK, beacons, 'Оценка позиции')
    draw_graphs(estimated_positions_MNK2, beacons, 'Оценка позиции без отбраковки скачков')

    draw_errors(np.linalg.norm(trajectory[:, :2] - estimated_positions, axis=1), STEP, 'График ошибки')
    draw_errors(errors, STEP, 'График ошибки')
    draw_errors(np.linalg.norm(trajectory[:, :2] - estimated_positions_MNK2[:, :2], axis=1), STEP, 'График ошибки без отбраковки скачков')

    draw_show()
