"""
Модуль для моделирования ГАНС с ВДБ
Автор: Панькин Глеб (2025)
Версия: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from trilaterate import trilaterate, averaged_position
from trajectory import interpolate_trajectory, generate_measurements

# Глобальные константы
SPEED_OF_SOUND = 1500  # Скорость звука в воде (м/с)


NOISE = 0 #5c = 7.5m
#WEIGHTS = 1 / (NOISE ** 2)

UPDATE_RATE = 5 #2сек
V_MARCH = 1 #0.5м/с

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
    #[400, 200, BEACON_DEPTH],  
    #[1100, 450, BEACON_DEPTH],
    #[650, 200, BEACON_DEPTH],  
    [300, 400, BEACON_DEPTH],  
])


def solve_xy_position(beacons, P):
    """
    Решает систему уравнений вида (x - xi)^2 + (y - yi)^2 = Pi^2
    
    Параметры:
    beacons - массив координат маяков [[x1, y1, z1], ..., [xN, yN, zN]]
    P - массив проекций расстояний на плоскость XY [P1, ..., PN]
    
    Возвращает:
    Координаты (x, y) в виде numpy-массива [x, y]
    """
    if len(beacons) < 2:
        raise ValueError("Требуется как минимум 2 маяка для решения")
    
    # Выбираем первый маяк в качестве опорного
    x0, y0, z0 = beacons[0]
    P0 = P[0]
    
    # Формируем матрицу коэффициентов и вектор свободных членов
    A = []
    b = []
    
    for i in range(1, len(beacons)):
        xi, yi, zi = beacons[i]
        xp, yp, zp = beacons[i-1]
        
        a_x = 2*(xi - xp)
        a_y = 2*(yi - yp)
        rhs = (P[i]**2 - P[i-1]**2) + (xp**2 - xi**2) + (yp**2 - yi**2)
        
        A.append([a_x, a_y])
        b.append(rhs)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        # Решение методом наименьших квадратов
        solution = np.linalg.lstsq(A, b, rcond=1e-6)[0]
        return np.array([-solution[0], -solution[1], 0])
    
    except np.linalg.LinAlgError:
        raise ValueError("Система уравнений вырождена. Проверьте расположение маяков")


def averaged_position(beacons, P):
    """
    Вычисляет усреднённые координаты по всем возможным комбинациям маяков
    
    Параметры:
    beacons - массив координат маяков [[x1,y1,z1], ..., [xn,yn,zn]]
    P - массив проекций расстояний [P1, ..., Pn]
    
    Возвращает:
    Усреднённые координаты (x, y) в виде numpy-массива [x_avg, y_avg]
    """
    n = len(beacons)
    if n < 3:
        raise ValueError("Требуется минимум 3 маяка для комбинаций")
    
    # Генерируем все комбинации по (n-1) маяков
    comb_indices = list(combinations(range(n), n-1))
    #print("=====================")
    solutions = []
    for indices in comb_indices:
        try:
            # Выбираем соответствующие маяки и измерения
            selected_beacons = beacons[list(indices)]
            selected_P = P[list(indices)]
            
            # Вычисляем позицию для текущей комбинации
            pos = trilaterate(selected_beacons[:, :2], selected_P)
            #print(selected_beacons, selected_P, pos)
            solutions.append(pos)
        except:
            continue  # Пропускаем невалидные комбинации
    
    if not solutions:
        raise ValueError("Не удалось вычислить ни одного решения")
    
    # Усреднение всех валидных решений
    solutions_array = np.array(solutions)
    #print(np.mean(solutions_array, axis=0))
    return np.mean(solutions_array, axis=0)


def simulate_vlbl(trajectory, measurements, buffer_size=4):
    """
    Моделирование VLBL-навигатора с использованием одного физического транспондера и
    накоплением buffer_size исторических измерений.
    
    trajectory - массив положений аппарата (X, Y, Z)
    measurements - измеренные расстояния до физического транспондера (одномерный массив)
    buffer_size - число измерений, используемых для формирования виртуальной сети
    """
    estimated_positions = []
    # Буферы для хранения исторических измерений и соответствующих положений аппарата
    measurement_buffer = []
    position_buffer = []
    
    for k in range(len(trajectory)):
        current_pos = trajectory[k]
        # Текущее измерение (для одного транспондера, индекс 0)
        measured_range = measurements[k][0]
        
        # Добавляем измерение и позицию в буфер
        measurement_buffer.append(measured_range)
        position_buffer.append(current_pos.copy())
        
        # Если накоплено недостаточно измерений, можно использовать текущее положение
        if len(measurement_buffer) < buffer_size:
            #estimated_positions.append(current_pos[:2].copy())
            continue
        
        # Для каждого измерения в буфере вычисляем виртуальное положение транспондера
        virtual_beacon_positions = []
        horizontal_ranges = []
        for i in range(buffer_size):
            # Определяем смещение от момента измерения до текущего момента
            delta = current_pos - position_buffer[i]
            #print(delta, current_pos - position_buffer[i])
            # Виртуальное положение транспондера: физический транспондер + смещение
            virtual_beacon = beacons[0] + delta
            virtual_beacon_positions.append(virtual_beacon)
            # Вычисляем горизонтальную проекцию измеренного расстояния
            # (глубины считаем постоянными: current_pos[2] и BEACON_DEPTH)
            try:
                horizontal_range = np.sqrt(measurement_buffer[i]**2 - (current_pos[2] - BEACON_DEPTH)**2)
            except ValueError:
                horizontal_range = 0  # или другое корректное значение, если подкоренное выражение отрицательно
            horizontal_ranges.append(horizontal_range)
        
        virtual_beacon_positions = np.array(virtual_beacon_positions)
        horizontal_ranges = np.array(horizontal_ranges)
        
        # Расчёт текущей 2D-позиции аппарата на основе виртуальных маяков и горизонтальных расстояний
        print(virtual_beacon_positions[:, :2], horizontal_ranges)
        current_estimate_2d = trilaterate(virtual_beacon_positions[:, :2], horizontal_ranges)
        estimated_positions.append(current_estimate_2d.copy())
        
        # Удаляем самое старое измерение, чтобы поддерживать размер буфера
        measurement_buffer.clear()
        position_buffer.clear()
        # measurement_buffer.pop(0)
        # position_buffer.pop(0)  
          
    return np.array(estimated_positions)



def draw_graphs(estimated_positions, title):
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'y-', linewidth=1.5, label='Исходная траектория')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'k-', linewidth=0.5, label='Оценённая траектория')
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='blue', marker='x', s=25, label='Оценённые точки')
    plt.scatter(beacons[:, 0], beacons[:, 1], c='lime', marker='^', s=100, label='Маяки')
    plt.title(title, fontsize=14)
    plt.xlabel('X (м)', fontsize=12)
    plt.ylabel('Y (м)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

def draw_errors(errors, title):
    errors = errors[:50]
    point_numbers = np.arange(1, len(errors) + 1)
    # Построение графика ошибки
    plt.figure(figsize=(10, 6))
    plt.plot(point_numbers, errors, 'bo-', linewidth=1, markersize=5, label='Ошибка')
    plt.fill_between(point_numbers, errors, color='blue', alpha=0.1)
    # Горизонтальная линия с константой
    plt.axhline(
            y=STEP, 
            color='red', 
            linestyle='--', 
            linewidth=1, 
            label=f'y = {STEP} м'  # Подпись в легенде
        )
    # Настройки графика
    plt.title(title, fontsize=14)
    plt.xlabel('Номер точки', fontsize=12)
    plt.ylabel('Ошибка (м)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    # Подписи для каждой 10-й точки на оси X
    xticks = np.arange(10, len(errors) + 1, 10)
    plt.xticks(xticks)  # Устанавливаем метки


if __name__ == "__main__":
    trajectory = interpolate_trajectory(meander_points, step=STEP)
    noisy_dists, weights = generate_measurements(trajectory, beacons, noise_std=NOISE)
    
    estimated_positions_vlbl = simulate_vlbl(trajectory, noisy_dists)
    
    # Визуализация результатов
    draw_graphs(estimated_positions_vlbl, 'Оценка позиции VLBL')
    #draw_errors(np.linalg.norm(trajectory[:, :2] - estimated_positions_vlbl, axis=1), 'График ошибки VLBL')
    
    plt.show()

