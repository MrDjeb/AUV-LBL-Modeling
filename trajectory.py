import numpy as np


def interpolate_trajectory(points, step=1):
    """
    Интерполяция траектории с фиксированным шагом между измерениями.
    Возвращает:
    - interp_points: массив координат [X, Y, Z] с сохранением последовательности.
    """
    # Создаем пустой список для результатов
    interp_points = [points[0]]  # Начинаем с первой точки
    
    # Проходим по всем сегментам траектории
    for i in range(1, len(points)):
        start_point = points[i - 1]
        end_point = points[i]
        
        # Рассчитываем разницу между точками
        segment = end_point - start_point
        length = np.linalg.norm(segment)
        
        if length == 0:
            continue  # Игнорируем нулевые сегменты
        
        # Направление сегмента (единичный вектор)
        direction = segment / length
        
        # Количество шагов в текущем сегменте
        num_steps = int(np.ceil(length / step))
        
        # Генерируем точки вдоль сегмента
        for k in range(1, num_steps + 1):
            dist = min(k * step, length)  # Чтобы не выйти за сегмент
            new_point = start_point + direction * dist
            interp_points.append(new_point)
    
    return np.array(interp_points)

# def interpolate_trajectory(points, num_steps):
#     """
#     Интерполяция опорных точек траектории
#     """
#     t = np.linspace(0, 1, len(points))
#     interpolators = [interp1d(t, points[:, i], kind='linear') for i in range(3)]
#     t_new = np.linspace(0, 1, num_steps)
#     return np.column_stack([f(t_new) for f in interpolators])


def generate_measurements(trajectory, beacons, noise_std=1):
    """
    Генерация зашумленных измерений расстояний и соответствующих весов.
    
    Параметры:
    trajectory - массив позиций траектории в формате [[x1, y1], [x2, y2], ..., [xn, yn]]
    beacons - массив координат маяков в формате [[x1, y1], [x2, y2], ..., [xm, ym]]
    noise_std - стандартное отклонение шума (по умолчанию 1)
    
    Возвращает:
    noisy_dists - зашумленные расстояния для каждой позиции траектории
    weights - веса, обратно пропорциональные дисперсии шума
    """
    # Вычисляем точные расстояния для каждой позиции траектории до маяков
    exact_dists = np.array([np.linalg.norm(beacons - pos, axis=1) for pos in trajectory])
    
    # Генерируем шум с заданным стандартным отклонением
    noise = np.random.normal(0, noise_std, size=exact_dists.shape)
    
    # Добавляем шум к точным расстояниям
    noisy_dists = exact_dists + noise
    
    # Вычисляем веса как обратные величины дисперсии шума
    if noise_std == 0:
        weights = np.ones_like(noisy_dists)
    else:
        weights = 1 / (noise_std ** 2)
    
    return noisy_dists, weights
    
if __name__ == "__main__":
    meander_points = np.array([
        [0, 0, -50],     
        [0, 2, -50],     
        [2, 2, -50],     
        [2, 0, -50],     
        [4, 0, -50],     
    ])

    interp_points = interpolate_trajectory(meander_points, step=1)

    # Вывод результата
    print("Интерполированные точки:")
    print(interp_points)

    import matplotlib.pyplot as plt

    # Визуализация
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(interp_points[:, 0], interp_points[:, 1], interp_points[:, 2], '-o', label="Интерполированная траектория")
    ax.scatter(meander_points[:, 0], meander_points[:, 1], meander_points[:, 2], c='red', s=100, label="Опорные точки")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
