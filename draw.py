import numpy as np
import matplotlib.pyplot as plt

def draw_graphs(trajectory, ssp_trajectory,true_positions, estimated_positions, beacons, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_positions[:, 0], true_positions[:, 1], c='green', marker='D', s=15, label='Истинные точки')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'y-', linewidth=1.5, label='Истинная траектория')
    plt.plot(ssp_trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, label='Траектория ССП')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'k-', linewidth=0.5, label='Оцененная траектория')
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='blue', marker='x', s=25, label='Оцененные точки')
    plt.scatter(beacons[:, 0], beacons[:, 1], c='lime', marker='^', s=100, label='Маяки')
    plt.title(title, fontsize=14)
    plt.xlabel('ζ, м', fontsize=12)
    plt.ylabel('ξ, м', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

def draw_trajectory(trajectory, title, beacons=None):
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5, label=title)
    plt.title(title, fontsize=14)
    plt.xlabel('ζ, м', fontsize=12)
    plt.ylabel('ξ, м', fontsize=12)
    if beacons is not None:
        plt.scatter(beacons[:, 0], beacons[:, 1], c='lime', marker='^', s=100, label='Маяки')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

def draw_value(time_array, x, dt):
    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(time_array, x, label='x_ssp(t)', linewidth=2)

    # Настройка внешнего вида
    plt.title('Зависимость координаты x_ssp от времени')
    plt.xlabel('Время, сек')
    plt.ylabel('x_ssp, м')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Добавление дополнительной информации на график
    plt.annotate(f'Шаг dt = {dt} сек', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=10)

def draw_errors(errors, step,  title):
    ind = np.where(errors[:, 0] > 100)
    if len(ind[0]) > 0:
        errors = errors[:ind[0][0]]
        # Подписи для каждой 10-й точки на оси X
        xticks = np.arange(5, 101, 5)
    else:
        xticks = np.arange(5, len(errors)+1, 5)

    # Построение графика ошибки
    plt.figure(figsize=(16, 4))
    plt.plot(errors[:, 0], errors[:, 1], 'bo-', linewidth=1, markersize=5, label='Ошибка')
    plt.fill_between(errors[:, 0], errors[:, 1], color='blue', alpha=0.1)
    # Горизонтальная линия с константой
    # plt.axhline(
    #         y=step, 
    #         color='red', 
    #         linestyle='--', 
    #         linewidth=1, 
    #         label=f'Порог отброски {step} м'  # Подпись в легенде
    #     )
    # Настройки графика
    plt.title(title, fontsize=14)
    plt.xlabel('Номер обсервации', fontsize=12)
    plt.ylabel('Погрешность координаты, м', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    plt.xticks(xticks)  # Устанавливаем метки

def draw_show():
    plt.show()