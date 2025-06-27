import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def digital_filter_freq_response(dt, tau, filter_type='lowpass', f_range=(0.1, 10000)):
    """
    Построение ЛАЧХ и ФЧХ цифрового фильтра первого порядка
    
    Параметры:
    dt - шаг дискретизации
    tau - постоянная времени фильтра
    filter_type - 'lowpass' или 'highpass'
    f_range - диапазон частот для анализа (Гц)
    """
    # Рассчет коэффициента фильтра
    alpha = dt / (tau + dt)
    
    # Определение коэффициентов фильтра
    if filter_type == 'lowpass':
        b = [alpha]          # Коэффициенты числителя
        a = [1, -(1 - alpha)]  # Коэффициенты знаменателя
    elif filter_type == 'highpass':
        b = [alpha, -alpha]  # Коэффициенты числителя
        a = [1, -(1 - alpha)]  # Коэффициенты знаменателя
    
    # Создание передаточной функции
    sys = signal.TransferFunction(b, a, dt=dt)
    
    # Генерация частотного диапазона
    frequencies = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), 500)
    
    # Расчет частотной характеристики
    w, mag, phase = signal.dbode(sys, w=frequencies * 2 * np.pi)
    
    # Преобразование в Гц
    f = w / (2 * np.pi)
    
    # Построение графиков
    plt.figure(figsize=(14, 10))
    
    # ЛАЧХ
    plt.subplot(2, 1, 1)
    plt.semilogx(f, mag)
    plt.title(f'ЛАЧХ {filter_type} фильтра (τ={tau} сек, α={alpha:.4f})')
    plt.ylabel('Амплитуда, дБ')
    plt.grid(True, which='both', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--', label='-3 дБ')
    
    # Частота среза (f_c = 1/(2πτ))
    f_cutoff = 1 / (2 * np.pi * tau)
    plt.axvline(x=f_cutoff, color='g', linestyle=':', label=f'f_c={f_cutoff:.2f} Гц')
    plt.legend()
    
    # ФЧХ
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase)
    plt.title(f'ФЧХ {filter_type} фильтра')
    plt.xlabel('Частота, Гц')
    plt.ylabel('Фаза, градусы')
    plt.grid(True, which='both', linestyle='--')
    plt.axvline(x=f_cutoff, color='g', linestyle=':', label=f'f_c={f_cutoff:.2f} Гц')
    plt.axhline(y=-45, color='r', linestyle='--', label='-45°')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return f, mag, phase

# Параметры для анализа
dt = 1500  # шаг дискретизации [сек]
tau = 0.1  # постоянная времени [сек]

# Построение характеристик для ФНЧ
f_lp, mag_lp, phase_lp = digital_filter_freq_response(dt, tau, 'lowpass')

# Построение характеристик для ФВЧ
f_hp, mag_hp, phase_hp = digital_filter_freq_response(dt, tau, 'highpass')