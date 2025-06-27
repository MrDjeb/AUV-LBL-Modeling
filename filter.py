import numpy as np

def digital_filter(x, dt, filter_type='lowpass', tau=0.1):
    """Цифровой фильтр первого порядка
    
    Параметры:
    x - входной сигнал
    dt - шаг дискретизации
    filter_type - 'lowpass' или 'highpass'
    tau - постоянная времени фильтра
    """
    y = np.zeros_like(x)
    alpha = dt / (tau + dt)
    print(alpha, dt, tau)
    if filter_type == 'lowpass':
        prev_output = x[0] if len(x) > 0 else 0.0
        for i in range(len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * prev_output
            prev_output = y[i]
    
    elif filter_type == 'highpass':
        prev_input = x[0] if len(x) > 0 else 0.0
        prev_output = 0
        for i in range(len(x)):
            y[i] = alpha * (prev_output + x[i] - prev_input)
            prev_output = y[i]
            prev_input = x[i]
            #print(x[i])
    
    return y
