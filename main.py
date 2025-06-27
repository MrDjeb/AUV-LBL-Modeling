"""
Система моделирования некоторых алгоритмов ГАНС с ДБ
Автор: Панькин Глеб (2025)
Версия: 1.0
"""

import numpy as np

from config.translator import load_config_from_file
from draw import draw_graphs, draw_errors, draw_show, draw_trajectory, draw_value
from simulate import trajectory_to_positions, simulate_lbl_navigation, simulate_vlbl_navigation
from filter import digital_filter

def lbl_main():
    lbl_config = load_config_from_file('dataset/lbl/2_1.json')
    trajectory, ssp_trajectory, gans_trajectory, true_positions, estimated_positions, estimated_clean_positions, period_steps, clean_errors, errors = simulate_lbl_navigation(**lbl_config)
    
    tau=15
    time_array = np.arange(0, lbl_config['total_time'], lbl_config['dt'])
    #gans_filtered_positions = digital_filter(gans_trajectory, lbl_config['dt'], filter_type='lowpass', tau=tau)
    #ssp_filtered_trajectory = digital_filter(ssp_trajectory, lbl_config['dt'], filter_type='highpass', tau=tau)
    draw_trajectory(gans_trajectory, 'Траектория ГАНС', lbl_config['beacons'])
    #draw_trajectory(ssp_trajectory, 'Траектория БАНС')
    #draw_trajectory(gans_filtered_positions, 'Траектория ФНЧ')
    #draw_trajectory(ssp_filtered_trajectory, 'Траектория ФВЧ')
    #draw_trajectory(ssp_filtered_trajectory+gans_filtered_positions, 'Траектория ФВЧ+ФНЧ')
    #draw_value(time_array, ssp_filtered_trajectory[:, 0], lbl_config['dt'])
    #draw_value(time_array, gans_trajectory[:, 0], lbl_config['dt'])
    #draw_value(time_array, ssp_trajectory[:, 0], lbl_config['dt'])
    XA = ssp_trajectory-gans_trajectory
    XB = digital_filter(XA, lbl_config['dt'], filter_type='lowpass', tau=tau)
    kompl_trajectory = ssp_trajectory-XB
    draw_trajectory(kompl_trajectory, 'Оценённая траектория КНС', lbl_config['beacons'])
    draw_trajectory(XB, 'Сигнал Xᵦ на выходе ФНЧ')
    
    gans_koml_positions = trajectory_to_positions(kompl_trajectory, lbl_config['system_period'], lbl_config['total_time'], lbl_config['dt'])

    #draw_graphs(trajectory, true_positions,    estimated_clean_positions,  lbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования ГАНС с ДБ')
    draw_graphs(trajectory, ssp_trajectory, true_positions,    np.array(gans_koml_positions),        lbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования КНС')
    print(f"Среднее значение ошибки, комплексированная: {round(np.mean(true_positions-gans_koml_positions), 2)}, Кол-во циклов обсервации {len(true_positions)}")
    print(f"Среднее значение ошибки, чисто ГАНС: {round(np.mean(errors[:, 1]), 2)}, Кол-во циклов обсервации {len(errors)}")

   
    #draw_graphs(trajectory, true_positions,    estimated_clean_positions,  lbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования ГАНС с ДБ')
    #draw_graphs(trajectory, true_positions,    estimated_positions,        lbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования ГАНС с ДБ, без отброски')
    #draw_errors(clean_errors, period_steps, 'Ошибки определения координат АНПА в процессе моделирования ГАНС с ДБ')
    #draw_errors(errors, period_steps, 'Ошибки определения координат АНПА в процессе моделирования ГАНС с ДБ, без отброски')
    
    #print(f"Среднее значение ошибки: {round(np.mean(clean_errors[:, 1]), 2)}, Кол-во циклов обсервации {len(clean_errors)}")
    #print(f"Среднее значение ошибки, без отброски: {round(np.mean(errors[:, 1]), 2)}, Кол-во циклов обсервации {len(errors)}")

    draw_show()

def vlbl_main():
    vlbl_config = load_config_from_file('dataset/vlbl/3_1.json')
    trajectory, ssp_trajectory, gans_trajectory, true_positions, estimated_positions, estimated_clean_positions, period_steps, clean_errors, errors = simulate_vlbl_navigation(**vlbl_config)

    tau=40
    time_array = np.arange(0, vlbl_config['total_time'], vlbl_config['dt'])
    XA = ssp_trajectory-gans_trajectory
    XB = digital_filter(XA, vlbl_config['dt'], filter_type='lowpass', tau=tau)
    kompl_trajectory = ssp_trajectory-XB
    draw_trajectory(gans_trajectory, 'Траектория ГАНС', vlbl_config['beacons'])
    draw_trajectory(kompl_trajectory, 'Оценённая траектория КНС', vlbl_config['beacons'])
    draw_trajectory(XB, 'Сигнал Xᵦ на выходе ФНЧ')

    gans_koml_positions = trajectory_to_positions(kompl_trajectory, vlbl_config['system_period']*vlbl_config['virtual_buffer_size'], vlbl_config['total_time'], vlbl_config['dt'])
    draw_graphs(trajectory, ssp_trajectory, true_positions,    np.array(gans_koml_positions),        vlbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования КНС')

    #draw_graphs(trajectory,ssp_trajectory, true_positions, estimated_clean_positions, vlbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования ГАНС с ВДБ, с отброской')
    #draw_graphs(trajectory, true_positions, estimated_positions, vlbl_config['beacons'], 'Траектория движения АНПА в процессе моделирования ГАНС с ВДБ')
    #draw_errors(clean_errors, period_steps, 'Ошибки определения координат АНПА в процессе моделирования ГАНС с ВДБ, с отброской')
    #draw_errors(errors, period_steps, 'Ошибки определения координат АНПА в процессе моделирования ГАНС с ВДБ')

    #print(f"Среднее значение ошибки: {round(np.mean(clean_errors[:, 1]), 2)}, Кол-во циклов обсервации {len(clean_errors)}")
    #print(f"Среднее значение ошибки, без отброски: {round(np.mean(errors[:, 1]), 2)}, Кол-во циклов обсервации {len(errors)}")
    print(f"Среднее значение ошибки, комплексированная: {round(np.mean(true_positions-gans_koml_positions), 2)}, Кол-во циклов обсервации {len(true_positions)}")
    print(f"Среднее значение ошибки, чисто ГАНС: {round(np.mean(errors[:, 1]), 2)}, Кол-во циклов обсервации {len(errors)}")

    draw_show()

if __name__ == "__main__":
    
    lbl_main()
