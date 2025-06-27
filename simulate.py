import numpy as np

from trilaterate import trilaterate
from robot import UnderwaterRobot

def trajectory_to_positions(trajectory, system_period, total_time, dt):
    positions = []
    period_index = int(system_period / dt)
    steps = int(total_time / dt)
    for i in range(steps):
        if i % period_index == 0:
            positions.append(trajectory[i][:2])
    return np.array(positions)

def simulate_lbl_navigation(
    x0=0,
    y0=0,
    z0=-100,
    initial_heading=0,
    initial_velocity=1.0,
    total_time=10,
    dt=0.1,
    system_period=20,
    distance_noise=None,
    course_changes=None,
    beacons=None,
    velocity_error=0,
    heading_error=0,
    ssp_period=1
):
    """
    Параметры:
    initial_heading - начальный курс (градусы)
    initial_velocity - маршевая скорость (м/с)
    total_time - общее время моделирования (сек)
    dt - шаг времени (сек)
    course_changes - словарь {время: новый_курс}
    noise_params - параметры шумов (можно добавить позже)
    """
    print(f"Velocity Error: {velocity_error}, Heading Error: {heading_error}, SSP Period: {ssp_period}")

    robot = UnderwaterRobot(x0=x0, y0=y0, z0=z0, velocity=initial_velocity, heading=initial_heading, velocity_error=velocity_error, heading_error=heading_error)
    steps = int(total_time / dt)
    system_period_index = int(system_period / dt)
    ssp_period_index = int(ssp_period / dt)
    period_steps = initial_velocity * system_period  

    trajectory = np.zeros((steps, 3))
    ssp_trajectory = np.zeros((steps, 3))
    gans_trajectory = np.zeros((steps, 3))

    true_positions = []
    true_clean_positions = []
    estimated_positions = []
    estimated_clean_positions = []
    clean_errors = []
    errors = []
    k = 0
    for i in range(steps):
        current_time = i * dt
        trajectory[i] = [robot.x, robot.y, robot.z]
        ssp_trajectory[i] = [robot.x_ssp, robot.y_ssp, robot.z]
        gans_trajectory[i] = [robot.x_gans, robot.y_gans, robot.z]


        new_heading = None

        # Получение измерений от маяков
        if beacons is not None:
            if i % system_period_index == 0:
                _, projection, weights = robot.get_measurements(beacons, distance_noise)
                #print(beacons, projection)

                if k > 0:
                    current_estimate = trilaterate(beacons[:, :2], projection, weights, estimated_positions[k-1])
                    epsilon_error = np.linalg.norm(estimated_positions[k-1] - current_estimate)
                else:
                   current_estimate = trilaterate(beacons[:, :2], projection, weights, [x0, y0])
                   epsilon_error = np.linalg.norm([x0, y0] - current_estimate)


                # current_estimate = trilaterate(beacons[:, :2], projection, weights) # TODO fix estimated_positions[k-1]
                # epsilon_error = np.linalg.norm([robot.x, robot.y] - current_estimate) # TODO fix estimated_positions[k-1]
                if epsilon_error < period_steps*1.2:
                    estimated_clean_positions.append(current_estimate)
                    true_clean_positions.append([robot.x, robot.y])
                    clean_errors.append([k, np.linalg.norm([robot.x, robot.y] - current_estimate)])
      
                errors.append([k, np.linalg.norm([robot.x, robot.y] - current_estimate)])
                true_positions.append([robot.x, robot.y])
                estimated_positions.append(current_estimate)

                robot.update_gans(current_estimate[0], current_estimate[1])

                k = k + 1

        
        # Применение циклограммы курса
        if course_changes and current_time in course_changes:
            new_heading = course_changes[current_time]
        robot.update_position(dt, new_heading)

        if i % ssp_period_index == 0:
            robot.update_ssp(ssp_period)

    return (
        trajectory,
        ssp_trajectory,
        gans_trajectory,
        np.array(true_positions),
        np.array(estimated_positions),
        np.array(estimated_clean_positions),
        period_steps,
        np.array(clean_errors),
        np.array(errors)
    )

def simulate_vlbl_navigation(
    x0=0,
    y0=0,
    z0=-100,
    initial_heading=0,
    initial_velocity=1.0,
    total_time=10,
    dt=0.1,
    system_period=20,
    distance_noise=None,
    course_changes=None,
    beacons=None,
    virtual_buffer_size=4,
    velocity_error=0,
    heading_error=0,
    ssp_period=1
):
    """
    Параметры:
    initial_heading - начальный курс (градусы)
    initial_velocity - маршевая скорость (м/с)
    total_time - общее время моделирования (сек)
    dt - шаг времени (сек)
    course_changes - словарь {время: новый_курс}
    noise_params - параметры шумов (можно добавить позже)
    """
    robot = UnderwaterRobot(x0=x0, y0=y0, z0=z0, velocity=initial_velocity, heading=initial_heading, velocity_error=velocity_error, heading_error=heading_error)
    steps = int(total_time / dt)
    system_period_index = int(system_period / dt)
    ssp_period_index = int(ssp_period / dt)
    period_steps = initial_velocity * system_period * virtual_buffer_size 

    trajectory = np.zeros((steps, 3))
    ssp_trajectory = np.zeros((steps, 3))
    gans_trajectory = np.zeros((steps, 3))

    true_positions = []
    true_clean_positions = []
    estimated_positions = []
    estimated_clean_positions = []
    clean_errors = []
    errors = []
    k = 0
    print(f"Velocity Error: {velocity_error}, Heading Error: {heading_error}, SSP Period: {ssp_period}")


    projection_buffer = []
    position_buffer = []
    
    for i in range(steps):
        current_time = i * dt
        trajectory[i] = [robot.x, robot.y, robot.z]
        ssp_trajectory[i] = [robot.x_ssp, robot.y_ssp, robot.z]
        gans_trajectory[i] = [robot.x_gans, robot.y_gans, robot.z]
        new_heading = None

        # Получение измерений от маяков
        if beacons is not None:
            if i % system_period_index == 0:
                _, projection, weights = robot.get_measurements(beacons, distance_noise)
                 
                # Добавляем измерение и позицию в буфер
                projection_buffer.append(projection[0])
                position_buffer.append(np.array([robot.x_ssp, robot.y_ssp, robot.z]))
                
                if len(projection_buffer) < virtual_buffer_size:
                    #estimated_positions.append(current_pos[:2].copy())
                    if course_changes and current_time in course_changes:
                        new_heading = course_changes[current_time]
                    robot.update_position(dt, new_heading)
                    if i % ssp_period_index == 0:
                        robot.update_ssp(ssp_period)

                    continue

                # Для каждого измерения в буфере вычисляем виртуальное положение маяка
                virtual_beacon_positions = np.zeros((virtual_buffer_size, 3))
                for i in range(virtual_buffer_size):
                    # Определяем смещение от момента измерения до текущего момента
                    #delta = np.array([robot.x, robot.y, robot.z]) - position_buffer[i]
                    delta = np.array([robot.x_ssp, robot.y_ssp, robot.z]) - position_buffer[i]
                    # Виртуальное положение маяка: физический маяк + смещение
                    virtual_beacon = beacons[0] + delta
                    virtual_beacon_positions[i] = virtual_beacon
               
                # current_estimate = trilaterate(virtual_beacon_positions[:, :2], np.array(projection_buffer), weights)
                if k > 0:
                    current_estimate = trilaterate(virtual_beacon_positions[:, :2], np.array(projection_buffer), weights, estimated_positions[k-1])
                    epsilon_error = np.linalg.norm(estimated_positions[k-1] - current_estimate)
                else:
                   current_estimate = trilaterate(virtual_beacon_positions[:, :2], np.array(projection_buffer), weights, [x0, y0])
                   epsilon_error = np.linalg.norm([x0, y0] - current_estimate)

                position_buffer.clear()
                projection_buffer.clear()
                
                if epsilon_error < period_steps:
                    estimated_clean_positions.append(current_estimate)
                    true_clean_positions.append([robot.x, robot.y])
                    clean_errors.append([k, epsilon_error])
      
                errors.append([k, np.linalg.norm([robot.x, robot.y] - current_estimate)])
                true_positions.append([robot.x, robot.y])
                estimated_positions.append(current_estimate)

                robot.update_gans(current_estimate[0], current_estimate[1])

                k = k + 1

        
        # Применение циклограммы курса
        if course_changes and current_time in course_changes:
            new_heading = course_changes[current_time]
        robot.update_position(dt, new_heading)

        if i % ssp_period_index == 0:
            robot.update_ssp(ssp_period)

    return (
        trajectory,
        ssp_trajectory,
        gans_trajectory,
        np.array(true_positions),
        np.array(estimated_positions),
        np.array(estimated_clean_positions),
        period_steps,
        np.array(clean_errors),
        np.array(errors)
    )
