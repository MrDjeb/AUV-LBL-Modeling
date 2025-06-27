flowchart TD
    A([Start]) --> B[Получить current_point из meander_points]
    B --> C[Вычислить distance = norm([xi_curr - current_point(1), zeta_curr - current_point(2)])]
    C --> D[Инициализировать new_target = current_target<br>mode = mode_curr]
    D --> E{mode_curr == 1?}

    E -- Да --> F{distance >= r_max?}
    F -- Да --> G[mode = 0]
    F -- Нет --> H[не меняем new_target и mode]
    G --> I
    H --> I

    E -- Нет --> J{distance <= r_min?}
    J -- Нет --> I
    J -- Да --> K{current_target < size(meander_points,1)?}
    K -- Да --> L[new_target = current_target + 1]
    K -- Нет --> M[mode = 1]
    L --> I
    M --> I

    I[Далее] --> N[Извлечь tgt = meander_points(new_target,:)]
    N --> O[xi_out = tgt(1)<br>zeta_out = tgt(2)]
    O --> P([End])
