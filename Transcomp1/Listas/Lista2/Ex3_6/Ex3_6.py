import numpy as np
import matplotlib.pyplot as plt

# Parámetros
alpha = 1  # Difusividad térmica
dx = 1  # Paso espacial
dt_values = [0.25, 0.5, 0.75]  # Valores de Δt
x_points = 5  # Número de puntos en la malla (0, 1, 2, 3, 4)
time_steps = 6  # Número de intervalos de tiempo

# Condiciones iniciales y de frontera
initial_condition = np.zeros(x_points)
boundary_conditions = [1, 1]  # T(0, t) = 1, T(4, t) = 1

# Resolución numérica para cada valor de Δt
for dt in dt_values:
    lambda_value = alpha * dt / dx**2
    

    # Inicialización de la malla de temperaturas
    T = np.zeros((time_steps + 1, x_points))
    T[0, :] = initial_condition
    T[:, 0] = boundary_conditions[0]
    T[:, -1] = boundary_conditions[1]

    # Iteraciones temporales
    for n in range(time_steps):
        for i in range(1, x_points - 1):
            T[n + 1, i] = T[n, i] + lambda_value * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1])

    # Crear gráfico para el caso actual de Δt
    plt.figure(figsize=(8, 5))
    x = np.arange(0, x_points)
    for n in range(time_steps + 1):
        plt.plot(x, T[n, :], label=f"t = {n*dt:.2f}")

    # Configuración de la gráfica
    plt.title(f"Temperatura vs X Δt = {dt}")
    plt.xlabel("X")
    plt.ylabel("Temperatura (T)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.show()

