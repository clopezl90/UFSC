import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
q_k = 5  # q''' / k [K/m]
L = 1.0  # Longitud del dominio [m]

# Función para resolver por volúmenes finitos
def solve_vf(dx):
    volumes = int(L / dx)  # Número de volúmenes
    x_vf = np.linspace(dx / 2, L - dx / 2, volumes)  # Centros de los volúmenes
    A_vf = np.zeros((volumes, volumes))  # Matriz del sistema
    b_vf = np.zeros(volumes)  # Vector independiente

    # Construcción de la matriz A y el vector b
    for i in range(volumes):
        if i == 0:  # Primer volumen (frontera izquierda)
            A_vf[i, i] = -2
            A_vf[i, i + 1] = 1
            b_vf[i] = -q_k * dx**2
        elif i == volumes - 1:  # Último volumen (frontera derecha)
            A_vf[i, i - 1] = 1
            A_vf[i, i] = -2
            b_vf[i] = -q_k * dx**2
        else:  # Volúmenes internos
            A_vf[i, i - 1] = 1
            A_vf[i, i] = -2
            A_vf[i, i + 1] = 1
            b_vf[i] = -q_k * dx**2

    # Resolviendo el sistema lineal
    T_vf = np.linalg.solve(A_vf, b_vf)
    return x_vf, T_vf

# Solución Analítica
def analytical_solution(x):
    return (q_k / 2) * x * (L - x)

# Rango continuo para solución analítica
x_analytical = np.linspace(0, L, 100)
T_analytical = analytical_solution(x_analytical)

# Graficar las soluciones para diferentes tamaños de malla
dx_values = [0.25, 0.1, 0.05]  # Tamaños de malla
plt.figure(figsize=(12, 8))

# Graficar solución analítica
plt.plot(x_analytical, T_analytical, label=" Analítica", color="red", linestyle="--")

# Graficar solución por volúmenes finitos para diferentes dx
for dx in dx_values:
    x_vf, T_vf = solve_vf(dx)
    plt.plot(x_vf, T_vf, label=f"Volumes finitos (dx = {dx})", marker="o", linestyle="-")

# Configuración de la gráfica

plt.xlabel("Espaço em x [m]")
plt.ylabel("Temperatura[K]")
plt.legend()
plt.grid()
plt.show()

