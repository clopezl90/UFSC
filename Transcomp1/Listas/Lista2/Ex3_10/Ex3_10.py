import numpy as np
import matplotlib.pyplot as plt

# Datos del problema
Tb = 373  # Temperatura en la base [K]
T_inf = 293  # Temperatura ambiente [K]
k = 10  # Conductividad térmica [W/mK]
D = 0.01  # Diámetro de la aleta [m]
L = 0.05  # Longitud de la aleta [m]
h = 5.0  # Coeficiente de convección [W/m^2K]
alpha = 1e-6  # Difusividad térmica [m^2/s]

# Geometría de la aleta
P = np.pi * D  # Perímetro [m]
A = (np.pi * D**2) / 4  # Área de la sección transversal [m^2]

# Discretización
Nx = 20  # Número de volúmenes
dx = L / Nx  # Tamaño del volumen
dt = 10  # Paso de tiempo [s]
Nt_min = int(60 / dt)  # Pasos de tiempo para 1 minuto
Nt_max = int(60 * 60 / dt)  # Pasos de tiempo para 60 minutos

# Coeficientes
Fo = alpha * dt / dx**2  # Número de Fourier
Bi = h * P / (k * A)  # Número de Biot

# Inicialización de temperaturas
T = np.ones(Nx + 1) * T_inf  # Inicialmente, la aleta está a T_inf
T[0] = Tb  # Condición de la base

# Matriz para almacenar soluciones
T_history_1min = np.zeros((Nt_min, Nx + 1))
T_history_60min = np.zeros((Nt_max, Nx + 1))

# Iteraciones para 1 minuto
for n in range(Nt_min):
    T_new = T.copy()
    for i in range(1, Nx):  # Para nodos internos
        T_new[i] = T[i] + Fo * (T[i+1] - 2*T[i] + T[i-1]) - dt * Bi * (T[i] - T_inf)
    T_new[-1] = T_new[-2]  # Condición de topo aislado
    T = T_new.copy()
    T_history_1min[n, :] = T

# Inicialización nuevamente para 60 minutos
T = np.ones(Nx + 1) * T_inf
T[0] = Tb

# Iteraciones para 60 minutos
for n in range(Nt_max):
    T_new = T.copy()
    for i in range(1, Nx):  # Para nodos internos
        T_new[i] = T[i] + Fo * (T[i+1] - 2*T[i] + T[i-1]) - dt * Bi * (T[i] - T_inf)
    T_new[-1] = T_new[-2]  # Condición de topo aislado
    T = T_new.copy()
    T_history_60min[n, :] = T

# Solución estacionaria analítica
x = np.linspace(0, L, Nx + 1)
m = np.sqrt((h * P) / (k * A))
theta_analytical = (np.cosh(m * (L - x)) + (h / (m * k)) * np.sinh(m * (L - x))) / \
                   (np.cosh(m * L) + (h / (m * k)) * np.sinh(m * L))

# Adimensionalización de las temperaturas
T_adimensional_1min = (T_history_1min[-1, :] - T_inf) / (Tb - T_inf)
T_adimensional_60min = (T_history_60min[-1, :] - T_inf) / (Tb - T_inf)
theta_analytical_adim = (theta_analytical - T_inf) / (Tb - T_inf)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(x, T_adimensional_1min, label="Solución Numérica (1 min)", color="blue")
plt.plot(x, T_adimensional_60min, label="Solución Numérica (60 min)", color="green")
plt.plot(x, theta_analytical, label="Solución Analítica (Estacionaria)", color="red", linestyle="--")
plt.title("Distribución de Temperatura Adimensional en la Aleta")
plt.xlabel("Posición a lo largo de la aleta (x) [m]")
plt.ylabel("Temperatura Adimensional ($\\theta$)")
plt.legend()
plt.grid()
plt.show()
