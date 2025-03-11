import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Tb = 373  # Temperatura en la base [K]
Tinf = 293  # Temperatura ambiente [K]
k = 10  # Conductividad térmica [W/mK]
alfa = 1e-6  # Difusividad térmica [m^2/s]
h = 5  # Coeficiente de convección [W/m^2K]
D = 0.01  # Diámetro de la aleta [m]
L = 0.05  # Longitud de la aleta [m]

# Configuración del problema
nmalha = 5  # Número de volúmenes
deltaz = L / nmalha  # Tamaño del elemento de malla
deltat = 0.1  # Paso temporal [s]
tempofinal = 16  # Tiempo final [s]
ntempofinal = int(tempofinal / deltat)  # Número de pasos temporales
temposparada = [0.5,1,2.5,5,10,15]  # Tiempos para graficar

# Coeficiente S
S = -4 * h / (D * k)

# Inicialización de temperaturas
T = np.ones((nmalha, 2)) * Tinf  # Temperaturas iniciales (todo el dominio a Tinf)

# Inicialización de coeficientes
aw = np.zeros(nmalha)
ae = np.zeros(nmalha)
ap = np.zeros(nmalha)
bp = np.zeros(nmalha)

# Evolución temporal
key = 0  # Control para tiempos de parada
temps_plot = []  # Almacenar temperaturas en tiempos específicos

for n in range(ntempofinal):  # Iteración en el tiempo
    for _ in range(100):  # Iteración para convergencia en cada paso de tiempo
        # Coeficientes en la base
        aw[0] = 0
        ae[0] = 1 / deltaz
        ap0 = (1 / alfa) * (deltaz / deltat)
        ap[0] = ap0 + ae[0] + 2 / deltaz - S * deltaz
        bp[0] = ap0 * T[0, 0] + (2 * Tb / deltaz) - S * deltaz * Tinf

        # Coeficientes en el interior
        for i in range(1, nmalha - 1):
            ap0 = (1 / alfa) * (deltaz / deltat)
            ae[i] = 1 / deltaz
            aw[i] = 1 / deltaz
            ap[i] = ap0 + ae[i] + aw[i] - S * deltaz
            bp[i] = ap0 * T[i, 0] - S * deltaz * Tinf

        # Coeficientes en la punta
        ae[-1] = 0
        aw[-1] = 1 / deltaz
        ap0 = (1 / alfa) * (deltaz / deltat)
        ap[-1] = ap0 + aw[-1] - S * deltaz
        bp[-1] = ap0 * T[-1, 0] - S * deltaz * Tinf

        # Solución por barrido hacia adelante y hacia atrás
        P = np.zeros(nmalha)
        Q = np.zeros(nmalha)

        # Barrido hacia adelante
        P[0] = ae[0] / ap[0]
        Q[0] = bp[0] / ap[0]
        for i in range(1, nmalha):
            P[i] = ae[i] / (ap[i] - aw[i] * P[i - 1])
            Q[i] = (bp[i] + aw[i] * Q[i - 1]) / (ap[i] - aw[i] * P[i - 1])

        # Barrido hacia atrás
        T[-1, 1] = Q[-1]
        for i in range(nmalha - 2, -1, -1):
            T[i, 1] = P[i] * T[i + 1, 1] + Q[i]

        # Calcular el error
        erro = np.max(np.abs(T[:, 1] - T[:, 0]))
        T[:, 0] = T[:, 1].copy()  # Actualizar temperaturas
        if erro < 1e-5:
            break

    # Guardar resultados en tiempos específicos
    if n * deltat >= temposparada[key]:
        temps_plot.append(T[:, 1].copy())
        key += 1
        if key >= len(temposparada):
            break

# Generar las posiciones espaciales de la malla
x = np.linspace(deltaz / 2, L - deltaz / 2, nmalha)  # Coordenadas espaciales de la malla

# Solución analítica para la aleta
def analytical_solution(x, Tb, Tinf, h, k, L):
    m = np.sqrt(4 * h / (k * D))  # Parámetro m
    theta_b = Tb - Tinf
    C1 = np.cosh(m * L)
    C2 = np.sinh(m * L)
    return Tinf + theta_b * (np.cosh(m * (L - x)) + (h / (m * k)) * np.sinh(m * (L - x))) / (C1 + (h / (m * k)) * C2)

# Generar puntos para la solución analítica
x_analytical = np.linspace(0, L, 100)  # Puntos de la longitud de la aleta
T_analytical = analytical_solution(x_analytical, Tb, Tinf, h, k, L)

# Adimensionalizar resultados
theta_plot = [(temp - Tinf) / (Tb - Tinf) for temp in temps_plot]  # Temperatura adimensional
theta_analytical = (T_analytical - Tinf) / (Tb - Tinf)  # Solución analítica adimensional
x_adim = x / L  # Posición adimensional para la solución numérica
x_analytical_adim = x_analytical / L  # Posición adimensional para la solución analítica

# Graficar las temperaturas adimensionales
plt.figure(figsize=(10, 6))

# Graficar solución numérica adimensional para tiempos específicos
for i, theta in enumerate(theta_plot):
    plt.plot(x_adim, theta, label=f"t = {temposparada[i]:.2f} s")

# Graficar solución analítica adimensional
plt.plot(x_analytical_adim, theta_analytical, label=" Analítica", color="black", linestyle="--", linewidth=2)

# Configuración de la gráfica

plt.xlabel("Espaço Adimensional ($x^*$)")
plt.ylabel("Temperatura Adimensional ($\\theta$)")
plt.legend()
plt.grid()
plt.show()
