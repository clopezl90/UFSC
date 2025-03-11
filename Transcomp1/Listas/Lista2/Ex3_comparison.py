import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
L = 1.0  # Longitud del dominio
max_iter = 10000  # Máximo de iteraciones para los métodos
tolerance = 1e-6  # Tolerancia para la convergencia

# Función para inicializar la distribución de temperatura
def inicializar_temperatura(N):
    x = np.linspace(0, L, N)
    T = np.sin(np.pi * x) + 0.1 * np.sin(N * np.pi * x)
    T[0], T[-1] = 0, 100  # Condiciones de contorno: T(0) = 0 y T(L) = 100
    return T, x

# Método iterativo (Jacobi o Gauss-Seidel)
def resolver_diffusion(T, N, metodo="Gauss-Seidel"):
    residuo_max_hist = []

    for iteration in range(max_iter):
        T_old = T.copy()
        
        if metodo == "Gauss-Seidel":
            # Gauss-Seidel: usa valores actualizados de inmediato
            for i in range(1, N-1):
                T[i] = 0.5 * (T_old[i+1] + T[i-1])
        
        elif metodo == "Jacobi":
            # Jacobi: calcula los nuevos valores usando solo los de la iteración anterior
            for i in range(1, N-1):
                T[i] = 0.5 * (T_old[i+1] + T_old[i-1])
        
        # Calcular el residuo máximo para verificar convergencia
        residuo = np.abs(T - T_old)
        residuo_max = np.max(residuo)
        residuo_max_hist.append(residuo_max)

        # Verificar convergencia
        if residuo_max < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration + 1} usando {metodo}")
            break
    else:
        print("Máximo de iteraciones alcanzado sin convergencia.")

    return T, residuo_max_hist

# Función para ejecutar y graficar el resultado para un método específico
def ejecutar_metodo(N, metodo="Gauss-Seidel"):
    T, x = inicializar_temperatura(N)
    T_solucion, residuo_max_hist = resolver_diffusion(T, N, metodo)

    # Graficar la convergencia del residuo
    plt.figure(figsize=(10, 6))
    plt.plot(residuo_max_hist, label="Residuo máximo por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Residuo máximo")
    plt.yscale("log")
    plt.title(f"Convergencia del residuo máximo usando {metodo} con N = {N}")
    plt.legend()
    plt.grid()
    plt.show()

    # Graficar la distribución de temperatura final
    plt.figure(figsize=(10, 6))
    plt.plot(x, T_solucion, label="Distribución final de temperatura")
    plt.xlabel("x")
    plt.ylabel("Temperatura T(x)")
    plt.title(f"Distribución de temperatura usando {metodo} con N = {N}")
    plt.legend()
    plt.grid()
    plt.show()

# Ejecutar el código con ambos métodos para un valor de N
N = 31  # Número de puntos en la malla
ejecutar_metodo(N, metodo="Gauss-Seidel")
#ejecutar_metodo(N, metodo="Jacobi")
