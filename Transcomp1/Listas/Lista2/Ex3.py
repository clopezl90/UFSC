import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
L = 1.0  # Longitud del dominio
max_iter = 10000  # Máximo de iteraciones para el método de Gauss-Seidel
tolerance = 1e-6  # Tolerancia para la convergencia

def inicializar_temperatura(N):
    x = np.linspace(0, L, N)
    T = np.sin(np.pi * x) + 0.1 * np.sin(N * np.pi * x)
    T[0], T[-1] = 0, 100  # Condiciones de contorno: T(0) = 0 y T(L) = 100
    return T, x

def gauss_seidel(T, N, dx):
    residuo_max_hist = []  # Para almacenar el máximo residuo en cada iteración

    for iteration in range(max_iter):
        T_old = T.copy()
        
        # Aplicar Gauss-Seidel en los puntos internos
        for i in range(1, N-1):
            T[i] = 0.5 * (T[i+1] + T[i-1])

        # Calcular residuo en cada punto
        residuo = np.zeros(N)
        for i in range(1, N-1):
            residuo[i] = T[i+1] - 2 * T[i] + T[i-1]
        
        # Calcular el máximo residuo en esta iteración
        residuo_max = np.max(np.abs(residuo))
        residuo_max_hist.append(residuo_max)

        # Verificar convergencia
        if residuo_max < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration + 1}")
            break
    else:
        print("Máximo de iteraciones alcanzado sin convergencia.")
    
    return T, residuo_max_hist

def resolver_con_dual_temperatura(N):
    dx = L / (N - 1)  # Tamaño de los volúmenes
    T, x = inicializar_temperatura(N)
    
    # Graficar la distribución inicial
    plt.figure()
    plt.plot(x, T, label="Distribuição inicial")
    
    # Aplicar el método de Gauss-Seidel
    T_solucion, residuo_max_hist = gauss_seidel(T, N, dx)
    
    # Graficar la distribución final
    plt.plot(x, T_solucion, label="Distribuição final")
    plt.xlabel("Espaço x")
    plt.ylabel("Temperatura T")
    #plt.title(f"Distribución de temperatura con N = {N}")
    plt.legend()
    plt.show()
    
    # Graficar la convergencia del residuo
    plt.figure()
    plt.plot(residuo_max_hist, label="Residuo máximo por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Residuo máximo")
    plt.yscale("log")  # Escala logarítmica para ver mejor la disminución
    plt.title(f"Convergencia del residuo máximo con N = {N}")
    plt.legend()
    plt.show()

# Ejecución para diferentes valores de N
for N in [5, 31, 101]:
    resolver_con_dual_temperatura(N)
