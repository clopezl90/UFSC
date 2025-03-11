import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N =8  # Número de nodos internos
L = 1.0  # Longitud del dominio
dx = L / (N + 1)  # Tamaño de paso
max_iter = 1000  # Número máximo de iteraciones
tol = 1e-6  # Tolerancia de convergencia

# Inicializar malla
x = np.linspace(0, L, N + 2)
T = np.sin(np.pi * x) * (1 + 0.1 * (-1)**np.arange(N + 2))  # Condición inicial
T[0] = T[-1] = 0  # Condiciones de contorno

# Método de Gauss-Seidel con almacenamiento de iteraciones seleccionadas
def gauss_seidel_with_tracking(T, max_iter, tol, track_iterations):
    T_new = T.copy()
    snapshots = {}  # Almacenará soluciones en iteraciones seleccionadas
    for k in range(max_iter):
        T_old = T_new.copy()
        for i in range(1, N + 1):  # Actualizar nodos internos
            T_new[i] = 0.5 * (T_new[i - 1] + T_old[i + 1])
        # Almacenar soluciones en iteraciones seleccionadas
        if k in track_iterations:
            snapshots[k] = T_new.copy()
        # Verificar convergencia
        if np.linalg.norm(T_new - T_old, np.inf) < tol:
            print(f"Convergencia alcanzada en {k+1} iteraciones.")
            break
    #snapshots["final"] = T_new.copy()  # Almacenar solución final
    return T_new, k + 1, snapshots

# Iteraciones a graficar
track_iterations = [0, 1, 5, 20, 100]  # Iteraciones seleccionadas para seguimiento

# Solución
T_sol, num_iter, snapshots = gauss_seidel_with_tracking(T, max_iter, tol, track_iterations)

# Gráficas
plt.figure(figsize=(10, 6))

# Distribución inicial
plt.plot(x, T, label="Inicial", linestyle="--", marker="")

# Soluciones intermedias en iteraciones seleccionadas
for k, snapshot in snapshots.items():
    label = f"Iter {k}"
    plt.plot(x, snapshot, label=label)

#plt.title("Evolución de los errores de baja frecuencia (Gauss-Seidel)")
plt.xlabel("Espaço (x)")
plt.ylabel("Temperatura (T)")
plt.legend()
plt.grid()
plt.show()
 