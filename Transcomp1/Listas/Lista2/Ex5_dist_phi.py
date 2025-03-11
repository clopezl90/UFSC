import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Nx, Ny = 20, 20  # Número de nodos en x e y
a, b = 1.0, 1.0  # Dimensiones del dominio (Δx / Δy = 1 implica a = b)
dx = a / (Nx - 1)  # Tamaño del paso en x
dy = b / (Ny - 1)  # Tamaño del paso en y
max_iter = 10000  # Máximo de iteraciones
tolerance = 1e-6  # Tolerancia para la convergencia

# Función para inicializar phi con condiciones de contorno
def inicializar_phi(a, b, dx, dy):
    phi = np.zeros((Nx, Ny))  # Crear matriz de ceros para phi
    x = np.linspace(0, a, Nx)  # Coordenadas x de la malla
    y = np.linspace(0, b, Ny)  # Coordenadas y de la malla
    
    # Condición de contorno superior (φ(x, b) = sin(πx/a))
    for i in range(Nx):
        phi[i, -1] = np.sin(np.pi * x[i] / a)
    
    # Condiciones de contorno inferior, izquierda y derecha (φ = 0)
    # Nota: ya está inicializado como 0, explícito aquí para claridad.
    for i in range(Nx):
        phi[i, 0] = 0  # Borde inferior
    for j in range(Ny):
        phi[0, j] = 0  # Borde izquierdo
        phi[-1, j] = 0  # Borde derecho
    
    return phi, x, y

# Método de Gauss-Seidel en 2D
def gauss_seidel_2d(phi, dx, dy):
    residuo_max_hist = []  # Para almacenar el máximo residuo en cada iteración
    
    for iteration in range(max_iter):
        phi_old = phi.copy()
        
        # Actualizar valores internos usando Gauss-Seidel
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                phi[i, j] = ((phi[i+1, j] + phi[i-1, j]) / dx**2 + 
                             (phi[i, j+1] + phi[i, j-1]) / dy**2) / (2 / dx**2 + 2 / dy**2)
        
        # Calcular el residuo máximo
        residuo = np.abs(phi - phi_old)
        residuo_max = np.max(residuo)
        residuo_max_hist.append(residuo_max)

        # Verificar convergencia
        if residuo_max < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration + 1}")
            break
    else:
        print("Máximo de iteraciones alcanzado sin convergencia.")
    
    return phi, residuo_max_hist

# Resolver el caso Δx / Δy = 1
phi, x, y = inicializar_phi(a, b, dx, dy)
phi_final, residuo_max_hist = gauss_seidel_2d(phi, dx, dy)

# Graficar la solución final como mapa de contorno
plt.figure(figsize=(10, 6))
plt.contourf(x, y, phi_final.T, levels=50, cmap='hot')  # .T para transponer la matriz
plt.colorbar(label="φ")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contornos de φ para Δx / Δy = 1")
plt.show()

# Graficar la convergencia del residuo
plt.figure(figsize=(10, 6))
plt.plot(residuo_max_hist, label="Residuo máximo por iteración")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Residuo máximo")
plt.title("Convergencia del residuo máximo para Δx / Δy = 1")
plt.legend()
plt.grid()
plt.show()
