import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Nx, Ny = 20, 20  # Número de nodos en x e y
max_iter = 10000  # Máximo de iteraciones
tolerance = 1e-6  # Tolerancia para la convergencia

# Función para inicializar la malla con las condiciones de contorno
def inicializar_phi(a, b, dx, dy):
    phi = np.zeros((Nx, Ny))  # Crear matriz de ceros para phi
    x = np.linspace(0, a, Nx)  # Coordenadas x de la malla
    y = np.linspace(0, b, Ny)  # Coordenadas y de la malla
    
    # Condición de contorno superior (φ(x, b) = sin(πx/a))
    for i in range(Nx):
        phi[i, -1] = np.sin(np.pi * x[i] / a)
    
    # Condiciones de contorno inferior (φ(x, 0) = 0)
    for i in range(Nx):
        phi[i, 0] = 0  # Ya está inicializado en ceros, pero se deja explícito
    
    # Condiciones de contorno en los bordes izquierdo y derecho (φ(0, y) = 0 y φ(a, y) = 0)
    for j in range(Ny):
        phi[0, j] = 0   # Borde izquierdo
        phi[-1, j] = 0  # Borde derecho
    
    return phi, x, y


# Método de Gauss-Seidel en 2D
def gauss_seidel_2d(phi, dx, dy):
    residuo_max_hist = []
    
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

# Resolver el caso Δx / Δy = 10
a_10, b_10 = 1.0, 1.0 / 10  # Ajustar el tamaño del dominio para Δx / Δy = 10
dx_10 = a_10 / (Nx - 1)
dy_10 = b_10 / (Ny - 1)
phi_10, x_10, y_10 = inicializar_phi(a_10, b_10, dx_10, dy_10)
phi_final_10, residuo_max_hist_10 = gauss_seidel_2d(phi_10, dx_10, dy_10)

# Resolver el caso Δx / Δy = 20
a_20, b_20 = 1.0, 1.0 / 20  # Ajustar el tamaño del dominio para Δx / Δy = 20
dx_20 = a_20 / (Nx - 1)
dy_20 = b_20 / (Ny - 1)
phi_20, x_20, y_20 = inicializar_phi(a_20, b_20, dx_20, dy_20)
phi_final_20, residuo_max_hist_20 = gauss_seidel_2d(phi_20, dx_20, dy_20)

# Graficar la evolución de los residuos para ambos radios
plt.figure(figsize=(10, 6))
plt.plot(residuo_max_hist_10, label="Δx / Δy = 10")
plt.plot(residuo_max_hist_20, label="Δx / Δy = 20")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Residuo máximo")
plt.title("Evolución de los residuos por iteración")
plt.legend()
plt.grid()
plt.show()
