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

# Función para ejecutar el problema para un caso dado de ratio dx/dy

    
dx = 1.0 / (Nx - 1)
dy = 1.0 / (Ny - 1)
a, b = 1.0, 1.0  
phi, x, y = inicializar_phi(a, b, dx, dy)
phi_final, residuo_max_hist = gauss_seidel_2d(phi, dx, dy)


    
dx = 1.0 / (Nx - 1)
dy = dx / 10  # Mantener la proporción dx/dy = 10
a, b = 1.0, (Ny - 1) * dy
phi10, x, y = inicializar_phi(a, b, dx, dy)
phi_final10, residuo_max_hist10 = gauss_seidel_2d(phi10, dx, dy)

# Graficar la convergencia del residuo
plt.figure(figsize=(10, 6))
plt.plot(residuo_max_hist, label="Δx / Δy = 1")
plt.plot(residuo_max_hist10, label="Δx / Δy = 10")
plt.yscale("log")
plt.xlabel("Iter")
plt.ylabel("Residuo máximo")
#plt.title("Evolución de los residuos por iteración")
plt.legend()
plt.grid()
plt.show()


