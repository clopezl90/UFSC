import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Lx = 1.0  # Longitud en la dirección x
Ly = 1.0  # Longitud en la dirección y
Nx, Ny = 20, 20  # Número de nodos en x e y
max_iter = 10000  # Máximo de iteraciones para el método de Gauss-Seidel
tolerance = 1e-6  # Tolerancia para la convergencia

# Calcular los pasos de malla para cada caso
def calcular_deltas(ratio):
    if ratio == 1:
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
    elif ratio == 10:
        dx = Lx / (Nx - 1)
        dy = dx / 10  # Mantener la proporción dx/dy = 10
    return dx, dy

# Inicialización de phi con condiciones de contorno
def inicializar_phi(dx, dy):
    phi = np.zeros((Nx, Ny))
    
    # Condición de contorno en el borde superior
    x = np.linspace(0, Lx, Nx)
    phi[:, -1] = np.sin(np.pi * x / Lx)  # φ(x, b) = sin(πx/a)
    
    return phi

# Método de Gauss-Seidel
def gauss_seidel(phi, dx, dy):
    residuo_max_hist = []

    for iteration in range(max_iter):
        phi_old = phi.copy()
        
        # Aplicar Gauss-Seidel en los puntos internos
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                phi_E = phi[i+1, j]     # Nodo este
                phi_W = phi[i-1, j]     # Nodo oeste
                phi_N = phi[i, j+1]     # Nodo norte
                phi_S = phi[i, j-1]     # Nodo sur
                phi[i, j] = (phi_E / dx**2 + phi_W / dx**2 + phi_N / dy**2 + phi_S / dy**2) / (2 / dx**2 + 2 / dy**2)

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

# Función para ejecutar el cálculo para un caso dado de ratio dx/dy
def resolver_caso(ratio):
    dx, dy = calcular_deltas(ratio)
    phi = inicializar_phi(dx, dy)
    phi_final, residuo_max_hist = gauss_seidel(phi, dx, dy)

    # Graficar la convergencia del residuo
    plt.figure()
    plt.plot(residuo_max_hist, label="Residuo máximo por iteración")
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Residuo máximo")
    plt.title(f"Convergencia del residuo máximo para Δx/Δy = {ratio}")
    plt.legend()
    plt.grid()
    plt.show()

    # Graficar la solución final
    plt.figure()
    plt.imshow(phi_final, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
    plt.colorbar(label="φ")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Distribución de φ para Δx/Δy = {ratio}")
    plt.show()

# Ejecutar para los dos casos
resolver_caso(1)    # Caso Δx/Δy = 1
resolver_caso(10)   # Caso Δx/Δy = 10
