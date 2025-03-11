import numpy as np
import matplotlib.pyplot as plt

def advection_solver(grid_size, scheme='UDS', u=1.0, v=2.0, phi_left=1.0, phi_bottom=2.0, tol=1e-6, max_iter=1000):
    Lx, Ly = 1.0, 1.0  # Dimensiones del dominio
    dx, dy = Lx / grid_size, Ly / grid_size
    
    # Inicializar phi
    phi = np.zeros((grid_size + 2, grid_size + 2))
    phi[:, 0] = phi_left  # Condición en x=0 (izquierda)
    phi[0, :] = phi_bottom  # Condición en y=0 (abajo)
    
    # Iteración
    for _ in range(max_iter):
        phi_old = phi.copy()
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                # Coeficientes según el esquema
                if scheme == 'CDS':
                    a_W = u / (2 * dx)
                    a_E = -u / (2 * dx)
                    a_S = v / (2 * dy)
                    a_N = -v / (2 * dy)
                    a_P = -(a_W + a_E + a_S + a_N)
                elif scheme == 'UDS':
                    a_W = u / dx if u > 0 else 0
                    a_E = -u / dx if u < 0 else 0
                    a_S = v / dy if v > 0 else 0
                    a_N = -v / dy if v < 0 else 0
                    a_P = -(a_W + a_E + a_S + a_N)
                
                # Actualización de phi en el nodo P
                if a_P != 0:  # Evitar divisiones por cero
                    phi[i, j] = (a_W * phi[i, j-1] + a_E * phi[i, j+1] +
                                 a_S * phi[i-1, j] + a_N * phi[i+1, j]) / -a_P
        
        # Convergencia
        if np.linalg.norm(phi - phi_old, ord=np.inf) < tol:
            break
    
    return phi

# Parámetros y resolución
grid_sizes = [10, 50, 100]
schemes = ['UDS']
diagonal_results = []

# Resolver para cada malla
for grid_size in grid_sizes:
    phi = advection_solver(grid_size, scheme='UDS')
    x = np.linspace(0, 1, grid_size + 2)
    y = np.linspace(0, 1, grid_size + 2)
    X, Y = np.meshgrid(x, y)
    
    # Guardar valores de la diagonal para comparación
    diag_phi = np.array([phi[i, grid_size - i] for i in range(1, grid_size + 1)])  # Diagonal desde (0,1) hasta (1,0)
    diag_dist = np.linspace(0, np.sqrt(2), len(diag_phi))  # Longitud de la diagonal
    diagonal_results.append((diag_dist, diag_phi, grid_size))
    
    # Graficar solo contornos para mallas 10x10 y 50x50
    if grid_size in [10, 50]:
        plt.figure()
        plt.contourf(X, Y, phi, levels=50, cmap='jet')
        plt.colorbar(label='Φ')
        plt.title(f'Φ malha{grid_size}x{grid_size}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Graficar comparación de Φ a lo largo de la diagonal para todas las mallas
plt.figure()
for diag_dist, diag_phi, grid_size in diagonal_results:
    plt.plot(diag_dist, diag_phi, label=f'Malha {grid_size}x{grid_size}')
    
plt.title('Φ (UDS)')
plt.xlabel('Diagonal [m]')
plt.ylabel('Φ')
plt.legend()
plt.grid()
plt.show()
