import numpy as np
import matplotlib.pyplot as plt

def alpha_e(Pe):
    """
    Calcula el valor de \(\alpha_e\) según la ecuación (4.35).
    """
    return Pe**2 / (10 + 2 * Pe**2)

def beta_e(Pe):
    """
    Calcula el valor de \(\beta_e\) según la ecuación (4.36).
    """
    return (1 + 0.005 * Pe**2) / (1 + 0.05 * Pe**2)

def solve_wuds(u, Gamma=0.1, L=1.0, n=20):
    """
    Resuelve el problema de advección-difusión 1D usando el esquema WUDS.
    
    Parámetros:
        u (float): Velocidad del flujo.
        Gamma (float): Coeficiente difusivo.
        L (float): Longitud del dominio.
        n (int): Número de nodos.
        
    Retorna:
        x (numpy array): Coordenadas espaciales.
        phi (numpy array): Distribución de phi en el dominio.
    """
    dx = L / (n - 1)  # Tamaño del paso de la malla
    Pe = u * dx / Gamma  # Número de Peclet

    # Construcción de la matriz de coeficientes
    A = np.zeros((n, n))  # Matriz de coeficientes
    b = np.zeros(n)       # Vector independiente

    # Condiciones de contorno
    phi_0 = 0.0          # Valor en x = 0
    phi_L = 1.0          # Valor en x = L

    # Ensamblado de la matriz A y el vector b usando WUDS
    for i in range(n):
        if i == 0:
            # Nodo en x = 0 (Condición de frontera izquierda)
            A[i, i] = 1.0
            b[i] = phi_0
        elif i == n - 1:
            # Nodo en x = L (Condición de frontera derecha)
            A[i, i] = 1.0
            b[i] = phi_L
        else:
            # Nodos internos
            alpha = alpha_e(Pe)
            beta = beta_e(Pe)
            
            A_W = Gamma / dx + alpha * max(u, 0) / dx
            A_E = Gamma / dx - beta * min(u, 0) / dx
            A_P = -(A_W + A_E)
            A[i, i - 1] = A_W
            A[i, i] = A_P
            A[i, i + 1] = A_E

    # Resolución del sistema de ecuaciones
    phi = np.linalg.solve(A, b)

    # Coordenadas espaciales para graficar
    x = np.linspace(0, L, n)
    return x, phi

# Parámetros del problema
velocities = [0, 0.1, 1.0, 10.0, 20.0]  # Velocidades para analizar
mesh_sizes = [20, 50]          # Dos tamaños de malla diferentes

# Generar gráficos para cada tamaño de malla
for n in mesh_sizes:
    plt.figure(figsize=(10, 6))
    for u in velocities:
        x, phi = solve_wuds(u, n=n)
        label = f'u = {u:.1f}'  # Etiqueta con velocidad y tamaño de malla
        plt.plot(x, phi, marker='', linestyle='-', label=label)
    
    # Personalización de la gráfica
    plt.title(f'Phi vs X WUDS malha {n}', fontsize=14)
    plt.xlabel(r'Espaco $x$', fontsize=12)
    plt.ylabel(r'$\phi$', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
