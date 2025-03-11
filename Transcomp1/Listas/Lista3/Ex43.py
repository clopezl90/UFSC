import numpy as np
import matplotlib.pyplot as plt

def solve_cds(u, Gamma=0.1, L=1.0, n=20):
    """
    Resuelve el problema de advección-difusión 1D con esquema CDS
    para una velocidad dada (u).
    
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

    # Construcción de la matriz de coeficientes
    A = np.zeros((n, n))  # Matriz de coeficientes
    b = np.zeros(n)       # Vector independiente

    # Condiciones de contorno
    phi_0 = 0          # Valor en x = 0
    phi_L = 1.0           # Valor en x = L

    # Ensamblado de la matriz A y el vector b usando CDS
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
            A_W = Gamma / dx + u / 2
            A_E = Gamma / dx - u / 2
            A_P = -(A_W + A_E)
            A[i, i - 1] = A_W
            A[i, i] = A_P
            A[i, i + 1] = A_E

    # Resolución del sistema de ecuaciones
    phi = np.linalg.solve(A, b)

    # Coordenadas espaciales para graficar
    x = np.linspace(0, L, n)
    return x, phi

# Lista de velocidades para analizar
velocities = [0, 0.1, 1.0, 10.0, 20.0]

# Graficar resultados
plt.figure(figsize=(10, 6))
for u in velocities:
    x, phi = solve_cds(u)
    label = "u=0" if u == 0 else f'u = {u:.1f}'  # Etiqueta personalizada para u = 0
    plt.plot(x, phi, marker='', label=label)

# Personalizar la gráfica
plt.title(r'Phi vs X', fontsize=14)
plt.xlabel(r'Espaço[m]', fontsize=12)
plt.ylabel(r'$\phi$', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
