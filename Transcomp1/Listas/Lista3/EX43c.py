import numpy as np
import matplotlib.pyplot as plt

def phi_exact(x, Pe):
    """
    Solución exacta de la ecuación 4.22 para un Peclet dado.
    """
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def phi_4_27(x, Pe):
    """
    Calcula la solución usando la ecuación 4.27.
    """
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

# Parámetros del problema
Pe_values = [0.1, 1, 10]  # Diferentes valores del número de Peclet
mesh_sizes = [20, 50]    # Tres tamaños de malla

# Generar una gráfica para cada tamaño de malla
for n in mesh_sizes:
    x = np.linspace(0, 1, n)  # Dominio espacial con n puntos

    plt.figure(figsize=(10, 6))
    for Pe in Pe_values:
        # Solución exacta
        phi_exact_solution = phi_exact(x, Pe)
        # Solución obtenida con la ecuación 4.27
        phi_4_27_solution = phi_4_27(x, Pe)
        
        # Graficar ambas soluciones
        plt.plot(x, phi_exact_solution, linestyle='--', label=f'Exacta (Pe={Pe})')
        plt.plot(x, phi_4_27_solution, marker='o', linestyle='', label=f'Inter (Pe={Pe})')

    # Personalización de la gráfica
    plt.title(f'Phi vs X Malha {n} volumes ', fontsize=14)
    plt.xlabel(r'Espaço [m]', fontsize=12)
    plt.ylabel(r'$\phi$', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
