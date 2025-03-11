import numpy as np
import pandas as pd

# Dimensiones de la malla
nx, ny = 5, 5
dx, dy = 1.0, 1.0  # Tamaño del volumen de control
rho = 1.0          # Densidad
u, v = 1000, 1.0     # Velocidades en x e y
dt = 0.1           # Paso de tiempo

# Parámetros difusivos y fuente
Gamma_phi = 1.0  # Coeficiente difusivo
S_u = 1.0        # Término fuente independiente
S_p = 1.0        # Término fuente dependiente

# Inicializar valores iniciales de phi_P^n
phi_initial = 1.0  # Valor constante inicial para toda la malla
phi_P_n = np.full((ny, nx), phi_initial)

def calculate_coefficients_quick(i, j, phi_P_n):
    """
    Calcula los coeficientes para una celda específica (i, j) en la malla,
    usando el esquema QUICK y el valor inicial de phi_P^n.
    """
    # Coeficientes difusivos
    D_x = Gamma_phi / dx
    D_y = Gamma_phi / dy

    # Coeficientes advectivos
    F_x = rho * u
    F_y = rho * v

    # Coeficientes en QUICK
    # Este
    A_E = D_x + (F_x / 8) * 3 if j < nx - 1 else D_x
    A_W = (F_x / 8) * -1 if j > 0 else 0  # Nodo aguas arriba
    # Norte y sur difusivos
    A_N = D_y + F_y if i > 0 else D_y
    A_S = D_y - F_y if i < ny - 1 else D_y

    # Coeficiente central
    A_P = (rho * dx * dy / dt) + A_E + A_W + A_N + A_S - (S_p * dx * dy)

    # Término fuente constante
    b_P = S_u * dx * dy + (rho * phi_P_n[i, j] * dx * dy / dt)

    return A_P, A_E, A_W, A_N, A_S, b_P

# Calcular los coeficientes para toda la malla con QUICK
coefficients_quick = []

for i in range(ny):  # Filas de la malla
    for j in range(nx):  # Columnas de la malla
        A_P, A_E, A_W, A_N, A_S, b_P = calculate_coefficients_quick(i, j, phi_P_n)
        coefficients_quick.append({
            "Celda": f"({i+1},{j+1})",
            "A_P": A_P,
            "A_E": A_E,
            "A_W": A_W,
            "A_N": A_N,
            "A_S": A_S,
            "b_P": b_P
        })

# Crear un DataFrame para visualizar los coeficientes
df_coefficients_quick = pd.DataFrame(coefficients_quick)

# Mostrar los coeficientes para la malla
print(df_coefficients_quick)



