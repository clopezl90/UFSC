import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
L = 3.0  # Espesor de la placa (m)
k = 1.0  # Conductividad térmica (W/m·K)
q_gen = 7.0  # Generación uniforme de calor (W/m^3)
q_in = 10.0  # Flujo de calor entrando (W/m^2)
q_out = 21.0  # Flujo de calor saliendo (W/m^2)
n = 6  # Número de nodos

dx = L / (n + 1)
alpha = q_gen * dx**2 / k

# Crear matriz del sistema
A = np.zeros((n, n))
b = np.zeros(n)

for i in range(n):
    if i > 0:
        A[i, i - 1] = -1
    A[i, i] = 2
    if i < n - 1:
        A[i, i + 1] = -1

# Condiciones de borde
b[0] += q_in * dx / k
b[-1] += q_out * dx / k
b += alpha

# Métodos iterativos
def jacobi(A, b, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros_like(x)
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        for i in range(n):
            sum_ = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum_) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        x[:] = x_new

    return x, iter_count

def gauss_seidel(A, b, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        x_old = x.copy()
        for i in range(n):
            sum_ = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum_) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            break

    return x, iter_count

def sor(A, b, omega=1.5, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        x_old = x.copy()
        for i in range(n):
            sum_ = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sum_) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            break

    return x, iter_count

# Resolver usando los métodos
x_jacobi, iter_jacobi = jacobi(A, b)
x_gauss_seidel, iter_gauss_seidel = gauss_seidel(A, b)
x_sor, iter_sor = sor(A, b)

# Gráficas
x_positions = np.linspace(0, L, n)

plt.figure(figsize=(6, 5))

# Método Jacobi
#plt.subplot(3, 1, 1)
# plt.plot(x_positions, x_jacobi, 'o-b', label=f"Jacobi (Iter: {iter_jacobi})")

# plt.xlabel("Espaço (m)")
# plt.ylabel("Temperatura (K)")
# plt.legend()

# Método Gauss-Seidel

# plt.plot(x_positions, x_gauss_seidel, 's-g', label=f"Gauss-Seidel (Iter: {iter_gauss_seidel})")

# plt.xlabel("Espaço (m)")
# plt.ylabel("Temperatura (K)")
# plt.legend()

# # Método SOR
#plt.subplot(3, 1, 3)
plt.plot(x_positions, x_sor, '^-r', label=f"SOR (Iter: {iter_sor})")

plt.xlabel("Espaço (m)")
plt.ylabel("Temperatura (K)")
plt.legend()

#plt.tight_layout()
plt.show()

