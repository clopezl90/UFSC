import matplotlib.pyplot as plt
import numpy as np


def phi_line_source(X, Y, K=0.0, a=0.0, b=0.0):
    r2 = np.arctan2(Y - b, X - a)
    phi = K * r2
    return phi


def phi_uniform_flow(X, Y, V=0.0, alpha=0.0):
    return V * (Y * np.cos(alpha) - X * np.sin(alpha))


def calculate_U_velocity(X, Y, phi):
    dx = X[0, 1] - X[0, 0]
    phi_e = np.roll(phi, -1, axis=1)
    phi_w = np.roll(phi, 1, axis=1)
    return (phi_e - phi_w) / (2 * dx)


def calculate_V_velocity(X, Y, phi):
    dy = Y[1, 0] - Y[0, 0]
    phi_n = np.roll(phi, -1, axis=0)
    phi_s = np.roll(phi, 1, axis=0)
    return (phi_n - phi_s) / (2 * dy)


x_min = -5.0  # [m]
x_max = 5.0  # [m]
y_min = -5.0  # [m]
y_max = 5.0  # [m]

N_I = 120
N_J = 120

xx = np.linspace(x_min, x_max, num=N_I)
yy = np.linspace(y_min, y_max, num=N_J)
X, Y = np.meshgrid(xx, yy)

a = 1.0
L = 1.0
U_inf = 1.0

Q = U_inf * a
K = a * Q / (np.pi * L)

alpha = 0.0
psi_1 = phi_line_source(X, Y, K=K, a=-a, b=0.0)  # fonte
psi_2 = phi_line_source(X, Y, K=-K, a=a, b=0.0)  # sumidouro
psi_3 = phi_uniform_flow(X, Y, V=U_inf, alpha=alpha)

psi = psi_1 + psi_2 + psi_3

U = calculate_U_velocity(X, Y, psi)
V = calculate_V_velocity(Y, Y, psi)

U[:, :2] = np.nan
U[:, -2:] = np.nan
V[:2, :] = np.nan
V[-2:, :] = np.nan


plt.subplot(1, 2, 1)
cont = plt.contour(X, Y, psi, levels=50, cmap="viridis")
plt.colorbar(cont, label="phi [m2/s]")
plt.quiver(X, Y, U, V)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis("equal")

plt.grid(True)


plt.subplot(1, 2, 2)
MAG_VEL = np.sqrt(U**2 + V**2.0)
cont = plt.pcolormesh(X, Y, MAG_VEL)
plt.colorbar(cont, label="|U| [m/s]")
plt.title(f"U_inf = {MAG_VEL[-5, -5]:.2f} m/s")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis("equal")

plt.tight_layout()
plt.show()
