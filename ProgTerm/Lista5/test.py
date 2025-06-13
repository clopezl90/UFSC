import matplotlib.pyplot as plt
import numpy as np

x_min = -5 # [m]
x_max = 5 # [m]
y_min = -5 # [m]
y_max = 5 # [m]
N_I = 100
N_J = 100
xx = np.linspace(x_min, x_max, num=N_I)
yy = np.linspace(y_min, y_max, num=N_J)
X, Y = np.meshgrid(xx, yy)

a = 1
b=1
L = 1.0
U_inf = 1.0

Q = U_inf * a
K = a*(Q/L) / np.pi

def phi_line_source(X, Y, K=1, a=1, b=1):
    r2 = np.sqrt((X - a) ** 2 + (Y - b) ** 2)
    phi = K * np.log(r2)
    return phi
#Calculating phi field
phi_1 = phi_line_source(X, Y, K, -a, b)
phi_2 = phi_line_source(X, Y, -K, a, b)

phi_total = phi_1 + phi_2

# phi_fonte_plot=plt.contourf(X,Y,phi_fonte,cmap='jet')
# plt.colorbar(phi_fonte_plot,label='phi')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(6, 5))
# pcm = plt.pcolormesh(X, Y, phi_source, shading='auto', cmap='viridis')
# plt.colorbar(pcm, label='phi [m2/s]')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.title('Campo de potencial com pcolormesh')
# plt.axis('equal')
# plt.grid(True)
# plt.show()

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

#U = calculate_U_velocity(X, Y, phi_1)
#V = calculate_V_velocity(Y, Y, phi_1)

# U[:, :2] = np.nan
# U[:, -2:] = np.nan
# V[:2, :] = np.nan
# V[-2:, :] = np.nan

def phi_uniform(X,Y,V,alpha):
    a = V*(X*np.cos(alpha) + Y*np.sin(alpha))
    return a

phi_3 = phi_uniform(X, Y, V=1.0, alpha=np.pi/6)

# U = calculate_U_velocity(X, Y, phi_3)
# V = calculate_V_velocity(Y, Y, phi_3)
# U[:, :2] = np.nan
# U[:, -2:] = np.nan
# V[:2, :] = np.nan
# V[-2:, :] = np.nan
#
#
#
# cont = plt.contour(X, Y, phi_3, levels=50, cmap='jet')
# #plt.figure(figsize=(6,5))
# plt.quiver(X, Y, U, V)
# plt.colorbar(cont, label='phi [m2/s]')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.axis('equal')
# plt.grid(True)
# plt.show()

#Fluid over a cylinder

phi = phi_1 + phi_2 + phi_3
U = calculate_U_velocity(X, Y, phi)
V = calculate_V_velocity(Y, Y, phi)

R = np.sqrt(X**2 + Y**2)
phi[R < a] = np.nan
U[R < a] = np.nan; V[R < a] = np.nan

U[:, :2] = np.nan
U[:, -2:] = np.nan
V[:2, :] = np.nan
V[-2:, :] = np.nan

MAG_VEL = np.sqrt(U**2 + V**2)

# plt.streamplot(
# X, Y, U, V,
# color=MAG_VEL,
# linewidth=1,
# cmap='viridis'
# )
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.axis('equal')
# plt.grid(True)
# plt.show()


# cont = plt.contour(X, Y, phi, levels=50, cmap="viridis")
# plt.colorbar(cont, label="phi [m2/s]")
# plt.quiver(X, Y, U, V)
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.axis("equal")
#
# plt.grid(True)
# plt.show()

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