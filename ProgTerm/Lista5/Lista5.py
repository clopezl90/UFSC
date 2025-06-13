import matplotlib.pyplot as plt
import numpy as np

#Grid and domain size definition
x_min = -5 # [m]
x_max = 5 # [m]
y_min = -5 # [m]
y_max = 5 # [m]
N_I = 100
N_J = 100
xx = np.linspace(x_min, x_max, num=N_I)
yy = np.linspace(y_min, y_max, num=N_J)
X, Y = np.meshgrid(xx, yy)

#source variables definition
a = 1
b=1
L = 1.0
U_inf = 1.0
Q = U_inf * a
K = a*(Q/L) / 2*np.pi


def psi_line_source(X, Y, K=1, a=1, b=1):
    """
    This function plots calculates the psi function as
    indicated in equation 10-44
    :param X: x coordinate in grid
    :param Y: y coordinate in grid
    :param K: volume rate coeficient
    :param a: intensity location inx axis
    :param b:intensity location in y axis
    :return: psi (x,x)
    """
    r2 = np.arctan2(Y-b,X-a)
    psi = K * r2
    return psi

def calculate_U_velocity(X, Y, psi):
    '''
    This funcions calculates the u1 velocity
    :param X: x coordinate in grid
    :param Y: y coordinate in grid
    :param psi: psi(x,y)
    :return: spatial gradient of psi = velocity vector
    '''
    dx = X[0, 1] - X[0, 0]
    psi_e = np.roll(psi, -1, axis=1)
    psi_w = np.roll(psi, 1, axis=1)
    return (psi_e - psi_w) / (2 * dx)


def calculate_V_velocity(X, Y, psi):
    '''
    This funcions calculates the u1 velocity
    :param X: x coordinate in grid
    :param Y: y coordinate in grid
    :param psi: psi(x,y)
    :return: spatial gradient of psi = velocity vector
    '''

    dy = Y[1, 0] - Y[0, 0]
    psi_n = np.roll(psi, -1, axis=0)
    psi_s = np.roll(psi, 1, axis=0)
    return (psi_n - psi_s) / (2 * dy)

def phi_uniform(X,Y,V,alpha):
    '''
    This function calculates uniform flow with alpha inclination
    as indicated in eq 10-41
    :param X: x coordinate in grid
    :param Y: y coordinate in grid
    :param V: Velocity magnitude parallel to psi
    :param alpha: psi field inclination over x axis
    :return: uniform stream with alpha angle
    '''
    return V * (Y * np.cos(alpha) - X * np.sin(alpha))

#source, sink and uniform fields
alpha = np.pi/4
psi_1 = psi_line_source(X, Y, K=K, a=-a, b=0.0)  # source
psi_2 = psi_line_source(X, Y, K=-K, a=a, b=0.0)  # sink
psi_3 = phi_uniform(X, Y, V=U_inf, alpha=alpha)

#Total field
psi = psi_1 + psi_2 + psi_3

#velocity field calculation based in psi function
U = calculate_U_velocity(X, Y, psi)
V = calculate_V_velocity(X, Y, psi)

#Internal cylinder values confirgution for neglecting them
R = np.sqrt(X**2 + Y**2)
psi[R < a] = np.nan
U[R < a] = np.nan; V[R < a] = np.nan

#Neglecting values on domain limits
U[:, :2] = np.nan
U[:, -2:] = np.nan
V[:2, :] = np.nan
V[-2:, :] = np.nan

#Velocity field maginute calculation
MAG_VEL = np.sqrt(U**2 + V**2)

#### Q2 Current field distribution over cylinder
cont = plt.contour(X, Y, psi, levels=50, cmap="jet")
plt.colorbar(cont, label="psi [m2/s]")
plt.quiver(X, Y, U, V)
plt.title("Total current field over cylinder")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis("equal")
plt.grid(True)
plt.show()

#### Q4 Velocity field and current lines
plt.figure(figsize=(6, 5))
cont = plt.contourf(X, Y, MAG_VEL, levels=50, cmap='jet')
plt.colorbar(cont, label='|U| [m/s]')
plt.contour(X, Y, psi, levels=50, colors='k', linewidths=0.5)
plt.title("Velocity field and current lines")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

#### Q4 Pressure field and current lines
p_ref = 100
p = 0.5 * MAG_VEL**2 + p_ref
plt.figure(figsize=(6, 5))
cont = plt.contourf(X, Y, p, levels=50, cmap='jet')
plt.colorbar(cont, label='p [Pa]')
plt.contour(X, Y, psi, levels=50, colors='k', linewidths=0.5)
plt.title("Pressure field and current lines")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()



##### Q5 pressure distribution over cylinder surface#################
# Neglecting cylinder borders

# pressure contour

tol = 1e-2
mask_cilindro = np.abs(psi) < tol

X_cil = X[mask_cilindro]
Y_cil = Y[mask_cilindro]
P_cil = p[mask_cilindro]

# Angle calculation 0, 2 pi
theta = np.arctan2(Y_cil, X_cil)
theta = np.mod(theta, 2 * np.pi)

# Ordering
sorted_idx = np.argsort(theta)
X_cil = X_cil[sorted_idx]
Y_cil = Y_cil[sorted_idx]
P_cil = P_cil[sorted_idx]
theta = theta[sorted_idx]

# Plot
plt.figure(figsize=(6, 6))
sc = plt.scatter(X_cil, Y_cil, c=P_cil, cmap='jet', s=40)
plt.colorbar(sc, label='Pressure [Pa]')
plt.title("Pressure distribution over cylinder surface")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.show()