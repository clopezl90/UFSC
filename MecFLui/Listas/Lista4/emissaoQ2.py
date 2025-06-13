import numpy as np
import matplotlib.pyplot as plt

# Parámetros
t0 = 0
t1 = np.pi / 2
temissao = 2  # Instante da linha de emissão

# Domínio de y
y_vals = np.linspace(-10, 10, 400)

# ----------------------
# Linhas de corrente
# ----------------------
# Em t = 0: x = 8 cos(y/4) - 8
x_linha_c0 = 8 * np.cos(y_vals / 4) - 8

# Em t = pi/2: x = 2 sin(y/4)
x_linha_c1 = 2 * np.sin(y_vals / 4)

# ----------------------
# Trajetórias
# ----------------------
# Partícula no t = 0: x = 0, y = 4t
t_vals = np.linspace(0, 3, 100)
x_traj0 = np.zeros_like(t_vals)
y_traj0 = 4 * t_vals

# Partícula no t = pi/2: x = 2(t - pi/2), y = 4(t - pi/2)
x_traj1 = 2 * (t_vals - t1)
y_traj1 = 4 * (t_vals - t1)

# ----------------------
# Linha de emissão (em t = temissao)
# x = (y/2) * sin(t - y/4)
y_emissao = np.linspace(0, 10, 300)
x_emissao = (y_emissao / 2) * np.sin(temissao - y_emissao / 4)

# ----------------------
# Plot
# ----------------------
plt.figure(figsize=(10, 8))

# Linhas de corrente
plt.plot(x_linha_c0, y_vals, 'b-', label='Linha de corrente (t=0)')
plt.plot(x_linha_c1, y_vals, 'g--', label='Linha de corrente (t=π/2)')

# Trajetórias
plt.plot(x_traj0, y_traj0, 'r-', label='Trajetória (t=0)')
plt.plot(x_traj1, y_traj1, 'r--', label='Trajetória (t=π/2)')

# Linha de emissão
plt.plot(x_emissao, y_emissao, 'm-', label=f'Linha de emissão (t={temissao})')

# Estilo
plt.title("Linhas de Corrente, Trajetórias e Linha de Emissão")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()
