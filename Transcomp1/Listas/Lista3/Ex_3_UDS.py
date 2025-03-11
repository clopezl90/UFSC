import numpy as np
import matplotlib.pyplot as plt

grid_size = 50
Lx = 1.0 
Ly =  1.0 
dx = Lx / grid_size
dy = Ly / grid_size
u = 1.0  
v = 1.0
phi_left = 1.0  
phi_bottom = 2.0 
res = 1e-6  
max_iter = 1000 
dt = 0.01  

def cds():
    phi = np.zeros((grid_size + 2, grid_size + 2))
    phi[:, 0] = phi_left  # BC botom
    phi[0, :] = phi_bottom  # BC left

    for _ in range(max_iter):
        phi_old = phi.copy()  
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):                
                a_W = u / dx if u > 0 else 0
                a_E = -u / dx if u < 0 else 0
                a_S = v / dy if v > 0 else 0
                a_N = -v / dy if v < 0 else 0
                a_P = -(a_W + a_E + a_S + a_N)
                
                
                if a_P != 0:  
                    phi[i, j] = (a_W * phi[i, j-1] + a_E * phi[i, j+1] +
                                 a_S * phi[i-1, j] + a_N * phi[i+1, j]) / -a_P

        
        diff = np.abs(phi - phi_old)  
        if np.max(diff) < res:  
            break


    return phi

# Función para extraer los valores de phi a lo largo de la diagonal
def extract_diagonal(phi, grid_size):
    diagonal_points = np.linspace(0, 1, grid_size + 2)  # Coordenadas a lo largo de la diagonal
    diagonal_phi = []
    
    for xi in diagonal_points:
        x_idx = xi / dx
        y_idx = (1 - xi) / dy
        
        i = int(np.floor(x_idx))
        j = int(np.floor(y_idx))
        
        # Interpolación bilineal
        phi_interp = (
            phi[i, j] * (1 - (x_idx - i)) * (1 - (y_idx - j)) +
            phi[i+1, j] * (x_idx - i) * (1 - (y_idx - j)) +
            phi[i, j+1] * (1 - (x_idx - i)) * (y_idx - j) +
            phi[i+1, j+1] * (x_idx - i) * (y_idx - j)
        )
        
        diagonal_phi.append(phi_interp)
    
    return np.array(diagonal_phi)










# Graficar los resultados
phi = cds()
x = np.linspace(0, 1, grid_size + 2)
y = np.linspace(0, 1, grid_size + 2)
X, Y = np.meshgrid(x, y)

# Graficar la solución
plt.figure()
plt.contourf(X, Y, phi, levels=50, cmap='jet')
plt.colorbar(label='Φ')
plt.title(f'CDS - malla {grid_size}x{grid_size}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Extraer y graficar Φ a lo largo de la diagonal
diagonal_phi = extract_diagonal(phi, grid_size)
diagonal_length = np.linspace(0, np.sqrt(2), len(diagonal_phi))  # Longitud de la diagonal
plt.figure()
plt.plot(diagonal_length, diagonal_phi, label=f'CDS {grid_size}x{grid_size}')
plt.title(f'Φ  (CDS)')
plt.xlabel('Diagonal [m]')
plt.ylabel('Φ')
plt.legend()
plt.grid()
plt.show()
