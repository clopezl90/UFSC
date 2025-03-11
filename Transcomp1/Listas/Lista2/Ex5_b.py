import numpy as np
import matplotlib.pyplot as plt


Nx, Ny = 20, 20  
max_iter = 10000  
tolerance = 1e-6  


def inicializar_phi(a, b, dx, dy):
    phi = np.zeros((Nx, Ny))  
    x = np.linspace(0, a, Nx)  
    y = np.linspace(0, b, Ny)  
    
    
    for i in range(Nx):
        phi[i, -1] = np.sin(np.pi * x[i] / a)
    
   
    for i in range(Nx):
        phi[i, 0] = 0  
    for j in range(Ny):
        phi[0, j] = 0  
        phi[-1, j] = 0  
    
    return phi, x, y


def tdma_solver(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

   
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denominator if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator

    
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def tdma_2d(phi, dx, dy, barrido="horizontal"):
    residuo_max_hist = []

    
    a = np.ones(Nx - 2) / dx**2  
    b = -2 * (1 / dx**2 + 1 / dy**2) * np.ones(Nx - 2)  
    c = np.ones(Nx - 2) / dx**2  

    for iteration in range(max_iter):
        phi_old = phi.copy()

        if barrido == "horizontal":
            
            for j in range(1, Ny - 1):
                d = -(phi[1:-1, j + 1] / dy**2 + phi[1:-1, j - 1] / dy**2)
                phi[1:-1, j] = tdma_solver(a, b, c, d)
            
            for i in range(1, Nx - 1):
                d = -(phi[i + 1, 1:-1] / dx**2 + phi[i - 1, 1:-1] / dx**2)
                phi[i, 1:-1] = tdma_solver(a, b, c, d)
        elif barrido == "vertical":
            
            for i in range(1, Nx - 1):
                d = -(phi[i + 1, 1:-1] / dx**2 + phi[i - 1, 1:-1] / dx**2)
                phi[i, 1:-1] = tdma_solver(a, b, c, d)
            
            for j in range(1, Ny - 1):
                d = -(phi[1:-1, j + 1] / dy**2 + phi[1:-1, j - 1] / dy**2)
                phi[1:-1, j] = tdma_solver(a, b, c, d)

        
        residuo = np.abs(phi - phi_old)
        residuo_max = np.max(residuo)
        residuo_max_hist.append(residuo_max)

        
        if residuo_max < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration + 1}")
            break
    else:
        print("Máximo de iteraciones alcanzado sin convergencia.")
    
    return phi, residuo_max_hist




a_1, b_1 = 1.0, 1.0
dx_1 = a_1 / (Nx - 1)
dy_1 = b_1 / (Ny - 1)
phi_case1, _, _ = inicializar_phi(a_1, b_1, dx_1, dy_1)
phi_final_case1, residuo_case1 = tdma_2d(phi_case1, dx_1, dy_1, barrido="horizontal")


phi_case2, _, _ = inicializar_phi(a_1, b_1, dx_1, dy_1)
phi_final_case2, residuo_case2 = tdma_2d(phi_case2, dx_1, dy_1, barrido="vertical")


a_10, b_10 = 1.0, 1.0 / 10
dx_10 = a_10 / (Nx - 1)
dy_10 = b_10 / (Ny - 1)
phi_case3, _, _ = inicializar_phi(a_10, b_10, dx_10, dy_10)
phi_final_case3, residuo_case3 = tdma_2d(phi_case3, dx_10, dy_10, barrido="horizontal")


phi_case4, _, _ = inicializar_phi(a_10, b_10, dx_10, dy_10)
phi_final_case4, residuo_case4 = tdma_2d(phi_case4, dx_10, dy_10, barrido="vertical")


plt.figure(figsize=(10, 6))
plt.plot(residuo_case1, label="Δx / Δy = 1, Varredura Horizontal")
plt.plot(residuo_case2, label="Δx / Δy = 1, Varredura Vertical")
plt.plot(residuo_case3, label="Δx / Δy = 10, Varredura Horizontal")
plt.plot(residuo_case4, label="Δx / Δy = 10, Varredura Vertical")
plt.yscale("log")
plt.xlabel("Iteracion")
plt.ylabel("Residuo Max")
#plt.title("Residuals Evolution (TDMA)")
plt.legend()
plt.grid()
plt.show()


def graficar_solucion(phi, x, y, titulo):
    plt.figure(figsize=(10, 6))
    plt.contourf(x, y, phi.T, levels=50, cmap='hot')
    plt.colorbar(label="φ")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(titulo)
    plt.show()

graficar_solucion(phi_final_case1, np.linspace(0, a_1, Nx), np.linspace(0, b_1, Ny), "TDMA Δx / Δy = 1, Horiz Sweep")
graficar_solucion(phi_final_case2, np.linspace(0, a_1, Nx), np.linspace(0, b_1, Ny), "TDMA Δx / Δy = 1, Vert Sweep")
graficar_solucion(phi_final_case3, np.linspace(0, a_10, Nx), np.linspace(0, b_10, Ny), "TDMA Δx / Δy = 10, Horiz Sweep")
graficar_solucion(phi_final_case4, np.linspace(0, a_10, Nx), np.linspace(0, b_10, Ny), "TDMA Δx / Δy = 10, Vert Sweep")
