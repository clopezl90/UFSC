import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos del archivo .txt
data = np.loadtxt('Re125.txt')  # asegúrate de tener el archivo en la misma carpeta

t = data[:, 0]  # tiempo
u = data[:, 1]  # velocidad

# Calcular la velocidad media
u_mean = np.mean(u)

# Fluctuaciones de velocidad
u_prime = u - u_mean

# Varianza (media de u'^2)
u_prime2_mean = np.mean(u_prime ** 2)

# Energía cinética turbulenta con isotropía
k = (3/2) * u_prime2_mean

print(f"Velocidad media ū = {u_mean:.4f} m/s")
print(f"Varianza u'² = {u_prime2_mean:.4f} (m/s)^2")
print(f"Energía cinética turbulenta k = {k:.4f} m²/s²")

# --------------------------------------------
# 2.ii - Coeficiente de correlação temporal e escala L
# --------------------------------------------

# Intervalo de tempo entre amostras
dt = t[1] - t[0]

# Fluctuación de velocidad ya calculada: u_prime
# Autocorrelación normalizada (solo la mitad positiva)
def autocorrelacion(u_prime):
    R = np.correlate(u_prime, u_prime, mode='full')
    R = R[R.size//2:]
    R /= R[0]  # normalización
    return R

R_tau = autocorrelacion(u_prime)
tau = np.arange(0, len(R_tau)) * dt  # retardo temporal

# Escala L = ∫ R(tau) d(tau) * ū hasta el primer cero
idx_cut = np.where(R_tau < 0)[0][0]
L = np.trapz(R_tau[:idx_cut], tau[:idx_cut]) * u_mean

print(f"Escala de comprimento das grandes escalas L = {L:.6f} m")

# Gráfico del coeficiente de correlación temporal
plt.figure(figsize=(8,4))
plt.plot(tau, R_tau, label='R(τ)')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Retardo temporal τ [s]')
plt.ylabel('R(τ)')
plt.title('Coeficiente de correlação temporal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------
# 2.iii - Número de Reynolds de las grandes escalas y escalas de Kolmogorov
# --------------------------------------------

# Viscosidad cinemática (aire a 20 °C)
nu = 1.5e-5  # m²/s

# Desviación estándar de u' (ya que u_prime tiene media cero)
u_rms = np.std(u_prime)

# Parámetros grandes
nu = 1.5e-5  # viscosidad cinemática [m^2/s]
u_0 = u_rms  # velocidad característica
l_0 = L      # escala de longitud integral
tau_0 = l_0 / u_0

# Número de Reynolds de las grandes escalas
Re_L = (u_0 * l_0) / nu

# Disipación turbulenta (por definición de escala grande)
epsilon = (u_0**3) / l_0

# Escalas de Kolmogorov usando relaciones con Re_L
eta = l_0 * Re_L**(-3/4)
u_eta = u_0 * Re_L**(-1/4)
tau_eta = tau_0 * Re_L**(-1/2)

# Mostrar resultados
print(u_0)
print(l_0)
print(tau_0)

print(f"Re_L = {Re_L:.2f}")
print(f"ε = {epsilon:.3e} m²/s³")
print(f"η = {eta:.3e} m")
print(f"u_η = {u_eta:.3e} m/s")
print(f"τ_η = {tau_eta:.3e} s")
