
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, savgol_filter
from scipy.fft import fft, fftfreq

# Cargar archivo con datos reales
df = pd.read_csv("Re125.txt", sep='\t', header=None, names=['time', 'U'])

# Cálculos básicos
dt = df['time'].diff().mean()
U_mean = df['U'].mean()
u_fluct = df['U'] - U_mean
u_rms = u_fluct.std()
k = 1.5 * u_rms**2

# Autocorrelación normalizada
corr = correlate(u_fluct, u_fluct, mode='full') / len(u_fluct)
mid = len(corr) // 2
rho_pos = corr[mid:] / (u_rms**2)
lags_pos = np.arange(0, len(rho_pos)) * dt
Theta = np.trapz(rho_pos[rho_pos > 0], lags_pos[:len(rho_pos[rho_pos > 0])])
L = U_mean * Theta

# Número de Reynolds y disipación
nu = 1.5e-5
Re_L = u_rms * L / nu
epsilon = u_rms**3 / L

# Escalas de Kolmogorov
eta = (nu**3 / epsilon)**0.25
tau_eta = (nu / epsilon)**0.5
u_eta = (nu * epsilon)**0.25
tau_0 = L / u_rms

# FFT y espectro
u_fluct_array = u_fluct.to_numpy()
N = len(u_fluct_array)
fft_vals = fft(u_fluct_array)
freqs = fftfreq(N, dt)[:N//2]
E_f = np.abs(fft_vals[:N//2])**2
E_f_smooth = savgol_filter(E_f[1:], window_length=101, polyorder=3)

# Guardar gráficos
# Gráfico 1: Coeficiente de correlación temporal
plt.figure(figsize=(10, 5))
plt.plot(lags_pos[:len(rho_pos)], rho_pos, color='blue')
plt.xlabel('τ [s]')
plt.ylabel('ρ(τ)')
plt.title('Coeficiente de correlación temporal')
plt.grid(True)
plt.show()
plt.savefig("correlacion_temporal.png", dpi=300)
plt.close()

# Gráfico 2: Espectro de energía con pendiente -5/3
plt.figure(figsize=(10, 6))
plt.loglog(freqs[1:], E_f_smooth, color='purple', label='Espectro suavizado')

# Agregar regiones
plt.axvspan(freqs[1], 300, color='green', alpha=0.2, label='Região dissipativa')
plt.axvspan(300, 3000, color='orange', alpha=0.2, label='Região inercial')
plt.axvspan(3000, freqs[len(E_f_smooth)], color='red', alpha=0.2, label='Região de energia')

# Línea de referencia con pendiente -5/3
f_ref_start = 300
f_ref_end = 3000
f_anchor = 1000
# índice más cercano a f_anchor
i_anchor = np.argmin(np.abs(freqs[1:] - f_anchor))
E_anchor = E_f_smooth[i_anchor]

# Construir la línea -5/3
f_line = np.linspace(f_ref_start, f_ref_end, 100)
slope = -5/3
E_line = E_anchor * (f_line / f_anchor)**slope
plt.loglog(f_line, E_line, 'w--', linewidth=2, label='Pendiente -5/3')

# Etiquetas y visualización
plt.xlabel('Freqüência [Hz]')
plt.ylabel('Densidade espectral de energia [u²/Hz]')
plt.title('Espectro de Energia Turbulenta com Pendiente -5/3')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("espectro_energia_com_pendiente.png", dpi=300)
plt.show()
plt.close()

