
import pandas as pd
import numpy as np
from scipy.signal import correlate, savgol_filter
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Cargar archivo CSV
df = pd.read_csv("data005.csv")
df.columns = ['time', 'U']

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
E_f_smooth = savgol_filter(E_f[1:], window_length=15, polyorder=3)

# Exportar a Excel
with pd.ExcelWriter("analisis_turbulencia_completo.xlsx", engine='xlsxwriter') as writer:
    # Hoja resumen
    pd.DataFrame({
        'Parámetro': ['Ū', "u'", 'k', 'Theta', 'L', 'Re_L', 'ε', 'η', 'τ_η', 'u_η', 'τ_0'],
        'Valor': [U_mean, u_rms, k, Theta, L, Re_L, epsilon, eta, tau_eta, u_eta, tau_0]
    }).to_excel(writer, sheet_name='Resumen', index=False)

    # Hoja señal
    pd.DataFrame({'t [s]': df['time'], 'U(t)': df['U'], "u(t)": u_fluct}).to_excel(writer, sheet_name='u(t)', index=False)

    # Hoja autocorrelación
    pd.DataFrame({'τ [s]': lags_pos[:len(rho_pos)], 'ρ(τ)': rho_pos}).to_excel(writer, sheet_name='Autocorrelacion', index=False)

    # Hoja espectro FFT
    pd.DataFrame({'Frecuencia [Hz]': freqs[1:], 'E(f)': E_f_smooth}).to_excel(writer, sheet_name='Espectro FFT', index=False)

print("Archivo Excel generado: analisis_turbulencia_completo.xlsx")
