import pandas as pd

# Parámetros comunes
rho = 1  # Densidad (unitaria)
cp = 1   # Calor específico (unitario)
k = 1    # Conductividad térmica (unitaria)
alpha = k / (rho * cp)  # Difusividad térmica

# Casos definidos en el problema
casos = [
    {"Delta_yP": 1, "Delta_ys": 10, "Delta_yn": 0.5, "Delta_t": 0.5},
    {"Delta_yP": 1, "Delta_ys": 10, "Delta_yn": 10, "Delta_t": 0.5},
    {"Delta_yP": 1, "Delta_ys": 10, "Delta_yn": 10, "Delta_t": 0.005},
    {"Delta_yP": 1, "Delta_ys": 0.1, "Delta_yn": 0.1, "Delta_t": 0.5},
]

# Cálculos
resultados = []
for i, caso in enumerate(casos):
    Delta_x = 1  # Constante
    Delta_yP = caso["Delta_yP"]
    Delta_ys = caso["Delta_ys"]
    Delta_yn = caso["Delta_yn"]
    Delta_t = caso["Delta_t"]

    # Cálculo de los coeficientes
    a_P = (rho * cp * Delta_x * Delta_yP) / Delta_t + k * (Delta_yP / Delta_x) + k / Delta_yn + k / Delta_ys
    a_W = k * Delta_yP / Delta_x
    a_E = k * Delta_yP / Delta_x
    a_N = k / Delta_yn
    a_S = k / Delta_ys

    # Guardar resultados
    resultados.append({
        "Caso": f"Caso {i+1}",
        "a_P": a_P,
        "a_W": a_W,
        "a_E": a_E,
        "a_N": a_N,
        "a_S": a_S,
    })

# Mostrar resultados en una tabla

tabla_resultados = pd.DataFrame(resultados)
print(tabla_resultados)
