# def f(x):
#     return x**3 - x - 2
# def df(x):
#     return 3*x**2 - 1
# def newton(x0, tol=1e-6, max_iter=20):
#     for i in range(max_iter):
#         x1 = x0 - f(x0)/df(x0)
#         if abs(x1 - x0) < tol:
#             return x1
#         x0 = x1
#     return None
# raiz = newton(1.5)
# print("Raiz:", raiz)

# def bissecao(f, a, b, tol=1e-6):
#     if f(a)*f(b) >= 0:
#         return None
#     while (b - a)/2 > tol:
#         c = (a + b)/2
#         if f(c) == 0:
#             return c
#         elif f(a)*f(c) < 0:
#             b = c
#         else:
#             a = c
#     return (a + b)/2
# raiz = bissecao(f, 1, 2)
# print("Raiz:", raiz)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
#
# def f(x):
#     return np.exp(-x) * np.sin(x) - np.cos(x)
#
# raiz = fsolve (f,x0=-5)
# print("Raiz ", raiz)
# x_vals = np.linspace(-15, 15, 400)
# y_vals = f(x_vals)
#
# plt.plot(x_vals, y_vals)
# plt.axhline(y=0, color='k')
# plt.grid(True)
# plt.show()

from scipy.optimize import fsolve
from scipy.optimize import root
import numpy as np

def sistema(vars):
    x, y = vars
    eq_1 = x**2 + np.sin(y) - 1
    eq_2 = x * y - 0.5
    return [eq_1, eq_2]

# raiz, info, ier, msg = fsolve(sistema, [0.5, 1], full_output=True)
# print("x =", raiz[0])
# print("y =", raiz[1])

# Chute inicial
x0 = [0.5, 2.7]
sol = root(sistema, x0, method='hybr', options={'xtol': 1e-12, 'maxiter': 10000})
print("Sucesso?", sol.success)
print("x =", sol.x[0])
print("y =", sol.x[1])
print("f(x, y) =", sol.fun)
print("Mensagem:", sol.message)
