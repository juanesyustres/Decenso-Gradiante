import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
eta = 0.5      # tasa de aprendizaje
w_star = 2.0   # peso óptimo
w0 = 0.0       # peso inicial

# Ecuación diferencial: dw/dt = -eta * (w - w_star)
def dw_dt(t, w):
    return -eta * (w - w_star)

# Resolver numéricamente
t_span = (0, 10)
t_eval = np.linspace(0, 10, 300)
sol = solve_ivp(dw_dt, t_span, [w0], t_eval=t_eval)

# Solución analítica para comparar
w_analitica = w_star + (w0 - w_star) * np.exp(-eta * t_eval)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label='Solución numérica (solve_ivp)', linewidth=2)
plt.plot(t_eval, w_analitica, '--', label='Solución analítica', linewidth=2)
plt.axhline(w_star, color='gray', linestyle=':', label=f'Peso óptimo w* = {w_star}')
plt.xlabel('Tiempo (iteraciones)')
plt.ylabel('Peso w(t)')
plt.title('Back Propagation: Convergencia del peso con descenso del gradiente')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('backpropagation_convergencia.png', dpi=150)
plt.show()