import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# Datos
# =========================
x = 1.0
t = 1.0

# Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Rango de valores
w_vals = np.linspace(-6, 6, 100)
b_vals = np.linspace(-6, 6, 100)

W, B = np.meshgrid(w_vals, b_vals)

# Forward
Z = W * x + B
Y = sigmoid(Z)

# Loss
L = 0.5 * (Y - t)**2

# =========================
# Gráfica 3D
# =========================
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(W, B, L)

ax.set_xlabel("Peso w")
ax.set_ylabel("Bias b")
ax.set_zlabel("Loss")
ax.set_title("Superficie de Error L(w, b)")

plt.show()