import numpy as np
import matplotlib.pyplot as plt

# =========================
# Datos de entrenamiento
# =========================
x = 1.0      # entrada
t = 1.0      # valor objetivo

# =========================
# Parámetros iniciales
# =========================
w = -2.0
b = 0.0
eta = 0.5
epochs = 50

loss_history = []

# =========================
# Función sigmoide
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# Entrenamiento (Backprop)
# =========================
for epoch in range(epochs):

    # Forward pass
    z = w * x + b
    y = sigmoid(z)
    
    # Loss
    loss = 0.5 * (y - t)**2
    loss_history.append(loss)
    
    # Backpropagation
    dL_dy = y - t
    dy_dz = y * (1 - y)
    
    dL_dw = dL_dy * dy_dz * x
    dL_db = dL_dy * dy_dz
    
    # Actualización
    w -= eta * dL_dw
    b -= eta * dL_db

# =========================
# Gráfica
# =========================
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Iteraciones")
plt.ylabel("Loss")
plt.title("Convergencia usando Backpropagation (1 neurona)")
plt.grid(True)
plt.show()

print("Peso final:", w)
print("Bias final:", b)