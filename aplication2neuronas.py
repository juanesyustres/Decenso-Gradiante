import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dataset simple
# =========================
X = np.array([0, 1])
T = np.array([0, 1])

# =========================
# Inicialización parámetros
# =========================
np.random.seed(0)

w1, b1 = np.random.randn(), np.random.randn()
w2, b2 = np.random.randn(), np.random.randn()
v1, v2, b3 = np.random.randn(), np.random.randn(), np.random.randn()

eta = 0.5
epochs = 2000

loss_history = []

# =========================
# Sigmoide
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# Entrenamiento
# =========================
for epoch in range(epochs):

    total_loss = 0
    
    for x, t in zip(X, T):
        
        # ---- Forward ----
        z1 = w1*x + b1
        z2 = w2*x + b2
        
        h1 = sigmoid(z1)
        h2 = sigmoid(z2)
        
        z3 = v1*h1 + v2*h2 + b3
        y = sigmoid(z3)
        
        loss = 0.5 * (y - t)**2
        total_loss += loss
        
        # ---- Backprop ----
        dL_dy = y - t
        dy_dz3 = y * (1 - y)
        
        # Gradientes salida
        dL_dv1 = dL_dy * dy_dz3 * h1
        dL_dv2 = dL_dy * dy_dz3 * h2
        dL_db3 = dL_dy * dy_dz3
        
        # Gradientes capa oculta
        dz3_dh1 = v1
        dz3_dh2 = v2
        
        dh1_dz1 = h1 * (1 - h1)
        dh2_dz2 = h2 * (1 - h2)
        
        dL_dw1 = dL_dy * dy_dz3 * dz3_dh1 * dh1_dz1 * x
        dL_db1 = dL_dy * dy_dz3 * dz3_dh1 * dh1_dz1
        
        dL_dw2 = dL_dy * dy_dz3 * dz3_dh2 * dh2_dz2 * x
        dL_db2 = dL_dy * dy_dz3 * dz3_dh2 * dh2_dz2
        
        # ---- Actualización ----
        w1 -= eta * dL_dw1
        b1 -= eta * dL_db1
        
        w2 -= eta * dL_dw2
        b2 -= eta * dL_db2
        
        v1 -= eta * dL_dv1
        v2 -= eta * dL_dv2
        b3 -= eta * dL_db3

    loss_history.append(total_loss)

# =========================
# Gráfica
# =========================
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Red neuronal 1-2-1 (Backpropagation manual)")
plt.grid(True)
plt.show()