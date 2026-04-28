import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. Definimos el dataset XOR según la guía
# Entradas binarias
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Etiquetas objetivo correspondientes
y = np.array([0, 1, 1, 0])

# 2. Implementación del Perceptron simple (Fase A)
print("Entrenando el Perceptron simple...")
# random_state asegura que los resultados sean reproducibles
modelo_perceptron = Perceptron(max_iter=1000, random_state=42)
modelo_perceptron.fit(X, y)

# 3. Evaluamos el modelo
y_pred = modelo_perceptron.predict(X)
precision = accuracy_score(y, y_pred) * 100

print("-" * 30)
print("Resultados Fase A:")
for entrada, prediccion, real in zip(X, y_pred, y):
    print(f"Entrada: {entrada} | Predicción: {prediccion} | Real: {real}")

print("-" * 30)
print(f"Precisión (Accuracy) alcanzada: {precision}%")