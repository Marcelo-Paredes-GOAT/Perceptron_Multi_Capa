import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Definimos los datos XOR nuevamente por si acaso
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 2. Implementación del MLP (Fase B)
print("Entrenando el Multi-Layer Perceptron (MLP)...")
# Arquitectura: 1 capa oculta con 4 neuronas, activación ReLU. 
# max_iter se aumenta para asegurar la convergencia.
# Agregamos solver='lbfgs'
# Cambiamos a activación sigmoide ('logistic') y subimos a 8 neuronas
modelo_mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='logistic', solver='lbfgs', max_iter=2000, random_state=42)
modelo_mlp.fit(X, y)

# 3. Evaluamos el modelo
y_pred = modelo_mlp.predict(X)
precision = accuracy_score(y, y_pred) * 100

print("-" * 30)
print("Resultados Fase B:")
for entrada, prediccion, real in zip(X, y_pred, y):
    print(f"Entrada: {entrada} | Predicción: {prediccion} | Real: {real}")
print("-" * 30)
print(f"Precisión (Accuracy) alcanzada: {precision}%")

# --- 4. Visualización de la Frontera de Decisión (La Gráfica) ---
# Crear una malla de puntos para evaluar el espacio bidimensional
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
# Predecir cada punto de la malla para dibujar las regiones
Z = modelo_mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar
plt.figure(figsize=(7, 5))
# Dibujar el fondo de colores según la predicción
plt.contourf(xx, yy, Z, alpha=0.7, cmap=plt.cm.RdBu)
# Dibujar los puntos originales del XOR
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=150)

plt.title('¿Cómo clasifica el MLP? - Frontera de Decisión (XOR)')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()