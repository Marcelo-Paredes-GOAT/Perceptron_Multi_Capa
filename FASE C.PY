import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 1. Generamos las predicciones de ambos modelos
y_pred_perceptron = modelo_perceptron.predict(X)
y_pred_mlp = modelo_mlp.predict(X)

# 2. Función auxiliar para calcular todas las métricas de un modelo
def calcular_metricas(y_real, y_pred):
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, zero_division=0) # zero_division=0 evita errores si no predice ningún 1
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    cm = confusion_matrix(y_real, y_pred)
    return acc, prec, rec, f1, cm

# Calculamos para Perceptron
acc_p, prec_p, rec_p, f1_p, cm_p = calcular_metricas(y, y_pred_perceptron)

# Calculamos para MLP
acc_m, prec_m, rec_m, f1_m, cm_m = calcular_metricas(y, y_pred_mlp)

# 3. Creamos la tabla comparativa con Pandas
tabla_resultados = pd.DataFrame({
    'Métrica': ['Accuracy (Exactitud)', 'Precision (Precisión)', 'Recall (Sensibilidad)', 'F1-Score'],
    'Perceptron (Fase A)': [f"{acc_p:.2f}", f"{prec_p:.2f}", f"{rec_p:.2f}", f"{f1_p:.2f}"],
    'MLP (Fase B)': [f"{acc_m:.2f}", f"{prec_m:.2f}", f"{rec_m:.2f}", f"{f1_m:.2f}"]
})

print("--- TABLA COMPARATIVA DE MÉTRICAS ---")
print(tabla_resultados.to_string(index=False))
print("\n" + "="*40 + "\n")

# 4. Mostramos las Matrices de Confusión
print("--- MATRICES DE CONFUSIÓN ---")
print("Perceptron:")
print(pd.DataFrame(cm_p, index=['Real 0', 'Real 1'], columns=['Pred 0', 'Pred 1']))
print("\nMLP:")
print(pd.DataFrame(cm_m, index=['Real 0', 'Real 1'], columns=['Pred 0', 'Pred 1']))