import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed()  # Semilla de aleatoriedad

# Simulación de datos de 30 empleados y 20 días de registros
empleados = [f"Empleado_{i}" for i in range(1, 31)]  # Creación de los 30 empleados

dias = pd.date_range(start="2025-01-01", periods=20, freq='D')  # 20 días

asistencia_data = []  # Registro de asistencias

# Crear registros con '1' para asistencia y '0' para ausencia (con algunas ausencias repetidas para anomalías)
#Dataset
for empleado in empleados:
    asistencias = np.random.choice([0, 1], size=20, p= [0.2,0.8] ) # Genera un arreglo de tamaño 20 con 0 o 1 simulando la asistencia
    asistencia_data.append(asistencias)

# Crear un DataFrame (Tabla con DIAS con el valor de asistencia por empleado)
df = pd.DataFrame(asistencia_data, columns=dias.strftime('%Y-%m-%d'), index=empleados)

# Obtener ruta del escritorio del usuario
escritorio = os.path.join(os.path.expanduser("~"), "Desktop")

# Crear la ruta completa del archivo
ruta_archivo = os.path.join(escritorio, "Asistencia de datos.xlsx")

# Guardar el DataFrame como .txt (convertido a matriz)
df.to_excel(ruta_archivo, index=True, header=True)

print(f"Archivo guardado en: {ruta_archivo}")

# Preprocesamiento de los datos
# Convertir ausencias (0) a 1 y asistencias (1) a 0 para detectar anomalías
df_bin = df.applymap(lambda x: 1 if x == 0 else 0)


# Aplicar Isolation Forest para detectar anomalías
model = IsolationForest(contamination=0.2, random_state=42)  # El valor de contaminacion esta en 20% dado que esperamos ese porcentaje de ausentismo.
model.fit(df_bin)

# Predecir anomalías
anomalies = model.predict(df_bin)

# Las predicciones son: 1 para normal, -1 para anomalía
# Crear un DataFrame con las predicciones
# Resultados del modelo
resultados = pd.DataFrame(index=df.index)
resultados["Predicción_Modelo"] = pd.Series(anomalies, index=df.index).map({1: "Normal", -1: "Anomalía"})


# FUNCIONALIDAD NUEVA: Etiquetas reales según 3 inasistencias seguidas
def tiene_inasistencias_seguidas(row, cantidad=3):
    conteo = 0
    for val in row:
        if val == 0:
            conteo += 1
            if conteo >= cantidad:
                return "Anomalía"
        else:
            conteo = 0
    return "Normal"

etiquetas_reales = df.apply(lambda row: tiene_inasistencias_seguidas(row, cantidad=3), axis=1)
resultados['Etiqueta_Real'] = etiquetas_reales

# Evaluación del modelo
y_true = etiquetas_reales.map({"Normal": 0, "Anomalía": 1})
y_pred = resultados['Predicción_Modelo'].map({"Normal": 0, "Anomalía": 1})


# Mostrar los resultados de las anomalías
print("\nResultados de Anomalías:")
print(resultados)

print("\nEtiquetas reales:")
print(etiquetas_reales.value_counts())

# Convertir etiquetas reales a binario
y_true = etiquetas_reales.map({"Normal": 0, "Anomalía": 1})

# --------- Ajuste de hiperparámetros: contamination ---------
contamination_vals = np.arange(0.05, 0.31, 0.05)
metricas = {
    "contamination": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

for c in contamination_vals:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(df_bin)
    y_pred = pd.Series(model.predict(df_bin), index=df.index).map({1: 0, -1: 1})

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metricas["contamination"].append(c)
    metricas["precision"].append(precision)
    metricas["recall"].append(recall)
    metricas["f1_score"].append(f1)

# Mostrar métricas
resultados_df = pd.DataFrame(metricas)
print("\nMétricas por nivel de contamination:")
print(resultados_df)

# Mostrar mejor valor de F1
mejor_f1 = resultados_df.loc[resultados_df['f1_score'].idxmax()]
print(f"\n✅ Mejor F1 Score: {mejor_f1['f1_score']:.3f} con contamination={mejor_f1['contamination']:.2f}")

# Graficar métricas
plt.figure(figsize=(10, 6))
plt.plot(resultados_df["contamination"], resultados_df["precision"], marker='o', label='Precisión')
plt.plot(resultados_df["contamination"], resultados_df["recall"], marker='s', label='Recall')
plt.plot(resultados_df["contamination"], resultados_df["f1_score"], marker='^', label='F1 Score')
plt.title("Evaluación por valores de 'contamination'")
plt.xlabel("Contamination")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap de asistencia
plt.figure(figsize=(15, 10))
sns.heatmap(df, cmap="Greens", cbar_kws={'label': 'Asistencia'}, linewidths=0.2, linecolor='gray')
plt.title("Mapa de calor de asistencia")
plt.ylabel("Empleados")
plt.xlabel("Días")
plt.tight_layout()
plt.show()
