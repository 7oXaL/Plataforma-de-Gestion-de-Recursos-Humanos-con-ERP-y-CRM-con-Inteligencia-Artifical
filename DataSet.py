import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import IsolationForest

np.random.seed()  # Semilla de aleatoriedad

# Simulación de datos de 100 empleados y 30 días de registros
empleados = [f"Empleado_{i}" for i in range(1, 31)]  # Creación de los 30 empleados

dias = pd.date_range(start="2025-01-01", periods=20, freq='D')  # 20 días

asistencia_data = []  # Registro de asistencias

# Crear registros con '1' para asistencia y '0' para ausencia (con algunas ausencias repetidas para anomalías)
#Dataset
for empleado in empleados:
    asistencias = np.random.choice([0, 1], size=20)  # Genera un arreglo de tamaño 20 con 0 o 1 simulando la asistencia
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

# Mostrar los primeros registros de asistencia
print("Datos de Asistencia:")
print(df)  # Mostramos el dataframe

# Preprocesamiento de los datos
# Convertir ausencias (0) a 1 y asistencias (1) a 0 para detectar anomalías
df_bin = df.applymap(lambda x: 1 if x == 0 else 0)


# Aplicar Isolation Forest para detectar anomalías
model = IsolationForest(contamination=0.1, random_state=42)  # El valor de contaminacion esta en 10% dado que esperamos ese porcentaje de ausentismo.
model.fit(df_bin)

# Predecir anomalías
anomalies = model.predict(df_bin)

# Las predicciones son: 1 para normal, -1 para anomalía
# Crear un DataFrame con las predicciones
resultados = pd.DataFrame(anomalies, index=df.index, columns=["Anomalía"])
resultados['Anomalía'] = resultados['Anomalía'].map({1: "Normal", -1: "Anomalía"})

# Mostrar los resultados de las anomalías
print("\nResultados de Anomalías:")
print(resultados)

# Graficar los resultados de las anomalías
# 1. Graficar la asistencia de los primeros 5 empleados (para ver la distribución)
plt.figure(figsize=(10, 6))

# Seleccionamos los empleados para visualización
empleados_grafico = empleados[:30]
df_grafico = df.loc[empleados_grafico]

# Crear un heatmap de asistencia: filas = empleados, columnas = días
plt.figure(figsize=(15, 10))
sns.heatmap(df, cmap="Greens", cbar_kws={'label': 'Asistencia'}, linewidths=0.2, linecolor='gray')

plt.title("Mapa de Calor de Asistencia de Empleados (1 = Asistió, 0 = Ausente)", fontsize=14)
plt.xlabel("Días")
plt.ylabel("Empleados")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
