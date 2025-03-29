# Plataforma-de-Gestion-de-Recursos-Humanos-con-ERP-y-CRM-con-Inteligencia-Artifical
Universidad de General Sarmiento - Laboratorio de construcción de Software

[![](https://inamu.musica.ar/uploads/paginas/imagenes/e0afe4e68162dcf9a292afea47ba3dd2.png)](https://inamu.musica.ar/uploads/paginas/imagenes/e0afe4e68162dcf9a292afea47ba3dd2.png)

# GRAFICO
0= inasistencia
1= asistencia
 En el grafico se muestra el comportamiento de asistencia de 5 empleados a lo largo del mes, en el mismo se observa que el empleado 1 no asistio el dia 06 al 08 es decir que falto 3 dias, lo que representa una anomalia, de igual manera el emplado 2 mas adelante, no asiste por 3 dias consecutivos.



# Dataframes con asitencias
Datos de Asistencia:
              2025-01-01  2025-01-02  ...  2025-01-29  2025-01-30
Empleado_1             1           1  ...           1           0
Empleado_2             1           1  ...           0           1
Empleado_3             1           1  ...           1           1
Empleado_4             1           1  ...           1           1
Empleado_5             1           1  ...           0           1
...                  ...         ...  ...         ...         ...
Empleado_96            1           1  ...           1           1
Empleado_97            1           1  ...           1           1
Empleado_98            1           0  ...           1           0
Empleado_99            1           1  ...           1           1
Empleado_100           1           1  ...           1           1

# Dataframes con los resultados de las anomalias
Resultados de Anomalías:
              
Empleado_1      Anomalía
Empleado_2    Anomalía
Empleado_3    Anomalía
Empleado_4      Normal
Empleado_5      Normal
...                ...
Empleado_96     Normal
Empleado_97   Anomalía
Empleado_98     Normal
Empleado_99     Normal
Empleado_100    Normal