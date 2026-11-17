---
title: "Guion de sustentación oral: 12 minutos + 5 de preguntas"
subtitle: "Proyecto Integrador · Tráfico en la autopista I-94"
author: "Tony Yu · Luis Rodríguez · Maydda Rodríguez"
---

# Reparto y tiempos (12 min)

| Min | Quién | Contenido | Pantalla |
|---|---|---|---|
| 0:00–1:00 | Persona A | Problema y pregunta: ¿se puede predecir el tráfico horario de la I-94 y qué lo determina? Datos: 48.204 filas, 40.575 horas únicas, 2012–2018. | Diapositiva 1–2 |
| 1:00–3:00 | Persona A | Fases 1 y 2: calidad (17 duplicados, 11 valores imposibles, 22,8 % de horas ausentes), partición 70/15/15 sin fuga, baseline R² 0,838 y F1 0,891. | Diapositiva 3 |
| 3:00–6:00 | Persona B | Fase 3: XGBoost R² 0,844 → 0,966 y F1 0,896 → 0,927 en prueba (una sola evaluación); IC 95 % sin cero; sobreajuste (brecha 0,011); SHAP: hora 70 %, clima < 7 %; 5 grupos, silueta 0,33. | Diapositivas 4–5 |
| 6:00–10:00 | Persona C | **Demo en vivo del dashboard** (ver guion abajo). | Navegador |
| 10:00–11:30 | Persona C | Impacto y límites: error en horas pico baja 59–82 %; simulador de alertas; limitaciones honestas. | Diapositiva 6 |
| 11:30–12:00 | Persona A | Conclusiones y trabajo futuro; repositorio reproducible. | Cierre |

# Guion de la demo (4 minutos)

1. **Filtros y KPI (45 s).** Abrir la narrativa. Filtrar *Tipo de día = Laboral* y *Hora 6–9*: las tarjetas y el texto de la curva horaria cambian. Mostrar que el error medio del modelo (MAE) se recalcula con el filtro.
2. **Narrativa (45 s).** Gráfico SHAP: "la hora pesa 70 %, el clima menos de 7 %". Mapa de calor: dos picos laborales, joroba de fin de semana.
3. **Predicción en vivo (90 s).** Fecha viernes 15 de junio de 2018, 17:00, 18 °C, despejado: volumen ≈ 5.680 veh/h, nivel ALTO. Cambiar a sábado: baja. Cambiar a lluvia o a festivo y comentar el intervalo del 80 %.
4. **Segmentación (45 s).** Dispersión PCA 2D y luego 3D; tabla de perfiles; heatmap de horas por grupo.
5. **Simulador de alertas (30 s)** en *Modelos y evidencia*: mover el umbral y mostrar el compromiso falsas alarmas / congestiones no detectadas.

*Plan B si falla internet:* correr `streamlit run dashboard/app.py` en local, o usar las capturas de `docs/capturas/`.

# Preguntas probables y respuestas honestas

**¿Por qué XGBoost y no el MLP?** Quedaron muy cerca (R² 0,966 y 0,962). XGBoost es más fácil de explicar con SHAP, se entrena en segundos y no requiere GPU. No hicimos una prueba estadística formal entre los finalistas.

**¿Hay fuga de datos?** No: el preprocesamiento va dentro de un *Pipeline*, los hiperparámetros se buscan con validación cruzada solo en entrenamiento y el conjunto de prueba se evaluó una vez, con la selección fijada antes. Se auditó que el escalador solo usa estadísticas de entrenamiento.

**La partición es aleatoria, ¿no infla el resultado?** Puede ser algo optimista, porque horas vecinas quedan en conjuntos distintos. En la Fase 2 una validación temporal (desde 2017) no mostró deterioro, pero no es una partición temporal estricta de todo el modelo. Es una limitación declarada.

**¿Por qué no usaron el tráfico de la hora anterior?** Porque se quería un modelo de calendario y clima que sirva para planificar sin depender de datos recientes, y porque la serie tiene 22,8 % de horas ausentes. Es trabajo futuro.

**¿Qué significa una silueta de 0,33?** Estructura débil a moderada: los grupos son regímenes de calendario con bordes difusos. K = 5 coincide entre codo y silueta, es estable (ARI 0,955) y coincide con el jerárquico (ARI 0,91).

**¿El clima no importa?** En este conjunto aporta poco una vez se conocen hora y día. Eso es asociación predictiva, no prueba de que no afecte al tráfico; hay pocas horas de precipitación intensa.

**¿Cuál es el impacto de negocio?** El error baja un 59 % en horas pico laborales y un 82 % en pico de fin de semana; los errores graves de clasificación pasan de 6 a 1. No cuantificamos costos: el umbral de alerta debe fijarse con la entidad operadora, y las acciones por grupo son propuestas nuestras.

**¿El intervalo del 80 % es confiable?** Se calcula con los residuos de prueba, por lo que en prueba cubre el 80 % por construcción; no es una validación independiente.

**¿Cómo se reproduce?** `pip install -r requirements.txt`, ejecutar los tres cuadernos en orden, `pytest tests` y `streamlit run dashboard/app.py`. Hay una prueba que comprueba que `src/` reproduce particiones y métricas.
