# Tráfico en la autopista I-94: predicción, clasificación y segmentación

Proyecto Integrador de Ciencia de Datos · Maestría en Inteligencia Artificial · Universidad de La Salle
Autores: Tony Yu, Luis Rodríguez y Maydda Rodríguez · Docente: Fabián Camilo Castro Riveros

Solución de extremo a extremo sobre *Metro Interstate Traffic Volume* (UCI; Hogue, 2019, CC BY 4.0):
regresión del volumen horario, clasificación del nivel (bajo / medio / alto), segmentación de las horas
(K-means, K = 5) y un dashboard interactivo en Streamlit.

**Aplicación desplegada:** _(agregar la URL de Streamlit Community Cloud tras el despliegue)_

## Resultados principales (conjunto de prueba, evaluado una sola vez)

| Tarea | Modelo base | XGBoost |
|---|---|---|
| Regresión: R² / RMSE (veh/h) | 0,844 / 783 | **0,966 / 364** |
| Clasificación: F1 macro / AUC | 0,896 / 0,972 | **0,927 / 0,986** |

La hora explica el 70 % de la atribución SHAP, el día de la semana el 18,5 % y todo el clima menos del 7 %.
Limitaciones (partición aleatoria, 22,8 % de horas ausentes, silueta moderada de 0,33) en `docs/Entregable4_Informe_Tecnico_Final.docx`.

## Estructura

```
data/raw/         datos originales (UCI)
data/processed/   train.csv, val.csv, test.csv (salida del cuaderno 02)
models/           resultados_fase2.json, resultados_fase3.json y artefactos del cuaderno 03
notebooks/        01_eda · 02_preprocesamiento · 03_modelado
src/              módulos reutilizables (preprocesamiento, modelos, clustering)
dashboard/        app.py (Streamlit) y requirements.txt para el despliegue
docs/             informes APA de los cuatro entregables y capturas del dashboard
tests/            prueba de reproducibilidad
```

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements.txt                        # versiones exactas
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_preprocesamiento.ipynb 03_modelado.ipynb && cd ..
pytest -q tests                                        # src/ reproduce particiones y métricas
streamlit run dashboard/app.py                         # dashboard en http://localhost:8501
```

El cuaderno 03 tarda unos 15 minutos con 2 núcleos de CPU, sin GPU. Semilla global: 42.
El dashboard reentrena los modelos XGBoost al iniciar (≈ 30 s la primera vez) con `data/processed/`.

## Despliegue en Streamlit Community Cloud

1. Publicar este repositorio en GitHub (público).
2. En https://share.streamlit.io elegir *Create app*, seleccionar el repositorio y la rama.
3. *Main file path*: `dashboard/app.py`. En *Advanced settings* elegir Python 3.13.
4. Streamlit instala las dependencias de `dashboard/requirements.txt`. Copiar la URL resultante a este README.

## Despliegue en Render (alternativa)

El repositorio incluye `render.yaml` (Blueprint). En https://dashboard.render.com: *New → Blueprint*, conectar el
repositorio de GitHub y aplicar. Equivale a crear un *Web Service* con:

- Build command: `pip install -r dashboard/requirements.txt`
- Start command: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Variable de entorno `PYTHON_VERSION` = 3.13.5 (ajustar si Render indica que no existe)
- Health check path: `/_stcore/health`

Los modelos precalculados están en `models/dashboard/` (se regeneran con `python -m src.exportar_modelos`); si no
cargan, el dashboard reentrena solo, pero en el plan gratuito (0,1 CPU) eso tardaría varios minutos.

## Módulos del dashboard

Filtros dinámicos (fechas, clima, día, hora, temperatura, nubosidad, precipitación, volumen, partición) · tarjetas de KPI ·
narrativa de datos · predicción en vivo (volumen, nivel, probabilidades, intervalo empírico del 80 %, grupo) ·
segmentación (PCA 2D/3D, perfiles, composición horaria) · evidencia de los modelos y simulador de alertas.
