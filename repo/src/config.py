"""Constantes del proyecto: semilla, umbrales, variables e hiperparámetros elegidos en la Fase 3."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
RUTA_PROCESADO = RAIZ / "data" / "processed"
RUTA_MODELOS = RAIZ / "models"

SEMILLA = 42
UMBRAL_BAJO, UMBRAL_ALTO = 1500, 4500          # vehículos por hora (bajo < 1.500 ≤ medio < 4.500 ≤ alto)
ORDEN = ["bajo", "medio", "alto"]
COD = {n: i for i, n in enumerate(ORDEN)}

LOG, NUM, CAT, BIN = ["rain_1h", "snow_1h"], ["temp", "clouds_all"], ["weather_main", "hora", "dia_semana", "mes"], ["es_festivo"]
ENTRADAS = LOG + NUM + CAT + BIN
HORAS_PICO = [6, 7, 8, 16, 17]

# Hiperparámetros elegidos con validación cruzada de 5 pliegues (Fase 3, resultados_fase3.json)
HP_REG = dict(n_estimators=600, max_depth=8, learning_rate=0.03, subsample=1.0, colsample_bytree=0.85, min_child_weight=1, reg_lambda=5)
HP_CLF = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=0.85, min_child_weight=1, reg_lambda=10)

# Segmentación (Fase 3): K elegido por codo y silueta; nombres de los grupos (ordenados por tráfico medio)
K_CLUSTERS = 5
NOMBRES_GRUPOS = ["Noches y madrugadas laborales", "Festivos", "Fines de semana", "Horas con precipitación", "Laborales diurnos"]
