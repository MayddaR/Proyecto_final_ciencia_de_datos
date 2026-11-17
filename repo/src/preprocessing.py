"""Limpieza, ingeniería de variables y partición (Fase 2). Reproduce data/processed/{train,val,test}.csv."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RUTA_CRUDO, SEMILLA, UMBRAL_ALTO, UMBRAL_BAJO, ENTRADAS


def cargar_crudo(ruta=RUTA_CRUDO) -> pd.DataFrame:
    """Lee el CSV de UCI. 'None' en holiday es una categoría válida (no festivo), no un faltante."""
    df = pd.read_csv(ruta, keep_default_na=False, na_values=[""], parse_dates=["date_time"])
    df["temp"] = df["temp"] - 273.15                                   # Kelvin → °C
    df["nivel_trafico"] = pd.cut(df["traffic_volume"], bins=[-1, UMBRAL_BAJO - 1, UMBRAL_ALTO - 1, np.inf], labels=["bajo", "medio", "alto"])
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicados, valores imposibles → NaN y variables de calendario. No usa la variable objetivo para derivar variables."""
    df = df.drop_duplicates()                                          # filas idénticas
    df = df.drop_duplicates(subset="date_time", keep="first").reset_index(drop=True)   # una fila por hora
    df["weather_description"] = df["weather_description"].str.lower()
    df.loc[df["temp"] == -273.15, "temp"] = np.nan                     # 0 K
    df.loc[df["rain_1h"] > 100, "rain_1h"] = np.nan                    # > 100 mm en una hora
    df["hora"] = df["date_time"].dt.hour
    df["dia_semana"] = df["date_time"].dt.day_of_week                  # 0 = lunes
    df["mes"] = df["date_time"].dt.month
    dias_festivos = df.loc[df["holiday"] != "None", "date_time"].dt.normalize().unique()   # la fuente marca solo la hora 00:00
    df["es_festivo"] = df["date_time"].dt.normalize().isin(dias_festivos).astype(int)
    return df


def particionar(df: pd.DataFrame, semilla: int = SEMILLA):
    """70/15/15 estratificado por nivel de tráfico; la imputación usa solo estadísticas de entrenamiento."""
    train, temp_ = train_test_split(df, test_size=0.30, stratify=df["nivel_trafico"], random_state=semilla)
    val, test = train_test_split(temp_, test_size=0.50, stratify=temp_["nivel_trafico"], random_state=semilla)
    mediana_temp = train.groupby(["mes", "hora"])["temp"].median()
    mediana_lluvia = train.loc[train["weather_description"] == "very heavy rain", "rain_1h"].median()

    def imputar(d):
        d = d.copy()
        falta = d["temp"].isna()
        claves = list(zip(d.loc[falta, "mes"], d.loc[falta, "hora"]))
        d.loc[falta, "temp"] = mediana_temp.loc[claves].to_numpy()
        d["rain_1h"] = d["rain_1h"].fillna(mediana_lluvia)
        return d

    return imputar(train), imputar(val), imputar(test)


COLUMNAS = ["date_time"] + ENTRADAS + ["traffic_volume", "nivel_trafico"]


def preparar(ruta=RUTA_CRUDO):
    """Del CSV crudo a (train, val, test) listos para modelar."""
    train, val, test = particionar(limpiar(cargar_crudo(ruta)))
    return train[COLUMNAS], val[COLUMNAS], test[COLUMNAS]


def cargar_procesado(carpeta):
    """Lee las particiones ya guardadas por el cuaderno de la Fase 2."""
    return [pd.read_csv(carpeta / f"{n}.csv", parse_dates=["date_time"]) for n in ("train", "val", "test")]
