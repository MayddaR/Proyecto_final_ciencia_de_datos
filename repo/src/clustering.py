"""Segmentación de las horas de operación (Fase 3): K-means con K = 5, numerado por tráfico medio creciente."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import K_CLUSTERS, NOMBRES_GRUPOS, SEMILLA


def feats_cluster(d: pd.DataFrame) -> pd.DataFrame:
    """Siete variables: tráfico, hora cíclica, fin de semana, festivo, temperatura y presencia de precipitación."""
    return pd.DataFrame({
        "tráfico": d["traffic_volume"].values,
        "hora_sen": np.sin(2 * np.pi * d["hora"].values / 24), "hora_cos": np.cos(2 * np.pi * d["hora"].values / 24),
        "fin_de_semana": (d["dia_semana"].values >= 5).astype(int), "festivo": d["es_festivo"].values,
        "temp": d["temp"].values, "precipitación": ((d["rain_1h"] > 0) | (d["snow_1h"] > 0)).astype(int).values})


@dataclass
class Segmentador:
    escalador: StandardScaler
    kmeans: KMeans
    renum: dict
    pca: PCA

    def etiquetar(self, d: pd.DataFrame) -> np.ndarray:
        z = self.escalador.transform(feats_cluster(d))
        return np.array([self.renum[int(l)] for l in self.kmeans.predict(z)])

    def proyectar(self, d: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.escalador.transform(feats_cluster(d)))


def ajustar_segmentador(train: pd.DataFrame, k: int = K_CLUSTERS) -> Segmentador:
    esc = StandardScaler().fit(feats_cluster(train))
    z = esc.transform(feats_cluster(train))
    km = KMeans(k, n_init=20, random_state=SEMILLA).fit(z)
    orden = np.argsort(pd.Series(train["traffic_volume"].values).groupby(km.labels_).mean().values)
    renum = {int(v): i for i, v in enumerate(orden)}
    pca = PCA(3, random_state=SEMILLA).fit(z)
    return Segmentador(esc, km, renum, pca)


def nombre_grupo(i: int) -> str:
    return NOMBRES_GRUPOS[i] if 0 <= i < len(NOMBRES_GRUPOS) else f"Grupo {i}"
