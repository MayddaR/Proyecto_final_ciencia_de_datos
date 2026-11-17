"""Preprocesadores (scikit-learn). Siempre se ajustan dentro de un Pipeline, solo con datos de entrenamiento."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import BIN, CAT, LOG, NUM


def preprocesador_lineal() -> ColumnTransformer:
    """Fase 2: log1p + estandarización para lluvia y nieve, estandarización, one-hot y festivo (modelos lineales y MLP)."""
    return ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), LOG),
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), CAT),
        ("bin", "passthrough", BIN),
    ])


def preprocesador_arboles() -> ColumnTransformer:
    """Fase 3: los árboles no necesitan escalar; hora, día y mes entran como números."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), ["weather_main"]),
        ("num", "passthrough", ["rain_1h", "snow_1h", "temp", "clouds_all", "hora", "dia_semana", "mes", "es_festivo"]),
    ], verbose_feature_names_out=False)
