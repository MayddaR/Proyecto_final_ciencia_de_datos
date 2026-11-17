"""Modelos finales de la Fase 3 (XGBoost de regresión y de clasificación) y métricas."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from .config import COD, ENTRADAS, HP_CLF, HP_REG, SEMILLA
from .features import preprocesador_arboles


def pipeline_regresor(n_jobs: int = 2) -> Pipeline:
    return Pipeline([("pre", preprocesador_arboles()),
                     ("m", XGBRegressor(tree_method="hist", n_jobs=n_jobs, random_state=SEMILLA, **HP_REG))])


def pipeline_clasificador(n_jobs: int = 2) -> Pipeline:
    return Pipeline([("pre", preprocesador_arboles()),
                     ("m", XGBClassifier(tree_method="hist", n_jobs=n_jobs, random_state=SEMILLA, **HP_CLF))])


def entrenar(train, n_jobs: int = 2):
    """Ajusta ambos modelos con el conjunto de entrenamiento."""
    reg = pipeline_regresor(n_jobs).fit(train[ENTRADAS], train["traffic_volume"].values)
    clf = pipeline_clasificador(n_jobs).fit(train[ENTRADAS], train["nivel_trafico"].astype(str).map(COD).values)
    return reg, clf


def met_reg(y, p) -> dict:
    return {"R²": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": mean_absolute_error(y, p)}


def met_clf(y, pred, proba) -> dict:
    return {"Accuracy": accuracy_score(y, pred), "Precisión macro": precision_score(y, pred, average="macro"),
            "Recall macro": recall_score(y, pred, average="macro"), "F1 macro": f1_score(y, pred, average="macro"),
            "AUC-ROC": roc_auc_score(y, proba, multi_class="ovr", average="macro")}
