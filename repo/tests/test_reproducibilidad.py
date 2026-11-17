"""Pruebas de reproducibilidad: python -m pytest tests -q   (o ejecutar este archivo directamente)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.config import ENTRADAS, RUTA_PROCESADO, COD
from src.modeling import entrenar, met_clf, met_reg
from src.preprocessing import cargar_procesado, preparar


def test_particiones_coinciden_con_la_fase_2():
    train, val, test = preparar()
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        guardado = pd.read_csv(RUTA_PROCESADO / f"{nombre}.csv", parse_dates=["date_time"])
        assert len(d) == len(guardado)
        pd.testing.assert_frame_equal(d.reset_index(drop=True).astype({"nivel_trafico": str}), guardado.astype({"nivel_trafico": str}), check_dtype=False, atol=1e-9)


def test_modelos_reproducen_la_fase_3():
    train, val, test = cargar_procesado(RUTA_PROCESADO)
    reg, clf = entrenar(train)
    r = met_reg(test["traffic_volume"], reg.predict(test[ENTRADAS]))
    c = met_clf(test["nivel_trafico"].map(COD).values, clf.predict(test[ENTRADAS]), clf.predict_proba(test[ENTRADAS]))
    assert abs(r["R²"] - 0.966) < 0.002 and abs(c["F1 macro"] - 0.927) < 0.004


if __name__ == "__main__":
    test_particiones_coinciden_con_la_fase_2(); print("OK particiones")
    test_modelos_reproducen_la_fase_3(); print("OK modelos")
