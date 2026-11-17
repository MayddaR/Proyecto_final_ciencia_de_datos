"""Entrena los modelos finales con train.csv y los guarda en models/dashboard/ para que el dashboard arranque rápido.

Uso (desde la raíz del repositorio):  python -m src.exportar_modelos
Los archivos dependen de las versiones de scikit-learn y xgboost de requirements.txt; si no cargan, el
dashboard vuelve a entrenar solo (más lento).
"""
import json
import joblib

from .clustering import ajustar_segmentador
from .config import RUTA_MODELOS, RUTA_PROCESADO
from .modeling import entrenar
from .preprocessing import cargar_procesado

DESTINO = RUTA_MODELOS / "dashboard"


def main():
    import sklearn, xgboost
    tr, _, _ = cargar_procesado(RUTA_PROCESADO)
    reg, clf = entrenar(tr)
    seg = ajustar_segmentador(tr)
    DESTINO.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, DESTINO / "regresor.joblib", compress=3)
    joblib.dump(clf, DESTINO / "clasificador.joblib", compress=3)
    joblib.dump(seg, DESTINO / "segmentador.joblib", compress=3)
    (DESTINO / "versiones.json").write_text(json.dumps({"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}), encoding="utf-8")
    print("Modelos guardados en", DESTINO)


if __name__ == "__main__":
    main()
