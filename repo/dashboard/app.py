"""Dashboard interactivo – Proyecto Integrador (Metro Interstate Traffic Volume, autopista I-94).

Ejecutar desde la raíz del repositorio:  streamlit run dashboard/app.py
Módulos: filtros dinámicos (barra lateral), tarjetas KPI, narrativa de datos, predicción en vivo,
segmentación (clustering) y evidencia de los modelos. Los modelos se reentrenan al iniciar (≈ 30 s, en caché)
con los hiperparámetros de la Fase 3, de modo que no dependen de archivos serializados de otra versión.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import precision_recall_curve

from src.clustering import ajustar_segmentador, nombre_grupo
from src.config import COD, ENTRADAS, HORAS_PICO, NOMBRES_GRUPOS, ORDEN, RUTA_MODELOS, RUTA_PROCESADO
from src.modeling import entrenar, met_clf, met_reg
from src.preprocessing import cargar_procesado

st.set_page_config(page_title="Tráfico I-94 · Dashboard", layout="wide", page_icon=None)

st.markdown("""<style>
[data-testid="stMetricValue"]{font-size:1.65rem;}
[data-testid="stMetricValue"] > div{overflow:visible;text-overflow:clip;white-space:normal;}
[data-testid="stMetricLabel"] p{font-size:.85rem;}
[data-testid="stMetricDelta"]{font-size:.75rem;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ utilidades
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
PALETA = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
COLOR_NIVEL = {"bajo": "#54a24b", "medio": "#f2b134", "alto": "#e45756"}
TEAL, NARANJA = "#0e7c86", "#d9731a"


def es(x, dec=0):
    """Formato español: 1.234,5."""
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def estilo(fig, alto=380):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=50, b=10), separators=",.",
                      font=dict(size=14), legend=dict(orientation="h", y=-0.2))
    return fig


# ------------------------------------------------------------------ datos y modelos (en caché)
@st.cache_data(show_spinner=False)
def cargar_datos():
    tr, va, te = cargar_procesado(RUTA_PROCESADO)
    tr["partición"], va["partición"], te["partición"] = "Entrenamiento", "Validación", "Prueba"
    return tr, va, te


def _modelos_finales(tr):
    """Carga los modelos precalculados (models/dashboard/, arranque en segundos); si faltan o no son
    compatibles con las versiones instaladas, los reentrena con los datos de entrenamiento."""
    try:
        d = RUTA_MODELOS / "dashboard"
        return joblib.load(d / "regresor.joblib"), joblib.load(d / "clasificador.joblib"), joblib.load(d / "segmentador.joblib")
    except Exception:
        reg, clf = entrenar(tr)
        return reg, clf, ajustar_segmentador(tr)


@st.cache_resource(show_spinner="Preparando los modelos…")
def cargar_modelos():
    tr, va, te = cargar_datos()
    reg, clf, seg = _modelos_finales(tr)
    p_reg = reg.predict(te[ENTRADAS])
    proba = clf.predict_proba(te[ENTRADAS])
    y_clf = te["nivel_trafico"].astype(str).map(COD).values
    resid = te["traffic_volume"].values - p_reg
    pico = te["hora"].isin(HORAS_PICO).values
    return dict(reg=reg, clf=clf, seg=seg, p_reg=p_reg, proba=proba, y_clf=y_clf,
                m_reg=met_reg(te["traffic_volume"], p_reg), m_clf=met_clf(y_clf, proba.argmax(1), proba),
                q_pico=np.percentile(resid[pico], [10, 90]), q_resto=np.percentile(resid[~pico], [10, 90]))


@st.cache_data(show_spinner=False)
def datos_completos():
    tr, va, te = cargar_datos()
    M = cargar_modelos()
    df = pd.concat([tr, va, te], ignore_index=True)
    df["tipo_dia"] = np.where(df["es_festivo"] == 1, "Festivo", np.where(df["dia_semana"] >= 5, "Fin de semana", "Laboral"))
    df["grupo"] = [nombre_grupo(i) for i in M["seg"].etiquetar(df)]
    pcs = M["seg"].proyectar(df)
    df["pc1"], df["pc2"], df["pc3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    df["precipitación"] = np.where((df["rain_1h"] > 0) | (df["snow_1h"] > 0), "Sí", "No")
    return df


@st.cache_data(show_spinner=False)
def resultados():
    return json.loads((RUTA_MODELOS / "resultados_fase3.json").read_text(encoding="utf-8"))


M = cargar_modelos()
df = datos_completos()
R3 = resultados()

# ------------------------------------------------------------------ filtros dinámicos
st.sidebar.header("Filtros dinámicos")
f_min, f_max = df["date_time"].min().date(), df["date_time"].max().date()
rango = st.sidebar.date_input("Rango de fechas", (f_min, f_max), min_value=f_min, max_value=f_max)
if isinstance(rango, tuple) and len(rango) == 2:
    f_ini, f_fin = rango
else:
    f_ini, f_fin = f_min, f_max
climas = sorted(df["weather_main"].unique())
sel_clima = st.sidebar.multiselect("Tipo de clima", climas, default=climas)
sel_dias = st.sidebar.multiselect("Día de la semana", DIAS, default=DIAS)
sel_tipo = st.sidebar.multiselect("Tipo de día", ["Laboral", "Fin de semana", "Festivo"], default=["Laboral", "Fin de semana", "Festivo"])
h_ini, h_fin = st.sidebar.slider("Hora del día", 0, 23, (0, 23))
t_min, t_max = float(df["temp"].min()), float(df["temp"].max())
temp_rng = st.sidebar.slider("Temperatura (°C)", float(np.floor(t_min)), float(np.ceil(t_max)), (float(np.floor(t_min)), float(np.ceil(t_max))))
nub_rng = st.sidebar.slider("Nubosidad (%)", 0, 100, (0, 100))
solo_precip = st.sidebar.checkbox("Solo horas con lluvia o nieve", value=False)
v_max = int(df["traffic_volume"].max())
vol_rng = st.sidebar.slider("Volumen de tráfico (vehículos/h)", 0, v_max, (0, v_max), step=100)
sel_part = st.sidebar.multiselect("Partición de los datos", ["Entrenamiento", "Validación", "Prueba"], default=["Entrenamiento", "Validación", "Prueba"],
                                  help="Los modelos se entrenaron con 'Entrenamiento' y se evaluaron en 'Prueba'.")

mask = (df["date_time"].dt.date.between(f_ini, f_fin) & df["weather_main"].isin(sel_clima) & df["dia_semana"].map(lambda i: DIAS[i]).isin(sel_dias)
        & df["tipo_dia"].isin(sel_tipo) & df["hora"].between(h_ini, h_fin) & df["temp"].between(*temp_rng) & df["clouds_all"].between(*nub_rng)
        & df["traffic_volume"].between(*vol_rng) & df["partición"].isin(sel_part))
if solo_precip:
    mask &= df["precipitación"] == "Sí"
dff = df[mask]
st.sidebar.caption(f"{es(len(dff))} de {es(len(df))} horas cumplen los filtros.")
if st.sidebar.button("Restablecer filtros"):
    st.rerun()

# ------------------------------------------------------------------ encabezado y KPI
st.title("Tráfico en la autopista I-94: qué pasa, cuánto y cuándo")
st.caption("Proyecto Integrador · Ciencia de Datos · Universidad de La Salle · Datos: Metro Interstate Traffic Volume (UCI; Hogue, 2019)")

if dff.empty:
    st.warning("Ningún registro cumple los filtros actuales. Amplíe los rangos en la barra lateral.")
    st.stop()

media_total = df["traffic_volume"].mean()
por_hora = dff.groupby("hora")["traffic_volume"].mean()
te_mask_idx = (dff["partición"] == "Prueba")
te_all = df[df["partición"] == "Prueba"]
mae_filtro = None
if te_mask_idx.any():
    idx_te = te_all.index.get_indexer(dff.index[te_mask_idx])
    mae_filtro = float(np.abs(te_all["traffic_volume"].values[idx_te] - M["p_reg"][idx_te]).mean())

k = st.columns(6)
with k[0].container(border=True):
    st.metric("Horas analizadas", es(len(dff)), f"{es(len(dff) / len(df) * 100, 1)} % del total", delta_color="off", delta_arrow="off")
with k[1].container(border=True):
    st.metric("Tráfico medio (veh/h)", es(dff["traffic_volume"].mean()), f"total: {es(media_total)} veh/h", delta_color="off", delta_arrow="off")
with k[2].container(border=True):
    st.metric("Hora de mayor tráfico", f"{int(por_hora.idxmax())}:00", f"{es(por_hora.max())} veh/h", delta_color="off", delta_arrow="off")
with k[3].container(border=True):
    st.metric("Horas con nivel alto", f"{es((dff['nivel_trafico'] == 'alto').mean() * 100, 1)} %", "≥ 4.500 veh/h", delta_color="off", delta_arrow="off")
with k[4].container(border=True):
    st.metric("Error medio del modelo (veh/h)", es(mae_filtro) if mae_filtro is not None else "s/d",
              "MAE en prueba, con los filtros" if mae_filtro is not None else "sin horas de prueba", delta_color="off", delta_arrow="off")
with k[5].container(border=True):
    st.metric("R² · F1 macro (prueba)", f"{es(M['m_reg']['R²'], 3)} · {es(M['m_clf']['F1 macro'], 3)}", "XGBoost, prueba completa", delta_color="off", delta_arrow="off")

tab_n, tab_p, tab_s, tab_m, tab_a = st.tabs(["Narrativa de datos", "Predicción en vivo", "Segmentación", "Modelos y evidencia", "Acerca del proyecto"])

# ------------------------------------------------------------------ 1. NARRATIVA
with tab_n:
    st.subheader("1. El tráfico sigue el reloj y el calendario")
    curva = dff.groupby(["tipo_dia", "hora"])["traffic_volume"].mean().reset_index()
    fig = px.line(curva, x="hora", y="traffic_volume", color="tipo_dia", markers=True,
                  color_discrete_map={"Laboral": TEAL, "Fin de semana": NARANJA, "Festivo": "#8d6e63"},
                  labels={"hora": "Hora del día", "traffic_volume": "Tráfico medio (veh/h)", "tipo_dia": "Tipo de día"})
    st.plotly_chart(estilo(fig), width="stretch")
    txt = []
    for t in ["Laboral", "Fin de semana", "Festivo"]:
        s = curva[curva["tipo_dia"] == t]
        if not s.empty:
            r = s.loc[s["traffic_volume"].idxmax()]
            txt.append(f"en días **{t.lower()}s** el máximo es de **{es(r['traffic_volume'])} veh/h a las {int(r['hora'])}:00**")
    st.info("Con los filtros actuales, " + "; ".join(txt) + ". Los días laborales muestran dos picos (mañana y tarde); los fines de semana, una sola joroba a media tarde.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2. Lo que más pesa en la predicción")
        sh = pd.DataFrame(R3["shap"]["tabla"]).rename(columns={"index": "variable", "% de la importancia SHAP": "pct"}).sort_values("pct")
        fig = px.bar(sh, x="pct", y="variable", orientation="h", color_discrete_sequence=[TEAL], labels={"pct": "% de la importancia SHAP", "variable": ""})
        st.plotly_chart(estilo(fig, 360), width="stretch")
        hora_pct = float(sh.loc[sh["variable"] == "hora", "pct"].iloc[0])
        dia_pct = float(sh.loc[sh["variable"] == "dia_semana", "pct"].iloc[0])
        clima_pct = float(sh.loc[sh["variable"].isin(["temp", "weather_main", "clouds_all", "rain_1h", "snow_1h"]), "pct"].sum())
        st.info(f"La **hora** explica el **{es(hora_pct, 1)} %** de la atribución, el **día de la semana** el **{es(dia_pct, 1)} %** y todo el **clima** junto, el **{es(clima_pct, 1)} %**. "
                "El tráfico de esta vía se predice sobre todo por calendario.")
    with c2:
        st.subheader("3. Mapa de calor: hora × día de la semana")
        mc = dff.pivot_table(index="dia_semana", columns="hora", values="traffic_volume", aggfunc="mean").reindex(range(7))
        mc.index = DIAS
        fig = px.imshow(mc, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(x="Hora", y="", color="veh/h"))
        st.plotly_chart(estilo(fig, 360), width="stretch")
        st.info("Las horas más oscuras (rojo) son las de mayor demanda: tardes y mañanas de lunes a viernes. Los sábados y domingos se mantienen en tonos claros.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("4. Estacionalidad mensual")
        mes = dff.groupby("mes")["traffic_volume"].mean().reindex(range(1, 13)).reset_index()
        mes["mes_n"] = MESES
        fig = px.bar(mes, x="mes_n", y="traffic_volume", color_discrete_sequence=[NARANJA], labels={"mes_n": "", "traffic_volume": "Tráfico medio (veh/h)"})
        st.plotly_chart(estilo(fig, 330), width="stretch")
    with c4:
        st.subheader("5. Clima: ¿cambia el tráfico?")
        cl = dff.groupby("weather_main")["traffic_volume"].agg(["mean", "size"]).reset_index().sort_values("mean")
        fig = px.bar(cl, x="mean", y="weather_main", orientation="h", hover_data={"size": True}, color_discrete_sequence=[TEAL],
                     labels={"mean": "Tráfico medio (veh/h)", "weather_main": "", "size": "horas"})
        st.plotly_chart(estilo(fig, 330), width="stretch")
    st.caption("Las diferencias por clima se ven pequeñas frente a las de la hora o el día, y mezclan horas del día distintas; por eso el modelo les da poco peso (gráfico 2).")

    st.subheader("6. De la historia a la acción")
    acciones = pd.DataFrame({
        "Grupo (segmentación)": NOMBRES_GRUPOS,
        "Qué lo caracteriza": ["Entre semana, de 19 h a 5 h; tráfico más bajo (≈ 1.600 veh/h)", "Todas las horas de los festivos; tráfico intermedio-bajo",
                               "Sábados y domingos sin festivo ni precipitación", "Todas las horas con lluvia o nieve; mezcla horas de todo el día",
                               "Entre semana, de 6 h a 18 h; la mayor demanda (≈ 5.300 veh/h)"],
        "Acción sugerida (propuesta del equipo, no validada con la operación)": [
            "Ventana natural para obras y mantenimiento", "Planear la operación con calendario propio de festivos", "Operación estándar con picos de media tarde",
            "Activar protocolos de clima y reforzar la información al conductor", "Concentrar la información al conductor y la respuesta a incidentes"]})
    st.dataframe(acciones, hide_index=True, width="stretch")
    st.caption("Aviso de cobertura: el 22,8 % de las horas del periodo no tiene registro (sobre todo 2014 y 2015); los resultados describen las horas disponibles.")

# ------------------------------------------------------------------ 2. PREDICCIÓN EN VIVO
with tab_p:
    st.subheader("Simulador de predicción")
    st.caption("Ingrese las condiciones de una hora y obtenga al instante el volumen esperado, el nivel de tráfico y el grupo de operación (modelos XGBoost entrenados en la Fase 3).")
    izq, der = st.columns([1, 1.4])
    with izq:
        with st.form("form_pred"):
            fecha = st.date_input("Fecha", pd.Timestamp("2018-06-15").date(), min_value=pd.Timestamp("2012-01-01").date(), max_value=pd.Timestamp("2030-12-31").date())
            hora = st.slider("Hora del día", 0, 23, 17)
            auto_fest = bool(len(USFederalHolidayCalendar().holidays(start=str(fecha), end=str(fecha))) > 0)
            festivo = st.checkbox("Es festivo", value=auto_fest, help="Sugerido con el calendario de festivos federales de EE. UU.; puede cambiarlo.")
            clima = st.selectbox("Tipo de clima", climas, index=climas.index("Clear") if "Clear" in climas else 0)
            temp = st.slider("Temperatura (°C)", -30.0, 40.0, 18.0, 0.5)
            nub = st.slider("Nubosidad (%)", 0, 100, 20)
            lluvia = st.number_input("Lluvia en la última hora (mm)", 0.0, 100.0, 0.0, 0.5)
            nieve = st.number_input("Nieve en la última hora (mm)", 0.0, 5.0, 0.0, 0.1)
            enviar = st.form_submit_button("Predecir", type="primary")
    ts = pd.Timestamp(fecha)

    def fila(h):
        return pd.DataFrame({"rain_1h": [lluvia], "snow_1h": [nieve], "temp": [temp], "clouds_all": [nub], "weather_main": [clima], "hora": [h],
                             "dia_semana": [ts.dayofweek], "mes": [ts.month], "es_festivo": [int(festivo)]})

    x = fila(hora)
    pred = float(max(M["reg"].predict(x[ENTRADAS])[0], 0))
    proba = M["clf"].predict_proba(x[ENTRADAS])[0]
    nivel = ORDEN[int(proba.argmax())]
    q = M["q_pico"] if hora in HORAS_PICO else M["q_resto"]
    lo, hi = max(pred + q[0], 0), pred + q[1]
    xc = x.assign(traffic_volume=pred)
    grupo = nombre_grupo(int(M["seg"].etiquetar(xc)[0]))
    hist = df[(df["hora"] == hora) & (df["tipo_dia"] == ("Festivo" if festivo else "Fin de semana" if ts.dayofweek >= 5 else "Laboral"))]["traffic_volume"].mean()
    with der:
        a, b, c = st.columns(3)
        with a.container(border=True):
            st.metric("Volumen esperado (veh/h)", es(pred), f"{'+' if pred >= hist else '−'}{es(abs(pred - hist))} vs. promedio de esa hora", delta_color="off", delta_arrow="off")
        with b.container(border=True):
            st.metric("Nivel de tráfico", nivel.upper(), f"{es(proba.max() * 100, 1)} % de probabilidad", delta_color="off", delta_arrow="off")
        with c.container(border=True):
            st.caption("Grupo de operación")
            st.markdown(f"#### {grupo}")
        st.caption(f"Intervalo empírico del 80 % para el volumen (según los errores del modelo en prueba en {'horas pico' if hora in HORAS_PICO else 'horas fuera de pico'}): "
                   f"**{es(lo)} a {es(hi)} veh/h**. El grupo se asigna con el volumen predicho como tráfico.")
        fig = go.Figure(go.Bar(x=[n.capitalize() for n in ORDEN], y=proba * 100, marker_color=[COLOR_NIVEL[n] for n in ORDEN],
                               text=[f"{es(p * 100, 1)} %" for p in proba], textposition="outside"))
        fig.update_layout(title="Probabilidad por nivel de tráfico", yaxis=dict(title="%", range=[0, 110]))
        st.plotly_chart(estilo(fig, 280), width="stretch")
    dia = pd.concat([fila(h) for h in range(24)], ignore_index=True)
    dia_pred = np.maximum(M["reg"].predict(dia[ENTRADAS]), 0)
    fig = go.Figure()
    fig.add_hline(y=4500, line_dash="dot", line_color="#e45756", annotation_text="umbral nivel alto (4.500)")
    fig.add_hline(y=1500, line_dash="dot", line_color="#54a24b", annotation_text="umbral nivel bajo (1.500)")
    fig.add_trace(go.Scatter(x=list(range(24)), y=dia_pred, mode="lines+markers", line=dict(color=TEAL), name="Predicción para ese día"))
    fig.add_trace(go.Scatter(x=[hora], y=[pred], mode="markers", marker=dict(size=16, color=NARANJA), name="Hora consultada"))
    fig.update_layout(title="Perfil de las 24 horas con las mismas condiciones", xaxis_title="Hora", yaxis_title="Volumen (veh/h)")
    st.plotly_chart(estilo(fig, 360), width="stretch")
    st.caption("Limitación: el modelo usa calendario y clima de esa hora; no conoce incidentes, obras ni el tráfico de las horas anteriores.")

# ------------------------------------------------------------------ 3. SEGMENTACIÓN
with tab_s:
    st.subheader("Segmentación de las horas de operación (K-means, K = 5)")
    modo = st.radio("Proyección", ["2D (PCA)", "3D (PCA)"], horizontal=True)
    muestra = dff.sample(min(len(dff), 5000), random_state=42)
    cmap = {n: PALETA[i] for i, n in enumerate(NOMBRES_GRUPOS)}
    hov = {"hora": True, "traffic_volume": True, "tipo_dia": True, "pc1": False, "pc2": False, "pc3": False}
    if modo.startswith("2D"):
        fig = px.scatter(muestra, x="pc1", y="pc2", color="grupo", color_discrete_map=cmap, category_orders={"grupo": NOMBRES_GRUPOS},
                         hover_data=hov, opacity=0.6, labels={"pc1": "Componente 1", "pc2": "Componente 2", "grupo": "Grupo", "traffic_volume": "veh/h"})
    else:
        fig = px.scatter_3d(muestra, x="pc1", y="pc2", z="pc3", color="grupo", color_discrete_map=cmap, category_orders={"grupo": NOMBRES_GRUPOS},
                            hover_data=hov, opacity=0.6, labels={"pc1": "C1", "pc2": "C2", "pc3": "C3", "grupo": "Grupo"})
    st.plotly_chart(estilo(fig, 560), width="stretch")
    st.caption(f"Muestra de {es(len(muestra))} horas de las {es(len(dff))} filtradas. Silueta del modelo: {es(R3['clusters']['silueta_train'], 3)} en entrenamiento y "
               f"{es(R3['clusters']['silueta_val'], 3)} en validación (estructura débil a moderada: los grupos son regímenes de calendario con bordes difusos).")
    perf = dff.groupby("grupo").agg(**{"% de horas": ("hora", lambda s: len(s) / len(dff) * 100), "tráfico medio (veh/h)": ("traffic_volume", "mean"),
                                       "hora mediana": ("hora", "median"), "% fin de semana": ("tipo_dia", lambda s: (s == "Fin de semana").mean() * 100),
                                       "% festivo": ("es_festivo", lambda s: s.mean() * 100), "temp. media (°C)": ("temp", "mean"),
                                       "% con precipitación": ("precipitación", lambda s: (s == "Sí").mean() * 100)}).reindex(NOMBRES_GRUPOS).dropna(how="all")
    st.markdown("**Perfil numérico promedio de cada grupo (con los filtros actuales)**")
    st.dataframe(perf.style.format({c: (lambda v: es(v, 1)) for c in perf.columns}), width="stretch")
    comp = pd.crosstab(dff["grupo"], dff["hora"], normalize="index").reindex(NOMBRES_GRUPOS).dropna(how="all") * 100
    fig = px.imshow(comp, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Hora", y="", color="% del grupo"), title="¿A qué horas ocurre cada grupo?")
    st.plotly_chart(estilo(fig, 320), width="stretch")

# ------------------------------------------------------------------ 4. MODELOS Y EVIDENCIA
with tab_m:
    st.subheader("Evidencia de los modelos (conjunto de prueba, evaluado una sola vez)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Regresión del volumen**")
        tr_ = pd.DataFrame(R3["test_reg"]).rename(columns={"index": "Modelo"}).set_index("Modelo")
        st.dataframe(tr_.style.format({"R²": lambda v: es(v, 3), "RMSE": lambda v: es(v, 1), "MAE": lambda v: es(v, 1)}), width="stretch")
    with c2:
        st.markdown("**Clasificación del nivel**")
        tc_ = pd.DataFrame(R3["test_clf"]).rename(columns={"index": "Modelo"}).set_index("Modelo")
        st.dataframe(tc_.style.format({c: (lambda v: es(v, 3)) for c in tc_.columns}), width="stretch")
    ic = pd.DataFrame(R3["ic_bootstrap"]).rename(columns={"index": "Métrica"}).set_index("Métrica")
    st.markdown("**Intervalos de confianza del 95 % (bootstrap)** · la mejora sobre el baseline no incluye el cero")
    st.dataframe(ic.style.format({c: (lambda v: es(v, 3)) for c in ic.columns}), width="stretch")

    st.markdown("**Sobreajuste**: brecha entre entrenamiento y validación")
    b1, b2 = st.columns(2)
    vr = pd.DataFrame(R3["val_reg"]).rename(columns={"index": "Modelo"}).set_index("Modelo")[["R² val.", "R² entren.", "brecha R²"]]
    vc = pd.DataFrame(R3["val_clf"]).rename(columns={"index": "Modelo"}).set_index("Modelo")[["F1 macro", "F1 entren.", "brecha F1"]]
    b1.dataframe(vr.style.format(lambda v: es(v, 3)), width="stretch")
    b2.dataframe(vc.style.format(lambda v: es(v, 3)), width="stretch")

    st.markdown("**Pérdida del MLP en PyTorch (entrenamiento vs. validación)**")
    cu = R3["curvas_mlp"]
    p1, p2 = st.columns(2)
    for col, clave, tit, mejor in [(p1, "reg", "MLP de regresión (MSE)", cu["mejor_epoca_reg"]), (p2, "clf", "MLP de clasificación (entropía cruzada)", cu["mejor_epoca_clf"])]:
        h = pd.DataFrame(cu[clave])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["época"], y=h["entrenamiento"], name="entrenamiento"))
        fig.add_trace(go.Scatter(x=h["época"], y=h["validación"], name="validación"))
        fig.add_vline(x=mejor, line_dash="dash", line_color="gray", annotation_text=f"mejor época ({mejor})")
        fig.update_layout(title=tit, xaxis_title="época", yaxis_title="pérdida")
        col.plotly_chart(estilo(fig, 330), width="stretch")

    st.markdown("**Impacto operativo: error absoluto medio en prueba**")
    op = pd.DataFrame(R3["operativo"]["tabla_op"]).set_index("grupo")
    st.dataframe(op.style.format({"horas": lambda v: es(v), "MAE baseline": lambda v: es(v, 1), "MAE seleccionado": lambda v: es(v, 1),
                                  "reducción del MAE (%)": lambda v: es(v, 1)}), width="stretch")

    st.markdown("**Simulador de alertas de congestión alta**")
    st.caption("Decisión operativa: emitir una alerta cuando la probabilidad de nivel *alto* supera un umbral. Un umbral bajo detecta más congestiones, pero genera más falsas alarmas.")
    umbral = st.slider("Umbral de alerta (probabilidad de nivel alto)", 0.05, 0.95, 0.50, 0.05)
    y_alto = (M["y_clf"] == COD["alto"])
    p_alto = M["proba"][:, COD["alto"]]
    alerta = p_alto >= umbral
    tp, fp, fn = int((alerta & y_alto).sum()), int((alerta & ~y_alto).sum()), int((~alerta & y_alto).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Congestiones detectadas (recall)", f"{es(rec * 100, 1)} %")
    q2.metric("Alertas correctas (precisión)", f"{es(prec * 100, 1)} %")
    q3.metric("Falsas alarmas", es(fp), f"de {es(tp + fp)} alertas", delta_color="off", delta_arrow="off")
    q4.metric("Congestiones no detectadas", es(fn), f"de {es(tp + fn)} horas altas", delta_color="off", delta_arrow="off")
    pr, rc, _ = precision_recall_curve(y_alto, p_alto)
    fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=TEAL), name="curva precisión–recall"))
    fig.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", marker=dict(size=14, color=NARANJA), name="umbral elegido"))
    fig.update_layout(title="Compromiso entre detectar congestiones y evitar falsas alarmas", xaxis_title="Recall", yaxis_title="Precisión")
    st.plotly_chart(estilo(fig, 340), width="stretch")

# ------------------------------------------------------------------ 5. ACERCA
with tab_a:
    st.subheader("Acerca del proyecto")
    st.markdown("""
**Problema.** Predecir el volumen horario de tráfico de la autopista I-94 (Minneapolis–St. Paul), clasificar su nivel (bajo, medio, alto) y segmentar las horas de operación.

**Datos.** *Metro Interstate Traffic Volume* (UCI; Hogue, 2019): 48.204 filas horarias de 2012 a 2018; tras la limpieza quedan 40.575 horas únicas. El 22,8 % de las horas del periodo no tiene registro.

**Fases.**
1. *EDA*: el tráfico depende del calendario más que del clima.
2. *Preprocesamiento y modelo base*: limpieza, valores imposibles, variables de calendario, partición 70/15/15 sin fuga de datos; base con R² de 0,838 y F1 de 0,891 (validación).
3. *Modelado avanzado*: Ridge con interacciones, XGBoost, Random Forest, MLP en PyTorch y K-means; XGBoost llega a R² de 0,966 y F1 macro de 0,927 en prueba.
4. *Dashboard* (esta aplicación) e informe técnico final.

**Limitaciones.** Partición aleatoria; serie con vacíos y sin rezagos del tráfico; silueta moderada; el MLP se afinó con una sola semilla. El simulador no conoce incidentes ni obras.

**Reproducibilidad.** Código en `src/`, cuadernos en `notebooks/`, versiones exactas en `requirements.txt`, semilla global 42.

**Autores.** Tony Yu, Luis Rodríguez y Maydda Rodríguez · Docente: Fabián Camilo Castro Riveros · Universidad de La Salle, Maestría en Inteligencia Artificial.
""")
