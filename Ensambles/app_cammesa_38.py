# Clase 38 sobre CAMMESA · bosque aleatorio, contra los modelos de la unidad 6 · aplicación Streamlit
# Para abrirla, desde la carpeta del cuaderno:  streamlit run app_cammesa_38.py
import calendar
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PREDICTORAS = ["ratio_dias", "var_mes_ant", "volatilidad", "var_hace_un_anio", "log_tamano", "mes_num", "es_gran_usuario", "es_residencial"]
AZUL, ROJO = "#0055a5", "#c0392b"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

st.set_page_config(page_title="¿Va a caer fuerte la demanda?", layout="centered")
st.title("¿La demanda de este agente va a caer fuerte?")
st.caption("Logística (clase 32), k vecinos y Bayes ingenuo (clase 34), árbol (clase 35) y bosque aleatorio (clase 38), sobre demanda-ltimos-aos.csv (CAMMESA).")

# ---------- 1 · Los modelos guardados por los cuadernos ----------
MODELOS = {
    "regresión logística": ("modelo_cammesa_32.joblib", "Clase_32_regresion_logistica_y_softmax_CAMMESA.ipynb"),
    "k vecinos": ("modelo_cammesa_34_knn.joblib", "Clase_34_k_vecinos_y_Bayes_ingenuo_CAMMESA.ipynb"),
    "Bayes ingenuo": ("modelo_cammesa_34_bayes.joblib", "Clase_34_k_vecinos_y_Bayes_ingenuo_CAMMESA.ipynb"),
    "árbol de decisión": ("modelo_cammesa_35.joblib", "Clase_35_arboles_de_decision_CAMMESA.ipynb"),
    "bosque aleatorio": ("modelo_cammesa_38_bosque.joblib", "Clase_38_ensambles_por_promediado_CAMMESA.ipynb"),
}
PROPIOS = ["bosque aleatorio"]

@st.cache_resource
def cargar(archivo):
    return joblib.load(archivo)

modelos = {}
for nombre, (archivo, cuaderno) in MODELOS.items():
    if Path(archivo).exists():
        modelos[nombre] = cargar(archivo)
    elif nombre in PROPIOS:
        st.error(f"Falta {archivo}. Ejecutar antes el cuaderno {cuaderno}, que lo genera."); st.stop()
    else:
        st.warning(f"Falta {archivo} (lo genera {cuaderno}): la comparación se muestra sin {nombre}.")
config = json.loads(Path("umbral_cammesa_32.json").read_text(encoding="utf-8")) if Path("umbral_cammesa_32.json").exists() else {"umbral": 0.15}

# ---------- 2 · La historia del agente ----------
st.subheader("Historia del agente")
c1, c2 = st.columns(2)
lag1 = c1.number_input("Demanda del mes anterior (MWh)", 1.0, 1_600_000.0, 5_000.0, step=100.0)
lag2 = c1.number_input("Demanda de hace dos meses (MWh)", 0.0, 1_600_000.0, 4_800.0, step=100.0)
conoce_anio = c1.checkbox("Conozco el año pasado", value=True)
lag12 = c1.number_input("Hace un año: el mismo mes (MWh)", 0.0, 1_600_000.0, 5_200.0, step=100.0, disabled=not conoce_anio)
lag13 = c1.number_input("Hace un año: el mes anterior (MWh)", 0.0, 1_600_000.0, 5_000.0, step=100.0, disabled=not conoce_anio)
anio = c2.number_input("Año", 2017, 2030, 2020)
mes = MESES.index(c2.selectbox("Mes a predecir", MESES, index=1)) + 1
tipo = c2.radio("Tipo de agente", ["Distribuidor · residencial", "Distribuidor · no residencial", "Gran usuario"])
umbral = c2.slider("Umbral de decisión", 0.05, 0.60, float(config["umbral"]), step=0.01)

anio_ant, mes_ant = (anio - 1, 12) if mes == 1 else (anio, mes - 1)
var_mes_ant = float(np.clip(lag1 / lag2 - 1, -1, 3)) if lag2 > 0 else np.nan
var_anio = float(np.clip(lag12 / lag13 - 1, -1, 3)) if conoce_anio and lag13 > 0 else np.nan
fila = pd.DataFrame([{"ratio_dias": calendar.monthrange(anio, mes)[1] / calendar.monthrange(anio_ant, mes_ant)[1],
                      "var_mes_ant": var_mes_ant, "volatilidad": abs(var_mes_ant) if not np.isnan(var_mes_ant) else np.nan,
                      "var_hace_un_anio": var_anio, "log_tamano": np.log10(lag1), "mes_num": mes,
                      "es_gran_usuario": int(tipo == "Gran usuario"), "es_residencial": int(tipo == "Distribuidor · residencial")}])[PREDICTORAS]

# ---------- 3 · Probabilidad y decisión de cada modelo ----------
probs = {nombre: float(m.predict_proba(fila)[0, 1]) for nombre, m in modelos.items()}
st.subheader("Predicción")
fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(probs)))
ax.barh(list(probs), list(probs.values()), color=[ROJO if p >= umbral else AZUL for p in probs.values()])
ax.axvline(umbral, color="black", ls="--", lw=1.5, label=f"umbral = {umbral:.2f}")
for k, p in enumerate(probs.values()): ax.text(min(p + 0.01, 0.9), k, f"{p:.3f}", va="center")
ax.set_xlim(0, 1); ax.set_xlabel("p̂(cae fuerte)"); ax.invert_yaxis(); ax.legend(loc="lower right")
plt.tight_layout(); st.pyplot(fig)
p_b = probs["bosque aleatorio"]
st.markdown(f"**Bosque:** {p_b * 100:.1f} % → :{'red' if p_b >= umbral else 'blue'}[**{'alarma: puede caer fuerte' if p_b >= umbral else 'sin alarma'}**]".replace(".", ",", 1))

# ---------- 4 · Por qué ----------
st.subheader("Por qué")
pestanas = dict(zip(modelos, st.tabs(list(modelos))))
with pestanas["bosque aleatorio"]:
    bosque = modelos["bosque aleatorio"]
    votos = np.array([a.predict_proba(fila.to_numpy())[0, 1] for a in bosque.estimators_])
    st.write(f"El bosque promedia lo que dicen sus **{len(votos)}** árboles: **{(votos >= umbral).sum()}** de ellos le dan una probabilidad mayor o igual al umbral. "
             f"El 80 % central va de {np.percentile(votos, 10):.2f} a {np.percentile(votos, 90):.2f}.")
    fig, ax = plt.subplots(figsize=(8, 3)); ax.hist(votos, bins=40, color=ROJO, alpha=0.7)
    ax.axvline(votos.mean(), color="black", lw=2, label=f"promedio = {votos.mean():.3f}"); ax.axvline(umbral, color="black", ls="--", lw=1, label="umbral")
    ax.set_xlabel("p̂ de cada árbol"); ax.set_ylabel("árboles"); ax.legend(); plt.tight_layout(); st.pyplot(fig)
if "árbol de decisión" in modelos:
    with pestanas["árbol de decisión"]:
        arbol = modelos["árbol de decisión"]; t = arbol.tree_; nodos = arbol.decision_path(fila).indices
        for nodo in nodos[:-1]:
            j = t.feature[nodo]; col, u, v = PREDICTORAS[j], t.threshold[nodo], fila.iloc[0, j]
            st.write(f"• **{col}** falta → {'≤' if t.missing_go_to_left[nodo] else '>'} {u:.3g}" if pd.isna(v) else f"• **{col}** = {v:.3g} {'≤' if v <= u else '>'} {u:.3g}")
        hoja = nodos[-1]; cuentas = (t.value[hoja][0] * t.weighted_n_node_samples[hoja]).round().astype(int)
        st.write(f"→ de las {cuentas.sum()} filas de entrenamiento de esa hoja, **{cuentas[1]}** cayeron.")
if "k vecinos" in modelos:
    with pestanas["k vecinos"]:
        m = modelos["k vecinos"]; knn = m[-1]; dist, pos = knn.kneighbors(m[:-1].transform(fila)); votos_k = knn._y[pos[0]]
        st.write(f"De sus **{knn.n_neighbors}** vecinos más parecidos, **{int(votos_k.sum())}** cayeron fuerte.")
for nombre in ["Bayes ingenuo", "regresión logística"]:
    if nombre in modelos:
        with pestanas[nombre]:
            m = modelos[nombre]; x = m[:-1].transform(fila)[0]; final = m[-1]
            if nombre == "Bayes ingenuo":
                def logpdf(v, mu, var): return -0.5 * np.log(2 * np.pi * var) - (v - mu) ** 2 / (2 * var)
                aportes = logpdf(x, final.theta_[1], final.var_[1]) - logpdf(x, final.theta_[0], final.var_[0])
            else:
                aportes = final.coef_[0] * x
            fig, ax = plt.subplots(figsize=(8, 3.4)); ax.barh(PREDICTORAS, aportes, color=[ROJO if v > 0 else AZUL for v in aportes]); ax.axvline(0, color="black", lw=0.8); ax.invert_yaxis()
            ax.set_xlabel("aporte a las log-chances (rojo: hacia la caída)"); plt.tight_layout(); st.pyplot(fig)

with st.expander("Ver la fila que reciben los modelos"):
    st.dataframe(fila)
