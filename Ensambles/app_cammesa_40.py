# Clase 40 sobre CAMMESA · boosting y apilamiento, contra los modelos anteriores · aplicación Streamlit
# Para abrirla, desde la carpeta del cuaderno:  streamlit run app_cammesa_40.py
import calendar
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PREDICTORAS = ["ratio_dias", "var_mes_ant", "volatilidad", "var_hace_un_anio", "log_tamano", "mes_num", "es_gran_usuario", "es_residencial"]
AZUL, ROJO, GRIS = "#0055a5", "#c0392b", "#6b7280"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

st.set_page_config(page_title="¿Va a caer fuerte la demanda?", layout="centered")
st.title("¿La demanda de este agente va a caer fuerte?")
st.caption("Los modelos de las unidades 6 y 7 sobre demanda-ltimos-aos.csv (CAMMESA): logística, vecinos, Bayes, árbol, bosque, potenciación del gradiente, por histogramas y apilado.")

# ---------- 1 · Los modelos guardados por los cuadernos ----------
MODELOS = {
    "regresión logística": ("modelo_cammesa_32.joblib", "Clase_32_regresion_logistica_y_softmax_CAMMESA.ipynb"),
    "k vecinos": ("modelo_cammesa_34_knn.joblib", "Clase_34_k_vecinos_y_Bayes_ingenuo_CAMMESA.ipynb"),
    "Bayes ingenuo": ("modelo_cammesa_34_bayes.joblib", "Clase_34_k_vecinos_y_Bayes_ingenuo_CAMMESA.ipynb"),
    "árbol de decisión": ("modelo_cammesa_35.joblib", "Clase_35_arboles_de_decision_CAMMESA.ipynb"),
    "bosque aleatorio": ("modelo_cammesa_38_bosque.joblib", "Clase_38_ensambles_por_promediado_CAMMESA.ipynb"),
    "potenciación del gradiente": ("modelo_cammesa_40_gradiente.joblib", "Clase_40_boosting_y_apilamiento_CAMMESA.ipynb"),
    "por histogramas": ("modelo_cammesa_40_histogramas.joblib", "Clase_40_boosting_y_apilamiento_CAMMESA.ipynb"),
    "apilado": ("modelo_cammesa_40_apilado.joblib", "Clase_40_boosting_y_apilamiento_CAMMESA.ipynb"),
}
PROPIOS = ["potenciación del gradiente", "por histogramas", "apilado"]

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
fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(probs)))
ax.barh(list(probs), list(probs.values()), color=[ROJO if p >= umbral else AZUL for p in probs.values()])
ax.axvline(umbral, color="black", ls="--", lw=1.5, label=f"umbral = {umbral:.2f}")
for k, p in enumerate(probs.values()): ax.text(min(p + 0.01, 0.9), k, f"{p:.3f}", va="center")
ax.set_xlim(0, 1); ax.set_xlabel("p̂(cae fuerte)"); ax.invert_yaxis(); ax.legend(loc="lower right")
plt.tight_layout(); st.pyplot(fig)

# ---------- 4 · Por qué: los modelos de esta clase ----------
st.subheader("Por qué")
pestana_gb, pestana_ap = st.tabs(["potenciación del gradiente", "apilado"])
with pestana_gb:
    m = modelos["potenciación del gradiente"]; gb = m[-1]
    tray = [p[0, 1] for p in gb.staged_predict_proba(m[:-1].transform(fila))]
    st.write(f"Arranca en la proporción de caídas del entrenamiento y cada uno de sus **{len(tray)}** árboles corrige un poco: termina en **{tray[-1]:.3f}**.")
    fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(range(1, len(tray) + 1), tray, color=ROJO, lw=2)
    ax.axhline(umbral, color="black", ls="--", lw=1); ax.set_xlabel("árboles sumados"); ax.set_ylabel("p̂"); plt.tight_layout(); st.pyplot(fig)
with pestana_ap:
    ap = modelos["apilado"]
    entradas = ap.transform(fila)[0]; nombres = list(ap.named_estimators_); coefs = ap.final_estimator_.coef_[0]
    st.write("El mezclador recibe la probabilidad de cada integrante y las combina con sus pesos (una logística):")
    st.dataframe(pd.DataFrame({"integrante": nombres, "p̂ que entrega": entradas.round(3), "peso en el mezclador": coefs.round(2)}), hide_index=True)
    st.write(f"Resultado del mezclador: **{probs['apilado']:.3f}**.")

with st.expander("Ver la fila que reciben los modelos"):
    st.dataframe(fila)
