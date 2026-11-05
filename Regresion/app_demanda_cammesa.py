# Regresión lineal sobre la demanda CAMMESA · aplicación Streamlit
# Para abrirla, desde la carpeta del cuaderno:  streamlit run app_demanda_cammesa.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Demanda mensual de un agente", layout="centered")
st.title("¿Cuántos MWh va a demandar este agente el mes que viene?")
st.caption("Regresión lineal sobre demanda-ltimos-aos.csv (CAMMESA), con rezagos de 1, 2 y 12 meses.")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_demanda_cammesa.joblib")

modelo = cargar_modelo()

st.subheader("Historia del agente")
c1, c2 = st.columns(2)
lag1 = c1.number_input("Demanda del mes anterior (MWh)", 0.0, 1_600_000.0, 50_000.0, step=1000.0)
lag2 = c1.number_input("Demanda de hace dos meses (MWh)", 0.0, 1_600_000.0, 48_000.0, step=1000.0)
tiene_lag12 = c1.checkbox("Conozco la demanda de hace un año", value=True)
lag12 = c1.number_input("Demanda del mismo mes del año pasado (MWh)", 0.0, 1_600_000.0, 52_000.0, step=1000.0, disabled=not tiene_lag12)
mes = c2.slider("Mes a predecir", 1, 12, 1)
tipo = c2.radio("Tipo de agente", ["Distribuidor · residencial", "Distribuidor · no residencial", "Gran usuario"])

fila = pd.DataFrame([{"lag1": lag1, "lag2": lag2, "lag12": lag12 if tiene_lag12 else np.nan, "mes_num": mes,
                      "es_gran_usuario": int(tipo == "Gran usuario"), "es_residencial": int(tipo == "Distribuidor · residencial")}])
estimacion = float(modelo.predict(fila)[0])

st.subheader("Predicción")
c3, c4 = st.columns(2)
c3.metric("Demanda estimada", f"{estimacion:,.0f} MWh", delta=f"{estimacion - lag1:+,.0f} vs. mes anterior")
c4.metric("Copiar el mes anterior daría", f"{lag1:,.0f} MWh")
if estimacion < 0:
    st.warning("La recta predice una demanda negativa: no tiene piso. Es uno de sus límites.")

curva = pd.concat([fila] * 200, ignore_index=True)
curva["lag1"] = np.linspace(0, max(lag1 * 2, 1000), 200)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(curva["lag1"], modelo.predict(curva), color="#c0392b", lw=2, label="predicción según la recta")
ax.plot(curva["lag1"], curva["lag1"], color="gray", ls=":", label="copiar el mes anterior")
ax.scatter([lag1], [estimacion], s=200, marker="*", color="#c0392b", zorder=5, label="este agente")
ax.set_xlabel("demanda del mes anterior (MWh)"); ax.set_ylabel("demanda predicha (MWh)"); ax.legend()
ax.set_title("Moviendo solo el mes anterior: la recta casi copia, con una corrección")
st.pyplot(fig)

with st.expander("Ver la fila que recibe el modelo"):
    st.dataframe(fila)
