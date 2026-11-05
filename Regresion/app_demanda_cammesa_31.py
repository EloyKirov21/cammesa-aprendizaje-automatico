# Clase 31 sobre CAMMESA · polinómica + Lasso contra la recta · aplicación Streamlit
# Para abrirla, desde la carpeta del cuaderno:  streamlit run app_demanda_cammesa_31.py
import calendar
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Demanda mensual: recta y polinómica", layout="centered")
st.title("¿Cuántos MWh va a demandar este agente?")
st.caption("Recta (clase 29) y polinómica de grado 2 con Lasso (clase 31), sobre demanda-ltimos-aos.csv (CAMMESA).")

@st.cache_resource
def cargar_modelos():
    return joblib.load("modelo_demanda_cammesa.joblib"), joblib.load("modelo_demanda_cammesa_31.joblib")

recta, polinomica = cargar_modelos()
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

st.subheader("Historia del agente")
c1, c2 = st.columns(2)
lag1 = c1.number_input("Demanda del mes anterior (MWh)", 0.0, 1_600_000.0, 50_000.0, step=1000.0)
lag2 = c1.number_input("Demanda de hace dos meses (MWh)", 0.0, 1_600_000.0, 48_000.0, step=1000.0)
tiene_lag12 = c1.checkbox("Conozco la demanda de hace un año", value=True)
lag12 = c1.number_input("Demanda del mismo mes del año pasado (MWh)", 0.0, 1_600_000.0, 52_000.0, step=1000.0, disabled=not tiene_lag12)
anio = c2.number_input("Año", 2017, 2030, 2020)
mes = MESES.index(c2.selectbox("Mes a predecir", MESES, index=1)) + 1
tipo = c2.radio("Tipo de agente", ["Distribuidor · residencial", "Distribuidor · no residencial", "Gran usuario"])

anio_ant, mes_ant = (anio - 1, 12) if mes == 1 else (anio, mes - 1)
ratio_dias = calendar.monthrange(anio, mes)[1] / calendar.monthrange(anio_ant, mes_ant)[1]
fila = pd.DataFrame([{"lag1": lag1, "lag2": lag2, "lag12": lag12 if tiene_lag12 else np.nan, "mes_num": mes,
                      "es_gran_usuario": int(tipo == "Gran usuario"), "es_residencial": int(tipo == "Distribuidor · residencial"),
                      "ratio_dias": ratio_dias}])
est_recta = float(recta.predict(fila.drop(columns="ratio_dias"))[0])
est_poli = float(polinomica.predict(fila)[0])

st.subheader("Predicción")
m1, m2, m3 = st.columns(3)
m1.metric("Recta (clase 29)", f"{est_recta:,.0f} MWh")
m2.metric("Polinómica + Lasso", f"{est_poli:,.0f} MWh", delta=f"{est_poli - est_recta:+,.0f} vs. recta", delta_color="off")
m3.metric("Mes anterior × días", f"{lag1 * ratio_dias:,.0f} MWh")
st.write(f"ratio_dias = {ratio_dias:.3f} (días de {MESES[mes - 1]} / días del mes anterior). "
         "Pasá de enero a febrero y a marzo: la polinómica reacciona al largo del mes (multiplica); la recta suma casi lo mismo cada mes.")

meses = np.arange(1, 13)
filas = []
for m in meses:
    aa, ma = (anio - 1, 12) if m == 1 else (anio, m - 1)
    filas.append({**fila.iloc[0].to_dict(), "mes_num": m, "ratio_dias": calendar.monthrange(anio, m)[1] / calendar.monthrange(aa, ma)[1]})
curva = pd.DataFrame(filas)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(meses, recta.predict(curva.drop(columns="ratio_dias")), marker="o", color="#6b7280", label="recta (clase 29)")
ax.plot(meses, polinomica.predict(curva), marker="o", color="#c0392b", label="polinómica + Lasso")
ax.axhline(lag1, color="black", lw=0.6, ls=":", label="mes anterior")
ax.set_xticks(meses); ax.set_xticklabels([m[:3] for m in MESES]); ax.set_ylabel("MWh predichos"); ax.legend(fontsize=8)
ax.set_title("La misma historia, predicha para cada mes del año")
st.pyplot(fig)

with st.expander("Ver la fila que recibe el modelo"):
    st.dataframe(fila)
