# Clase 46 sobre CAMMESA · Detección de anomalías · aplicación Streamlit
# Carga tres detectores guardados por el cuaderno y dice si un agente nuevo es raro, con el percentil entre las 868 series.
# Para abrirla:  streamlit run app_cammesa_46.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="¿Este agente es raro?", layout="centered")
st.title("¿Este agente es raro?")
st.caption("Tres detectores (bosque de aislamiento, factor local en modo novedad, envolvente elíptica) sobre el perfil de 868 series de demanda-ltimos-aos.csv (CAMMESA, clase 46).")

g = joblib.load("anomalias_cammesa_46.joblib")
preparar, detectores, referencia, COLUMNAS, perfiles, votos_actuales = g["preparar"], g["detectores"], g["referencia"], g["columnas"], g["perfiles"], g["votos"]
N = len(perfiles)
umbral_pct = st.sidebar.slider("Proporción de raros que se espera (%)", 1, 10, 2)

st.subheader("Perfil del agente")
c1, c2 = st.columns(2)
mwh = c1.number_input("Demanda media (MWh por mes)", 1.0, 800_000.0, 800.0, step=50.0)
ver_inv = c1.slider("Verano / invierno (1 = parejo)", 0.3, 4.0, 1.4, step=0.05)
volatilidad = c1.slider("Variación mensual típica (%)", 0.0, 80.0, 12.0, step=0.5)
crecimiento = c2.slider("Crecimiento por año (%)", -60.0, 60.0, 3.0, step=0.5)
caidas = c2.slider("Meses con caída de más de 15 % (%)", 0.0, 60.0, 17.0, step=1.0)
cv = c2.slider("Variabilidad (desvío / media)", 0.0, 2.0, 0.25, step=0.01)

fila = pd.DataFrame([{"log_tamano": np.log10(mwh), "estacionalidad": np.log2(ver_inv), "volatilidad": volatilidad / 100,
                      "tendencia_anual": np.log10(1 + crecimiento / 100), "frac_caidas": caidas / 100, "cv": cv}])[COLUMNAS]
z = preparar.transform(fila)

filas = []
for nombre, d in detectores.items():
    p = float(d.score_samples(z)[0]); ref = referencia[nombre]
    percentil = float((ref < p).mean() * 100)                                                 # qué fracción de los 868 es más rara que este agente
    filas.append({"detector": nombre, "puntaje": p, "percentil": percentil, "raro": percentil < umbral_pct})
tabla = pd.DataFrame(filas).set_index("detector")
votos = int(tabla["raro"].sum())

st.subheader("Veredicto")
m1, m2 = st.columns(2)
m1.metric("Detectores que lo marcan", f"{votos} de 3")
m2.metric("Por consenso", "RARO" if votos >= 2 else ("dudoso" if votos == 1 else "normal"))
st.write(f"Un agente es «raro» para un detector si su puntaje está entre el {umbral_pct} % más bajo de las {N} series actuales. El umbral lo decide quien conoce la proporción de raros o el costo de cada error, no el modelo.")

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
colores = ["#c0392b" if r else "#0055a5" for r in tabla["raro"]]
a.barh(tabla.index, tabla["percentil"], color=colores); a.axvline(umbral_pct, color="black", ls="--", label=f"umbral: {umbral_pct} %")
for y, v in enumerate(tabla["percentil"]): a.text(min(v, 90) + 1, y, f"percentil {v:.1f}", va="center", fontsize=9)
a.set_xlim(0, 100); a.set_xlabel(f"percentil del puntaje entre las {N} series (0 = el más raro)"); a.invert_yaxis(); a.legend(loc="lower right"); a.set_title("Dónde cae para cada detector")
raros = votos_actuales >= 2
b.scatter(perfiles.loc[~raros, "log_tamano"], perfiles.loc[~raros, "volatilidad"], s=8, alpha=0.4, color="#0055a5", label="series actuales")
b.scatter(perfiles.loc[raros, "log_tamano"], perfiles.loc[raros, "volatilidad"], s=30, color="#9ca3af", marker="x", label="raras por consenso (2 %)")
b.scatter([np.log10(mwh)], [volatilidad / 100], marker="*", s=350, color="#c0392b" if votos >= 2 else "#2e8b57", edgecolor="black", zorder=5, label="este agente")
b.set_xlabel("log₁₀ del tamaño"); b.set_ylabel("volatilidad"); b.set_ylim(0, max(1.0, volatilidad / 100 * 1.1)); b.legend(fontsize=7); b.set_title("Entre las series actuales")
plt.tight_layout(); st.pyplot(fig)

with st.expander("Ver los puntajes y la fila que reciben los detectores"):
    st.dataframe(tabla.round(3))
    st.dataframe(fila)
