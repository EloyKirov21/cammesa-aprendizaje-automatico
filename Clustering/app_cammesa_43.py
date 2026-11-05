# Clase 43 sobre CAMMESA · Agrupamiento por densidad · aplicación Streamlit
# Carga lo guardado por el cuaderno (escalador, centrales de DBSCAN, eps, k-medias) y dice si un agente nuevo se parece a alguno.
# Para abrirla:  streamlit run app_cammesa_43.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="¿Se parece a algún agente?", layout="centered")
st.title("¿Este agente se parece a alguno que ya tenemos?")
st.caption("DBSCAN sobre el perfil de 868 series de demanda-ltimos-aos.csv (CAMMESA, clase 43): una región densa y ruido. Y a qué grupo de k-medias iría (clase 41).")

g = joblib.load("densidad_cammesa_43.joblib")
preparar, centrales, EPS, kmedias, nombres, COLUMNAS, perfiles = g["preparar"], g["centrales"], g["eps"], g["kmedias"], g["nombres"], g["columnas"], g["perfiles"]
es_ruido = g["ruido"]
buscador = NearestNeighbors(n_neighbors=1).fit(centrales)

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
distancia = float(buscador.kneighbors(z)[0][0, 0])
adentro = distancia <= EPS
grupo = int(kmedias.predict(z)[0])

st.subheader("Resultado")
m1, m2 = st.columns(2)
m1.metric("Distancia a la central más cercana", f"{distancia:.2f}", help=f"eps = {EPS}: más lejos que eso, ruido")
m2.metric("Según la densidad", "se parece a agentes que tenemos" if adentro else "RUIDO: no se parece a ninguno")
st.write(f"k-medias (clase 41) lo mandaría al grupo «{nombres[grupo]}»" + ("." if adentro else ", pero a esa distancia no hay agentes parecidos: el grupo le queda grande."))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.bar(["este agente"], [distancia], color="#2e8b57" if adentro else "#c0392b"); a.axhline(EPS, color="black", ls="--", label=f"eps = {EPS}")
a.set_ylabel("distancia a la central más cercana (escalada)"); a.set_ylim(0, max(EPS * 2, distancia * 1.2)); a.legend(); a.set_title("Adentro o afuera de la región densa")
b.scatter(perfiles.loc[~es_ruido, "log_tamano"], perfiles.loc[~es_ruido, "volatilidad"], s=8, alpha=0.4, color="#0055a5", label="en la región densa")
b.scatter(perfiles.loc[es_ruido, "log_tamano"], perfiles.loc[es_ruido, "volatilidad"], s=14, alpha=0.8, color="#9ca3af", marker="x", label="ruido (DBSCAN)")
b.scatter([np.log10(mwh)], [volatilidad / 100], marker="*", s=350, color="#2e8b57" if adentro else "#c0392b", edgecolor="black", zorder=5, label="este agente")
b.set_xlabel("log₁₀ del tamaño (MWh por mes)"); b.set_ylabel("volatilidad"); b.set_ylim(0, max(0.8, volatilidad / 100 * 1.1)); b.legend(fontsize=7); b.set_title("Dónde cae entre las 868 series")
plt.tight_layout(); st.pyplot(fig)
