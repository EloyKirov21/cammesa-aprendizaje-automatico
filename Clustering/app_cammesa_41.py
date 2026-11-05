# Clase 41 sobre CAMMESA · Agrupamiento · aplicación Streamlit
# Carga el agrupamiento guardado por el cuaderno (escalar → k-medias con 3 grupos) y ubica a un agente nuevo.
# Para abrirla:  streamlit run app_cammesa_41.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="¿A qué grupo de agentes se parece?", layout="centered")
st.title("¿A qué grupo de agentes se parece?")
st.caption("k-medias con 3 grupos sobre el perfil de 868 series de demanda-ltimos-aos.csv (CAMMESA, clase 41). La tarifa y el tipo de agente no entran.")

guardado = joblib.load("agrupamiento_cammesa_41.joblib")
modelo, nombres, COLUMNAS, perfiles = guardado["modelo"], guardado["nombres"], guardado["columnas"], guardado["perfiles"]
grupos = modelo.predict(perfiles)
COLORES = ["#0055a5", "#c0392b", "#2e8b57"]

st.subheader("Perfil del agente")
c1, c2 = st.columns(2)
mwh = c1.number_input("Demanda media (MWh por mes)", 1.0, 800_000.0, 800.0, step=50.0)
ver_inv = c1.slider("Verano / invierno (1 = parejo)", 0.3, 3.0, 1.4, step=0.05)
volatilidad = c1.slider("Variación mensual típica (%)", 0.0, 60.0, 12.0, step=0.5)
crecimiento = c2.slider("Crecimiento por año (%)", -50.0, 50.0, 3.0, step=0.5)
caidas = c2.slider("Meses con caída de más de 15 % (%)", 0.0, 60.0, 17.0, step=1.0)
cv = c2.slider("Variabilidad (desvío / media)", 0.0, 2.0, 0.25, step=0.01)

fila = pd.DataFrame([{"log_tamano": np.log10(mwh), "estacionalidad": np.log2(ver_inv), "volatilidad": volatilidad / 100,
                      "tendencia_anual": np.log10(1 + crecimiento / 100), "frac_caidas": caidas / 100, "cv": cv}])[COLUMNAS]
grupo = int(modelo.predict(fila)[0])
distancias = modelo.transform(fila)[0]

st.subheader("Grupo")
st.metric("Se parece a los", nombres[grupo])
segundo = int(np.argsort(distancias)[1])
st.write("Distancia a cada centroide (en unidades escaladas): " + " · ".join(f"{nombres[j]} {d:.2f}" for j, d in enumerate(distancias))
         + f". El segundo más cercano es «{nombres[segundo]}»" + (": está entre dos grupos." if distancias[segundo] - distancias[grupo] < 0.5 else "."))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.bar([nombres[j] for j in range(3)], distancias, color=[COLORES[j] if j == grupo else "#9ca3af" for j in range(3)])
a.set_ylabel("distancia al centroide"); a.set_title("Más cerca del grupo coloreado"); a.tick_params(axis="x", labelsize=7, rotation=10)
for j in range(3):
    sel = grupos == j
    b.scatter(perfiles.loc[sel, "log_tamano"], perfiles.loc[sel, "volatilidad"], s=8, alpha=0.4, color=COLORES[j], label=nombres[j])
b.scatter([np.log10(mwh)], [volatilidad / 100], marker="*", s=350, color="black", edgecolor="white", zorder=5, label="este agente")
b.set_xlabel("log₁₀ del tamaño (MWh por mes)"); b.set_ylabel("volatilidad"); b.set_ylim(0, 0.6); b.set_title("Dónde cae entre las 868 series"); b.legend(fontsize=6)
plt.tight_layout(); st.pyplot(fig)

with st.expander("Ver los centroides de los tres grupos"):
    st.dataframe(guardado["centros"])
