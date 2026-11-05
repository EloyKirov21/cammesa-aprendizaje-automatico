# Clase 44 sobre CAMMESA · Reglas de asociación · aplicación Streamlit
# Carga las reglas guardadas por el cuaderno y, con lo que se sabe de un agente, dice qué comportamiento esperar.
# Para abrirla:  streamlit run app_cammesa_44.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="¿Qué esperar de este agente?", layout="centered")
st.title("¿Qué comportamiento esperar de este agente?")
st.caption("Reglas de asociación (Apriori + association_rules) sobre 868 agentes de demanda-ltimos-aos.csv (CAMMESA, clase 44): atributo ⇒ comportamiento.")

g = joblib.load("reglas_cammesa_44.joblib")
reglas, regiones, soportes = g["reglas"], g["regiones"], g["soportes"]
nombre = lambda s: ", ".join(sorted(s))

st.subheader("Lo que se sabe del agente")
c1, c2 = st.columns(2)
region = c1.selectbox("Región", regiones, format_func=lambda r: r.replace("región ", ""))
tipo = c1.radio("Tipo de agente", ["distribuidor", "gran usuario"])
tarifa = c2.radio("Tarifa", ["tarifa residencial", "tarifa comercial", "tarifa industrial"])
tamano = c2.radio("Tamaño", ["no sé / mediano", "grande", "chico"])
minimo_elevacion = st.slider("Elevación mínima", 1.0, 3.0, 1.3, step=0.1)

tiene = frozenset([region, tipo, tarifa] + ([tamano] if tamano in ("grande", "chico") else []))
aplican = reglas[reglas["antecedents"].apply(lambda A: A <= tiene) & (reglas["lift"] >= minimo_elevacion)].sort_values("lift", ascending=False)
mejores = aplican.drop_duplicates(subset="consequents").head(5)                     # una fila por comportamiento: la regla de mayor elevación

st.subheader("Qué esperar")
if len(mejores) == 0:
    st.write("Ninguna regla con esa elevación aplica: con estos datos, nada en este agente hace un comportamiento más probable que el promedio.")
else:
    for r in mejores.itertuples():
        c = nombre(r.consequents)
        st.write(f"**{c}** · porque {{{nombre(r.antecedents)}}} ⇒ {{{c}}}: confianza {r.confidence:.0%} (contra {soportes[c]:.0%} en todos los agentes), elevación {r.lift:.2f}, soporte {r.support:.1%}")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    etiquetas = [nombre(r.consequents) for r in mejores.itertuples()]
    ax.barh(etiquetas, mejores["lift"], color="#2e8b57"); ax.axvline(1, color="#c0392b", ls="--", label="elevación 1: azar")
    for y, (l, cf) in enumerate(zip(mejores["lift"], mejores["confidence"])): ax.text(l + 0.05, y, f"confianza {cf:.0%}", va="center", fontsize=9)
    ax.set_xlabel("elevación"); ax.invert_yaxis(); ax.legend(loc="lower right"); ax.set_title("Lo que este agente hace más probable que el promedio")
    st.pyplot(fig)

with st.expander("Ver todas las reglas que aplican"):
    muestra = aplican.copy(); muestra["regla"] = [f"{{{nombre(a)}}} ⇒ {{{nombre(b)}}}" for a, b in zip(muestra["antecedents"], muestra["consequents"])]
    st.dataframe(muestra[["regla", "support", "confidence", "lift", "conviction"]].round(3).reset_index(drop=True))
