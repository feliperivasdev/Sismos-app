"""
Módulo Educativo: tooltips, glosario sísmico, modo aprendizaje y mensajes contextuales.
"""
import streamlit as st

GLOSARIO = {
    "Magnitud": "Medida de la energía liberada por un sismo (escala logarítmica). Un aumento de 1 unidad implica ~32 veces más energía.",
    "Profundidad": "Distancia (km) desde el foco del sismo hasta la superficie. Sismos superficiales (< 70 km) suelen sentirse más.",
    "Gutenberg-Richter": "Ley empírica: log₁₀(N) = a - b·M. Relaciona la frecuencia de sismos con su magnitud; 'b' suele estar ~1.0.",
    "R² (coeficiente de determinación)": "Indica qué proporción de la variabilidad de los datos explica el modelo (0 a 1).",
    "RMSE": "Raíz del error cuadrático medio. Mide el error típico del modelo en las mismas unidades que la variable.",
    "Foco (hipocentro)": "Punto en el interior de la Tierra donde se origina el sismo.",
    "Epicentro": "Punto en la superficie terrestre directamente sobre el foco.",
}

TOOLTIPS = {
    "Magnitude": "Magnitud: energía liberada (escala logarítmica).",
    "Depth": "Profundidad del foco en km.",
    "Time": "Fecha y hora del evento (UTC).",
}


def get_tooltip(field):
    return TOOLTIPS.get(field, "")


def render_educational_sidebar():
    """Toggle Modo Educativo. Usar st.session_state['educational_mode']."""
    if "educational_mode" not in st.session_state:
        st.session_state["educational_mode"] = False
    st.session_state["educational_mode"] = st.checkbox(
        "📘 Activar modo aprendizaje (explicaciones y glosario)",
        value=st.session_state["educational_mode"],
        key="educational_mode_cb"
    )
    return st.session_state["educational_mode"]


def render_glossary():
    """Muestra el glosario sísmico en un expander."""
    with st.expander("📘 Glosario sísmico", expanded=False):
        for term, definition in GLOSARIO.items():
            st.markdown(f"**{term}**  \n{definition}")


def educational_caption(text, key=None):
    """Si modo educativo está activo, muestra un caption explicativo."""
    if st.session_state.get("educational_mode"):
        st.caption(text)


def wrap_with_tooltip(label, field_key):
    """Devuelve label con ícono de info si modo educativo; tooltip en caption debajo si se usa."""
    if st.session_state.get("educational_mode"):
        tip = get_tooltip(field_key) or GLOSARIO.get(field_key, "")
        if tip:
            return f"{label} ℹ️"
    return label
