"""
Módulo de Alertas y Riesgo: umbrales configurables, zonas críticas, mensajes contextuales.
"""
import streamlit as st
import pandas as pd
import numpy as np


def render_alerts_sidebar():
    """
    Configuración de umbrales (dentro del expander de la app).
    session_state: alert_mag_above, alert_depth_below, alerts_enabled.
    """
    if "alerts_enabled" not in st.session_state:
        st.session_state["alerts_enabled"] = True
    if "alert_mag_above" not in st.session_state:
        st.session_state["alert_mag_above"] = 6.0
    if "alert_depth_below" not in st.session_state:
        st.session_state["alert_depth_below"] = 70.0

    st.caption("Alertas: marcar sismos que superen cierto tamaño o sean muy superficiales.")
    st.session_state["alerts_enabled"] = st.checkbox("Usar alertas", value=st.session_state["alerts_enabled"], key="alerts_cb")
    st.session_state["alert_mag_above"] = st.slider("Alertar si magnitud >", 4.0, 9.0, st.session_state["alert_mag_above"], 0.1, key="alert_mag")
    st.session_state["alert_depth_below"] = st.slider("Alertar si profundidad < (km)", 10, 200, int(st.session_state["alert_depth_below"]), 5, key="alert_depth")


def get_alerts_contextual_messages(df, mag_above, depth_below):
    """
    Genera mensajes contextuales según datos: eventos que superan umbrales,
    zonas críticas (agrupando por celdas lat/lon si hay coordenadas).
    """
    messages = []
    if df is None or df.empty:
        return messages

    high_mag = df[df["Magnitude"] > mag_above]
    if not high_mag.empty:
        messages.append(f"⚠️ **{len(high_mag)}** evento(s) con magnitud > {mag_above}")

    if "Depth" in df.columns:
        shallow = df[df["Depth"] < depth_below]
        if not shallow.empty:
            messages.append(f"⚠️ **{len(shallow)}** evento(s) superficiales (profundidad < {depth_below} km)")

    if "Latitude" in df.columns and "Longitude" in df.columns and not df.empty:
        # Zona "crítica" = celda con más eventos de alta magnitud
        df_high = df[df["Magnitude"] >= (mag_above - 0.5)]
        if not df_high.empty:
            # Discretizar en grilla gruesa
            df_high = df_high.copy()
            df_high["_lat_bin"] = (df_high["Latitude"] // 0.5) * 0.5
            df_high["_lon_bin"] = (df_high["Longitude"] // 0.5) * 0.5
            counts = df_high.groupby(["_lat_bin", "_lon_bin"]).size().reset_index(name="count")
            if not counts.empty:
                top = counts.nlargest(1, "count").iloc[0]
                messages.append(f"🗺️ Zona de mayor actividad reciente: lat ~{top['_lat_bin']:.1f}°, lon ~{top['_lon_bin']:.1f}° ({int(top['count'])} eventos)")

    return messages


def get_critical_mask(df, mag_above, depth_below):
    """Máscara booleana: True para eventos que disparan alerta (magnitud alta o profundidad baja)."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    mask_mag = df["Magnitude"] > mag_above
    if "Depth" in df.columns:
        mask_depth = df["Depth"].notna() & (df["Depth"] < depth_below)
    else:
        mask_depth = pd.Series(False, index=df.index)
    return mask_mag | mask_depth


def render_alerts_section(df):
    """Renderiza semáforo y mensajes contextuales en la página."""
    if not st.session_state.get("alerts_enabled", True):
        return
    mag_above = st.session_state.get("alert_mag_above", 6.0)
    depth_below = st.session_state.get("alert_depth_below", 70.0)
    messages = get_alerts_contextual_messages(df, mag_above, depth_below)

    st.subheader("🚦 Nivel de Riesgo y Alertas")
    # Semáforo basado en cantidad de eventos que superan umbrales
    critical_count = get_critical_mask(df, mag_above, depth_below).sum()
    if critical_count == 0:
        st.success("🟢 Riesgo bajo: ningún evento supera los umbrales configurados.")
    elif critical_count <= 5:
        st.warning("🟡 Riesgo moderado: algunos eventos superan los umbrales.")
    else:
        st.error("🔴 Riesgo alto: varios eventos superan los umbrales.")

    for msg in messages:
        st.markdown(msg)
    st.caption("Clasificación según umbrales de magnitud y profundidad configurados en la barra lateral.")
