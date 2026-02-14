"""
Módulo de Comparaciones Geográficas: regiones, ranking de zonas, evolución por región, nacional vs local.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def get_regions_bbox():
    """Devuelve diccionario de regiones predefinidas (nombre -> (lat_min, lat_max, lon_min, lon_max))."""
    return {
        "Nariño (Colombia)": (0.5, 2.5, -79.0, -76.5),
        "Chile central": (-38, -32, -74, -70),
        "Perú costa": (-18, -4, -82, -68),
        "Japón": (30, 46, 128, 146),
        "Indonesia": (-12, 6, 95, 141),
        "California (EE.UU.)": (32, 42, -125, -114),
    }


def filter_by_bbox(df, lat_min, lat_max, lon_min, lon_max):
    """Filtra DataFrame por bounding box (requiere Latitude, Longitude)."""
    if df is None or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["Latitude"] >= lat_min) & (df["Latitude"] <= lat_max) &
        (df["Longitude"] >= lon_min) & (df["Longitude"] <= lon_max)
    ].copy()


def ranking_zonas(df, bin_deg=0.5, top_n=10):
    """
    Ranking de celdas (lat/lon) por número de eventos.
    bin_deg: tamaño de celda en grados.
    """
    if df is None or df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["_lat_bin"] = (d["Latitude"] // bin_deg) * bin_deg
    d["_lon_bin"] = (d["Longitude"] // bin_deg) * bin_deg
    rank = d.groupby(["_lat_bin", "_lon_bin"]).agg(
        eventos=("Magnitude", "count"),
        mag_promedio=("Magnitude", "mean"),
        mag_max=("Magnitude", "max"),
    ).reset_index()
    rank = rank.sort_values("eventos", ascending=False).head(top_n)
    return rank


def render_geo_comparison(df):
    """Renderiza pestaña/sección de comparaciones geográficas."""
    if df is None or df.empty:
        st.warning("No hay datos para comparación geográfica.")
        return

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.warning("Se requieren Latitud y Longitud para comparación geográfica.")
        return

    st.subheader("🗺️ Comparaciones Geográficas")

    regions = get_regions_bbox()
    reg_names = list(regions.keys())

    col1, col2 = st.columns(2)
    with col1:
        r1 = st.selectbox("Región A", reg_names, key="geo_region_a")
    with col2:
        r2 = st.selectbox("Región B", reg_names, key="geo_region_b")

    bbox1 = regions[r1]
    bbox2 = regions[r2]
    df1 = filter_by_bbox(df, *bbox1)
    df2 = filter_by_bbox(df, *bbox2)

    # Vista lado a lado: métricas y evolución
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{r1}**")
        if df1.empty:
            st.caption("Sin datos en este rango.")
        else:
            st.metric("Eventos", len(df1))
            st.metric("Mag. máxima", f"{df1['Magnitude'].max():.2f}")
            if "Time" in df1.columns:
                df1_y = df1.copy()
                df1_y["Year"] = df1_y["Time"].dt.year
                ev1 = df1_y.groupby("Year").size()
                fig1 = go.Figure(go.Bar(x=ev1.index, y=ev1.values, name=r1[:15]))
                fig1.update_layout(title="Evolución por año", height=220, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig1, use_container_width=True, key="geo_evol_region_a")

    with c2:
        st.markdown(f"**{r2}**")
        if df2.empty:
            st.caption("Sin datos en este rango.")
        else:
            st.metric("Eventos", len(df2))
            st.metric("Mag. máxima", f"{df2['Magnitude'].max():.2f}")
            if "Time" in df2.columns:
                df2_y = df2.copy()
                df2_y["Year"] = df2_y["Time"].dt.year
                ev2 = df2_y.groupby("Year").size()
                fig2 = go.Figure(go.Bar(x=ev2.index, y=ev2.values, name=r2[:15]))
                fig2.update_layout(title="Evolución por año", height=220, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig2, use_container_width=True, key="geo_evol_region_b")

    # Ranking global de zonas (nacional / todo el dataset)
    st.markdown("---")
    st.markdown("**📊 Ranking de zonas más activas** (todo el dataset)")
    rank = ranking_zonas(df, bin_deg=0.5, top_n=15)
    if not rank.empty:
        rank_display = rank.rename(columns={"_lat_bin": "Lat (centro)", "_lon_bin": "Lon (centro)", "eventos": "Eventos", "mag_promedio": "Mag. prom.", "mag_max": "Mag. máx."})
        st.dataframe(rank_display, use_container_width=True, hide_index=True)
    else:
        st.caption("No se pudo calcular el ranking.")
