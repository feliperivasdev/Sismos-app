"""
Módulo de Exploración Inteligente.
Búsqueda por fecha/magnitud, selección de evento, eventos similares, filtro global.
"""
import streamlit as st
import pandas as pd
import numpy as np


def apply_search_filters(df, date_from=None, date_to=None, mag_min=None, mag_max=None):
    """Aplica filtros de búsqueda por fecha y magnitud. Devuelve copia filtrada."""
    out = df.copy()
    if "Time" in out.columns and (date_from or date_to):
        out["_date"] = pd.to_datetime(out["Time"]).dt.date
        if date_from:
            out = out[out["_date"] >= date_from]
        if date_to:
            out = out[out["_date"] <= date_to]
        out = out.drop(columns=["_date"], errors="ignore")
    if mag_min is not None:
        out = out[out["Magnitude"] >= mag_min]
    if mag_max is not None:
        out = out[out["Magnitude"] <= mag_max]
    return out


def get_similar_events(df, event_row, mag_tolerance=0.3, depth_tolerance_km=20, max_results=15):
    """
    Encuentra sismos similares: magnitud dentro de ±mag_tolerance y profundidad cercana (si existe).
    event_row: Serie de pandas con al menos Magnitude; opcional Depth, Latitude, Longitude.
    """
    if df.empty or event_row is None:
        return pd.DataFrame()
    mag = event_row["Magnitude"]
    mag_lo, mag_hi = mag - mag_tolerance, mag + mag_tolerance
    mask = (df["Magnitude"] >= mag_lo) & (df["Magnitude"] <= mag_hi)
    similar = df[mask].copy()
    if "Depth" in df.columns and pd.notna(event_row.get("Depth")):
        depth = event_row["Depth"]
        d_lo = depth - depth_tolerance_km
        d_hi = depth + depth_tolerance_km
        similar = similar[(similar["Depth"] >= d_lo) & (similar["Depth"] <= d_hi)]
    similar = similar.head(max_results)
    return similar


def render_exploration_sidebar(df):
    """
    Renderiza en sidebar: búsqueda por fecha/magnitud y selector de evento.
    Actualiza st.session_state: search_date_from, search_date_to, search_mag_min, search_mag_max,
    selected_event_idx, exploration_mag_tol, exploration_depth_tol.
    Devuelve (df_filtered, selected_row o None).
    """
    if df is None or df.empty:
        return df, None

    # Filtros de búsqueda
    st.caption("Opcional: filtra por fecha o magnitud y elige un sismo para resaltarlo en todos los gráficos.")
    min_d = max_d = None
    if "Time" in df.columns:
        min_d = pd.to_datetime(df["Time"]).min().date()
        max_d = pd.to_datetime(df["Time"]).max().date()
    mag_min_global = float(df["Magnitude"].min())
    mag_max_global = float(df["Magnitude"].max())

    with st.expander("Por fecha y magnitud", expanded=False):
        if min_d and max_d:
            st.date_input("Desde (fecha)", value=min_d, min_value=min_d, max_value=max_d, key="search_date_from")
            st.date_input("Hasta (fecha)", value=max_d, min_value=min_d, max_value=max_d, key="search_date_to")
        st.number_input("Magnitud mínima", value=mag_min_global, min_value=mag_min_global, max_value=10.0, step=0.1, key="search_mag_min")
        st.number_input("Magnitud máxima", value=mag_max_global, min_value=0.0, max_value=mag_max_global, step=0.1, key="search_mag_max")

    search_date_from = st.session_state.get("search_date_from", min_d)
    search_date_to = st.session_state.get("search_date_to", max_d)
    search_mag_min = st.session_state.get("search_mag_min", mag_min_global)
    search_mag_max = st.session_state.get("search_mag_max", mag_max_global)

    df_filtered = apply_search_filters(df, search_date_from, search_date_to, search_mag_min, search_mag_max)

    # Selector de evento
    if df_filtered.empty:
        st.warning("No hay eventos con los filtros actuales.")
        return df_filtered, None

    # Crear opciones: índice + fecha + magnitud para el selector
    def _fmt(r):
        t = r.get('Time')
        ts = pd.to_datetime(t).strftime('%Y-%m-%d') if pd.notna(t) else 'N/A'
        return f"M {r['Magnitude']:.1f} | {ts}"[:50]
    options = df_filtered.apply(_fmt, axis=1)
    options = options.tolist()
    indices = list(range(len(df_filtered)))
    choice = st.selectbox(
        "Elige un sismo para resaltarlo en los gráficos",
        options=indices,
        format_func=lambda i: f"Evento {i+1}: {options[i] if i < len(options) else ''}",
        index=0,
        key="selected_event_idx"
    )
    selected_row = df_filtered.iloc[choice]
    return df_filtered, selected_row


def render_similar_events_main(df_filtered, selected_row):
    """
    Muestra en el área principal la tabla de eventos similares al seleccionado.
    Incluye controles de tolerancia (magnitud y profundidad).
    """
    if df_filtered is None or selected_row is None or df_filtered.empty:
        return
    mag_tol = st.slider("Tolerancia en magnitud (±)", 0.1, 1.0, 0.3, key="similar_mag_tol")
    depth_tol = 20
    if "Depth" in df_filtered.columns:
        depth_tol = st.slider("Tolerancia en profundidad (km)", 5, 100, 20, key="similar_depth_tol")
    similar = get_similar_events(df_filtered, selected_row, mag_tolerance=mag_tol, depth_tolerance_km=depth_tol)
    st.caption(f"**{len(similar)}** eventos con magnitud y profundidad parecidas al sismo elegido.")
    if not similar.empty:
        cols = ["Magnitude", "Time", "Depth"] if "Depth" in similar.columns else ["Magnitude", "Time"]
        st.dataframe(similar[cols].head(15), use_container_width=True, hide_index=True)
    else:
        st.caption("No se encontraron eventos similares con esas tolerancias.")
