import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- CONFIGURACIÓN CONSTANTE ---
NARINO_LAT_RANGE = (0.5, 2.5)
NARINO_LON_RANGE = (-79.0, -76.5)
ANIMATION_STEP_DAYS = 30  # Días por frame de animación
# Límites para evitar MessageSizeError (payload al navegador)
MAX_EVENTS_FOR_ANIMATION = 10_000   # si hay más, se muestrea
MAX_FRAMES = 40                     # máximo de frames en la animación
MAX_POINTS_PER_FRAME = 2_500        # puntos por frame (por traza); si hay más, se muestrea

def _filter_narino_region(df):
    """
    Filtra el DataFrame para la región geográfica de Nariño.
    """
    required_cols = ['Latitude', 'Longitude', 'Time', 'Magnitude']
    if not all(col in df.columns for col in required_cols):
        st.error(f"El dataset requiere las columnas: {required_cols}")
        return pd.DataFrame()

    mask = (
        (df['Latitude'] >= NARINO_LAT_RANGE[0]) & 
        (df['Latitude'] <= NARINO_LAT_RANGE[1]) &
        (df['Longitude'] >= NARINO_LON_RANGE[0]) & 
        (df['Longitude'] <= NARINO_LON_RANGE[1])
    )
    # Ordenar por fecha es crucial para la animación
    return df[mask].copy().sort_values(by="Time")

def _get_highlights(df):
    """Eventos destacados para saltos rápidos."""
    highlights = {}
    if df.empty:
        return highlights

    idx_max = df['Magnitude'].idxmax()
    max_event = df.loc[idx_max]
    highlights['max_mag'] = {
        'date': max_event['Time'].date(),
        'mag': max_event['Magnitude']
    }

    df_temp = df.copy()
    df_temp['Year'] = df_temp['Time'].dt.year
    year_counts = df_temp['Year'].value_counts()
    if not year_counts.empty:
        busiest_year = year_counts.idxmax()
        highlights['active_year'] = {
            'date': datetime.date(busiest_year, 1, 1),
            'year': busiest_year,
            'count': int(year_counts.max())
        }
    # Zona más activa (celda lat/lon con más eventos)
    if 'Latitude' in df.columns and 'Longitude' in df.columns and not df.empty:
        d = df.copy()
        d['_lat_bin'] = (d['Latitude'] // 0.5) * 0.5
        d['_lon_bin'] = (d['Longitude'] // 0.5) * 0.5
        zone_counts = d.groupby(['_lat_bin', '_lon_bin']).size().reset_index(name='n')
        top_zone = zone_counts.nlargest(1, 'n').iloc[0]
        highlights['active_zone'] = {
            'lat': float(top_zone['_lat_bin']),
            'lon': float(top_zone['_lon_bin']),
            'count': int(top_zone['n'])
        }
    return highlights


def _render_narrative_panel(highlights, total_events):
    """Panel de narrativa guiada: año más activo, zona más activa, mensajes dinámicos."""
    if not highlights:
        st.caption("No hay suficientes datos para el resumen.")
        return
    lines = []
    if 'active_year' in highlights:
        h = highlights['active_year']
        lines.append(f"📅 **Año con más sismos:** {h['year']} ({h['count']} eventos).")
    if 'max_mag' in highlights:
        h = highlights['max_mag']
        lines.append(f"📌 **Sismo más fuerte:** {h['date']} (magnitud {h['mag']:.1f}).")
    if 'active_zone' in highlights:
        h = highlights['active_zone']
        lines.append(f"🗺️ **Zona con más eventos:** alrededor de lat {h['lat']:.1f}°, lon {h['lon']:.1f}° ({h['count']} sismos).")
    if lines:
        for line in lines:
            st.markdown(line)
    st.caption(f"En total hay **{total_events}** eventos en el rango que elegiste.")

def _generate_frames(df, start_date, end_date):
    """
    Genera frames para Plotly. Cada frame es acumulativo.
    Limita eventos por frame y número de frames para evitar MessageSizeError.
    """
    frames = []
    start_ts = pd.Timestamp(start_date)
    try:
        start_ts = start_ts.tz_localize('UTC')
    except Exception:
        pass
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    try:
        end_ts = end_ts.tz_localize('UTC')
    except Exception:
        pass
    end = end_ts

    # Generar fechas de frame: como máximo MAX_FRAMES
    total_days = (end - start_ts).days
    if total_days <= 0:
        total_days = 1
    step_days = max(1, total_days // MAX_FRAMES)
    step = pd.Timedelta(days=step_days)
    frame_dates = []
    current = start_ts
    while current <= end and len(frame_dates) < MAX_FRAMES:
        frame_dates.append(current)
        current += step
    if frame_dates and frame_dates[-1] < end:
        frame_dates.append(end)

    def _sample_if_needed(d, max_pts):
        if d is None or len(d) <= max_pts:
            return d
        return d.sample(n=max_pts, random_state=42)

    for date in frame_dates:
        d_sub = df[df['Time'] <= date]
        recent_threshold = date - pd.Timedelta(days=60)
        hist = d_sub[d_sub['Time'] < recent_threshold]
        recent = d_sub[d_sub['Time'] >= recent_threshold]

        hist = _sample_if_needed(hist, MAX_POINTS_PER_FRAME)
        recent = _sample_if_needed(recent, MAX_POINTS_PER_FRAME)

        if hist is None or hist.empty:
            hist_lat, hist_lon = [], []
        else:
            hist_lat, hist_lon = hist['Latitude'].tolist(), hist['Longitude'].tolist()
        if recent is None or recent.empty:
            rec_lat, rec_lon, rec_mag, rec_text = [], [], [], []
        else:
            rec_lat = recent['Latitude'].tolist()
            rec_lon = recent['Longitude'].tolist()
            rec_mag = recent['Magnitude'].tolist()
            rec_text = recent.apply(lambda x: f"M {x['Magnitude']} - {x['Time'].date()}", axis=1).tolist()

        frames.append(go.Frame(
            data=[
                go.Scattermapbox(
                    lat=hist_lat, lon=hist_lon,
                    marker=dict(color='gray', size=4, opacity=0.3),
                    hoverinfo='none'
                ),
                go.Scattermapbox(
                    lat=rec_lat, lon=rec_lon,
                    marker=dict(
                        color=rec_mag,
                        size=[m * 3 for m in rec_mag],
                        colorscale='Inferno',
                        cmin=4, cmax=8,
                        showscale=False
                    ),
                    text=rec_text,
                    hoverinfo='text'
                )
            ],
            name=str(date.date())
        ))

    return frames, frame_dates

def _render_animated_map(df, start_date, end_date, center_lat=None, center_lon=None, zoom=6):
    """
    Genera la figura Plotly con animación.
    center_lat, center_lon: si no se pasan, se usa el centro de Nariño.
    zoom: nivel de zoom del mapa (menor = más alejado).
    """
    if center_lat is None:
        center_lat = (NARINO_LAT_RANGE[0] + NARINO_LAT_RANGE[1]) / 2
    if center_lon is None:
        center_lon = (NARINO_LON_RANGE[0] + NARINO_LON_RANGE[1]) / 2

    fig = go.Figure(
        data=[
            # Trace 0: Histórico (Gris)
            go.Scattermapbox(
                lat=[], lon=[], mode='markers',
                marker=go.scattermapbox.Marker(size=4, color='gray', opacity=0.3),
                name='Histórico'
            ),
            # Trace 1: Reciente (Color)
            go.Scattermapbox(
                lat=[], lon=[], mode='markers',
                marker=go.scattermapbox.Marker(
                    size=6, color='red', colorscale='Inferno', 
                    cmin=4, cmax=8, opacity=0.9, showscale=True
                ),
                name='Nuevo Evento'
            )
        ]
    )

    # 2. Generar Frames
    with st.spinner("Generando animación suave (esto solo ocurre una vez)..."):
        frames, frame_dates = _generate_frames(df, start_date, end_date)
        fig.frames = frames

    # Controles arriba del mapa para que no se monten con el slider
    fig.update_layout(
        title=f"Evolución: {start_date} a {end_date}",
        mapbox_style="open-street-map",
        mapbox_zoom=zoom,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        height=600,
        margin={"r": 0, "t": 80, "l": 0, "b": 80},
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
            pad=dict(t=5, b=15),
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                method='animate',
                args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=f.name if (i % max(1, len(frames) // 8)) == 0 else ""
            ) for i, f in enumerate(frames)],
            currentvalue=dict(prefix="Fecha: ", visible=True, font=dict(size=13)),
            pad=dict(t=40, b=10),
            len=0.9,
            x=0.05,
            ticklen=4,
            minorticklen=0
        )]
    )
    
    return fig

def _render_period_comparison(df, min_date, max_date):
    """Comparación antes vs después: dos periodos lado a lado."""
    # Normalizar a datetime.date y asegurar min <= max (evita errores con st.date_input)
    try:
        min_d = min_date.date() if hasattr(min_date, 'date') and not isinstance(min_date, datetime.date) else min_date
        max_d = max_date.date() if hasattr(max_date, 'date') and not isinstance(max_date, datetime.date) else max_date
    except Exception:
        min_d, max_d = min_date, max_date
    if hasattr(min_d, 'date'):
        min_d = min_d.date()
    if hasattr(max_d, 'date'):
        max_d = max_d.date()
    if min_d > max_d:
        min_d, max_d = max_d, min_d
    delta = (max_d - min_d).days if hasattr(max_d - min_d, 'days') else 1
    delta = max(1, delta)
    half = delta // 2
    mid = min_d + datetime.timedelta(days=half)
    if mid > max_d:
        mid = max_d
    st.markdown("#### 🗓️ Comparar periodos (antes vs después)")
    col_a, col_b = st.columns(2)
    with col_a:
        start_a = st.date_input("Periodo A: inicio", value=min_d, min_value=min_d, max_value=max_d, key="story_period_a_start")
        end_a = st.date_input("Periodo A: fin", value=mid, min_value=min_d, max_value=max_d, key="story_period_a_end")
    with col_b:
        start_b = st.date_input("Periodo B: inicio", value=mid, min_value=min_d, max_value=max_d, key="story_period_b_start")
        end_b = st.date_input("Periodo B: fin", value=max_d, min_value=min_d, max_value=max_d, key="story_period_b_end")

    ts = pd.to_datetime(df['Time']).dt.tz_localize(None) if df['Time'].dt.tz is not None else pd.to_datetime(df['Time'])
    df_copy = df.copy()
    df_copy['_d'] = pd.to_datetime(df_copy['Time']).dt.date

    def count_in_range(s, e):
        return len(df_copy[(df_copy['_d'] >= s) & (df_copy['_d'] <= e)])

    ca, cb = st.columns(2)
    with ca:
        n_a = count_in_range(start_a, end_a)
        st.metric("Eventos Periodo A", n_a)
        st.caption(f"{start_a} → {end_a}")
    with cb:
        n_b = count_in_range(start_b, end_b)
        st.metric("Eventos Periodo B", n_b)
        st.caption(f"{start_b} → {end_b}")
    if n_a + n_b > 0:
        diff = n_b - n_a
        st.info(f"Diferencia (B - A): **{diff:+d}** eventos." + (" El segundo periodo tiene más actividad." if diff > 0 else " El primer periodo tiene más actividad."))


def render_storytelling(df):
    """
    Función Principal del Módulo.
    """
    st.markdown("## 🎬 Cómo ocurrieron los sismos en el tiempo")
    st.caption("Pulsa Play para ver la animación; los puntos aparecen según la fecha del sismo.")

    required_cols = ['Latitude', 'Longitude', 'Time', 'Magnitude']
    if not all(col in df.columns for col in required_cols):
        st.warning("Para esta vista hacen falta columnas de latitud, longitud, fecha y magnitud.")
        return

    use_narino_only = st.checkbox("Usar solo datos de Nariño (Colombia)", value=True, key="story_narino")
    if use_narino_only:
        df_work = _filter_narino_region(df)
    else:
        df_work = df.sort_values(by="Time").copy()

    if df_work.empty:
        st.warning("No hay datos en la región seleccionada.")
        return

    # Limitar datos para evitar MessageSizeError (677 MB) al deseleccionar Nariño
    n_total = len(df_work)
    if n_total > MAX_EVENTS_FOR_ANIMATION:
        df_work = df_work.sample(n=MAX_EVENTS_FOR_ANIMATION, random_state=42).sort_values(by="Time")
        st.warning(f"Se usan **{MAX_EVENTS_FOR_ANIMATION:,}** de **{n_total:,}** eventos para que la animación cargue bien. El rango de fechas se mantiene.")

    min_date_ts = df_work['Time'].min()
    max_date_ts = df_work['Time'].max()
    min_date = min_date_ts.date() if hasattr(min_date_ts, 'date') else min_date_ts
    max_date = max_date_ts.date() if hasattr(max_date_ts, 'date') else max_date_ts

    highlights = _get_highlights(df_work)
    with st.expander("📌 Resumen: año más activo, sismo más fuerte, zona con más eventos", expanded=True):
        _render_narrative_panel(highlights, len(df_work))

    with st.expander("🗓️ Comparar dos periodos (antes vs después)", expanded=False):
        _render_period_comparison(df_work, min_date, max_date)

    c1, c2 = st.columns([2, 1])
    with c1:
        date_range = st.date_input(
            "Rango de fechas a animar",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="story_date_range"
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d, end_d = min_date, max_date

    with st.expander("🔍 Ir al sismo más fuerte"):
        if 'max_mag' in highlights:
            st.info(f"Sismo más fuerte: **{highlights['max_mag']['date']}** (magnitud **{highlights['max_mag']['mag']:.1f}**). Puedes ajustar el rango de fechas arriba para centrarlo.")

    # 6. Renderizar Animación
    ts_start = pd.Timestamp(start_d).tz_localize('UTC') if hasattr(pd.Timestamp(start_d), 'tz_localize') else pd.Timestamp(start_d)
    ts_end = pd.Timestamp(end_d).tz_localize('UTC') + pd.Timedelta(hours=23, minutes=59) if hasattr(pd.Timestamp(end_d), 'tz_localize') else pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59)
    try:
        if df_work['Time'].dt.tz is None:
            ts_start = ts_start.tz_localize(None)
            ts_end = ts_end.tz_localize(None)
    except Exception:
        pass

    df_filtered = df_work[(df_work['Time'] >= ts_start) & (df_work['Time'] <= ts_end)]

    if df_filtered.empty:
        st.warning("No hay eventos en este rango.")
        return

    cen_lat = float(df_filtered["Latitude"].mean())
    cen_lon = float(df_filtered["Longitude"].mean())
    map_zoom = 6 if use_narino_only else 2

    st.markdown("---")
    st.caption("**Rango de fechas:** elige arriba. Abajo: Play para animar o mueve la barra para ver un momento.")
    fig = _render_animated_map(df_filtered, start_d, end_d, center_lat=cen_lat, center_lon=cen_lon, zoom=map_zoom)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Total eventos en rango:** {len(df_filtered)}")
