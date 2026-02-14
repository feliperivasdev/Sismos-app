import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import storytelling_module
import exploration_module
import educational_module
import alerts_module
import geo_comparison_module

# Configuración de la página
st.set_page_config(page_title="Análisis de Sismos", layout="wide")

# --- Personalización: tema (aplicado a gráficos Plotly)
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

# --- Encabezado claro para no expertos ---
st.title("🌍 Análisis de Sismos")
st.markdown("""
**¿Qué hace esta app?** Subes un archivo con datos de terremotos (CSV) y ves gráficos, mapas y un resumen de riesgo.  
No necesitas ser experto: si activas **Modo aprendizaje** en la barra lateral, verás explicaciones sencillas.
""")

# --- Sidebar simplificado: flujo en 2 pasos ---
with st.sidebar:
    st.header("📂 Paso 1: Sube tus datos")
    uploaded_file = st.file_uploader("Archivo CSV de sismos", type=["csv"], help="Tu archivo debe tener al menos magnitud y fecha.")
    st.caption("Ejemplo: datos de Kaggle o exportados de catálogos sísmicos.")

    st.divider()
    st.header("👀 Paso 2: ¿Qué quieres ver?")
    mode = st.radio(
        "Elige una vista",
        ["Gráficos y mapas", "Historia animada"],
        format_func=lambda x: "📊 Gráficos y mapas" if x == "Gráficos y mapas" else "🎬 Historia animada (play/pausa)",
        label_visibility="collapsed"
    )
    # Normalizar para el resto del código
    mode = "Análisis General" if mode == "Gráficos y mapas" else "Storytelling"

    # Valores por defecto para opciones avanzadas
    if "model_option" not in st.session_state:
        st.session_state["model_option"] = "Gutenberg-Richter"
    if "tipo_modelo" not in st.session_state:
        st.session_state["tipo_modelo"] = "Ajuste G-R Estándar"

    # Opciones que no abruman: en expanders
    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        st.caption("Solo si algo no funciona o quieres más control.")
        st.session_state["theme"] = st.radio("Tema", ["light", "dark"], horizontal=True, key="theme_radio", label_visibility="visible")
        educational_module.render_educational_sidebar()
        if st.session_state.get("educational_mode"):
            st.caption("Glosario: ver abajo en la página cuando esté cargado el archivo.")
        st.session_state["tipo_modelo"] = st.radio("Enfoque de ajuste", ["Ajuste G-R Estándar", "Regresión Lineal (Escala Log)"], index=0, key="tipo_modelo_radio", label_visibility="visible")
        st.session_state["model_option"] = st.selectbox("Modelo predictivo", ["Gutenberg-Richter", "Regresión Temporal"], key="model_option_select", label_visibility="visible")
    model_option = st.session_state["model_option"]
    tipo_modelo = st.session_state["tipo_modelo"]

def _export_chart_ui(fig, label):
    """Botón para exportar gráfico como HTML (compatible sin kaleido)."""
    try:
        html = fig.to_html(include_plotlyjs="cdn")
        st.download_button("📤 Exportar gráfico", html, file_name=f"{label}.html", mime="text/html", key=f"export_{label}")
    except Exception:
        pass


# Función auxiliar para encontrar índice por defecto
def get_column_index(columns, search_terms):
    """Devuelve el índice de la primera columna que coincida con alguno de los términos de búsqueda."""
    cols_lower = [c.lower().strip() for c in columns]
    for term in search_terms:
        if term in cols_lower:
            return cols_lower.index(term)
    return 0 # Default to first column if no match

@st.cache_data
def load_raw_data(file):
    """Carga el CSV 'crudo' intentando diferentes encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            file.seek(0)
            # Intentamos leer con el encoding actual
            df = pd.read_csv(file, encoding=encoding)
            
            # Validación básica de estructura
            if len(df.columns) < 2:
                # Si falló la detección de separador, intentamos engine python con separador automático
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine='python', encoding=encoding)
                
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
            
    # Último intento: Forzar lectura ignorando errores de caracteres
    file.seek(0)
    return pd.read_csv(file, sep=None, engine='python', encoding_errors='replace')

if uploaded_file is not None:
    # 1. Carga Cruda
    raw_df = load_raw_data(uploaded_file)
    all_columns = list(raw_df.columns)
    
    # 2. Configuración de columnas (solo si hace falta)
    st.sidebar.divider()
    with st.sidebar.expander("🛠️ Indicar columnas del archivo", expanded=False):
        st.caption("Abre esto solo si la app no reconoció bien tus columnas (magnitud, fecha, etc.).")
        
        # --- MAGNITUD (Obligatorio) ---
        mag_terms = ["magnitude", "magnitud", "mag", "m", "magnitude_val", 
                    "eq primary", "mw magnitude", "ms magnitude", "mb magnitude", 
                    "ml magnitude", "mfa magnitude", "unknown magnitude"]
        
        col_mag_idx = get_column_index(all_columns, mag_terms)
        col_mag = st.selectbox("Magnitud (Obligatorio)", all_columns, index=col_mag_idx)
        
        # --- TIEMPO (Obligatorio) ---
        time_terms = ["time", "date", "fecha", "datetime", "timestamp", "t", "sismo_time"]
        
        use_multi_date = st.checkbox("¿Fecha en columnas separadas? (Año, Mes, Día)")
        
        col_time = None
        col_year, col_month, col_day = None, None, None
        
        if use_multi_date:
            c_y = get_column_index(all_columns, ["year", "año", "ano", "yyyy"])
            c_m = get_column_index(all_columns, ["month", "mes", "mm"])
            c_d = get_column_index(all_columns, ["day", "dia", "dd"])
            
            col_year = st.selectbox("Año", all_columns, index=c_y)
            col_month = st.selectbox("Mes", all_columns, index=c_m)
            col_day = st.selectbox("Día", all_columns, index=c_d)
            # Opcionales: Hora/Minuto (se podría agregar después si el usuario lo pide)
        else:
            col_time_idx = get_column_index(all_columns, time_terms)
            col_time = st.selectbox("Fecha/Hora (Obligatorio)", all_columns, index=col_time_idx)

        # --- PROFUNDIDAD (Opcional) ---
        depth_terms = ["depth", "depth_km", "profundidad", "prof", "focal depth"]
        col_depth_idx = get_column_index(all_columns, depth_terms)
        # Hack: if no match found, logic usually returns 0. Check if 0 is actually a depth term.
        # Better: Add "No existe/None" option.
        
        # Opciones con 'None' al principio
        opts_optional = ["(No Incluir)"] + all_columns
        
        # Recalcular indices para offset +1
        def get_opt_index(cols, terms):
            idx = get_column_index(cols, terms)
            # Verificar si realmente hubo match. get_column_index retorna 0 si no match.
            # Si cols[0] NO está en terms, entonces fue fallback.
            first_col_is_match = str(cols[0]).lower().strip() in terms
            
            # Si retorna 0 y la col 0 NO es match, devolvemos 0 (que es "(No Incluir)")
            if idx == 0 and not first_col_is_match:
                return 0
            return idx + 1

        d_idx = get_opt_index(all_columns, depth_terms)
        col_depth_sel = st.selectbox("Profundidad (Opcional)", opts_optional, index=d_idx)
        col_depth = col_depth_sel if col_depth_sel != "(No Incluir)" else None

        # --- LAT/LON (Opcional) ---
        lat_terms = ["latitude", "lat", "latitud"]
        lon_terms = ["longitude", "lon", "long", "longitud", "lng"]
        
        lat_idx = get_opt_index(all_columns, lat_terms)
        lon_idx = get_opt_index(all_columns, lon_terms)
        
        col_lat_sel = st.selectbox("Latitud (Opcional)", opts_optional, index=lat_idx)
        col_lon_sel = st.selectbox("Longitud (Opcional)", opts_optional, index=lon_idx)
        
        col_lat = col_lat_sel if col_lat_sel != "(No Incluir)" else None
        col_lon = col_lon_sel if col_lon_sel != "(No Incluir)" else None

    # 3. Procesamiento y Estandarización
    # Creamos un nuevo DF solo con lo que nos interesa, renombrado
    df = pd.DataFrame()
    
    # Magnitud
    df["Magnitude"] = raw_df[col_mag]
    
    # Tiempo
    try:
        if use_multi_date:
            # Construir fecha desde partes
            date_dict = {
                "year": raw_df[col_year],
                "month": raw_df[col_month],
                "day": raw_df[col_day]
            }
            df["Time"] = pd.to_datetime(pd.DataFrame(date_dict), errors='coerce', utc=True)
        else:
            df["Time"] = pd.to_datetime(raw_df[col_time], errors='coerce', utc=True)
        
        df = df.dropna(subset=["Time"])
    except Exception as e:
        st.error(f"Error procesando fechas: {e}")
        st.stop()
        
    # Profundidad
    if col_depth:
        df["Depth"] = raw_df[col_depth]
        
    # Coordenadas
    if col_lat and col_lon:
        df["Latitude"] = raw_df[col_lat]
        df["Longitude"] = raw_df[col_lon]

    # Limpieza final básica
    df = df.dropna(subset=["Magnitude"])

    # --- Mensaje claro para el usuario ---
    n = len(df)
    mag_min_d = float(df["Magnitude"].min())
    mag_max_d = float(df["Magnitude"].max())
    st.success(f"**Listo.** Se cargaron **{n:,}** sismos (magnitud entre {mag_min_d:.1f} y {mag_max_d:.1f}).")
    st.info("👆 **Siguiente:** usa las pestañas de abajo para ver gráficos, mapa y riesgo. ¿Primera vez? Activa **Modo aprendizaje** en *Opciones avanzadas* (barra lateral).")

    # --- Filtro principal siempre visible ---
    st.sidebar.divider()
    min_mag = st.sidebar.slider("Solo sismos con magnitud ≥", 0.0, 10.0, 4.0, help="Oculta sismos muy pequeños.")
    df = df[df["Magnitude"] >= min_mag]

    # --- Búsqueda de un sismo y alertas: opcionales, en expander ---
    with st.sidebar.expander("🔍 Buscar un sismo o ver alertas", expanded=False):
        df, selected_event = exploration_module.render_exploration_sidebar(df)
        alerts_module.render_alerts_sidebar()

    # --- VISUALIZACIÓN POR PESTAÑAS ---
    if mode == "Storytelling":
        storytelling_module.render_storytelling(df)
        if st.session_state.get("educational_mode"):
            educational_module.render_glossary()
        st.stop()

    plotly_template = "plotly_dark" if st.session_state.get("theme") == "dark" else "plotly_white"

    # Eventos similares en el área principal (más práctico que en la barra)
    if selected_event is not None:
        with st.expander("🧠 Eventos similares al sismo que elegiste", expanded=True):
            exploration_module.render_similar_events_main(df, selected_event)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resumen", "📈 En el tiempo", "🗺️ Mapa", "📉 Profundidad",
        "📈 Para expertos", "🗺️ Comparar regiones"
    ])

    with tab1:
        st.subheader("Resumen de magnitudes")
        st.caption("Cuántos sismos hay de cada magnitud (más alto = más frecuente).")
        if st.session_state.get("educational_mode"):
            st.caption("ℹ️ La ley de Gutenberg-Richter modela esta distribución.")
        # Optimization: Histogram sampling
        if len(df) > 10000:
            df_hist = df.sample(10000)
            st.caption("Visualizando muestra de 10,000 registros.")
        else:
            df_hist = df

        fig_hist = go.Figure(data=[go.Histogram(x=df_hist["Magnitude"], nbinsx=50, marker_color='indianred')])
        if selected_event is not None:
            fig_hist.add_vline(x=float(selected_event["Magnitude"]), line_dash="dash", line_color="gold", annotation_text="Evento seleccionado")
        fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template=plotly_template)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            st.subheader("Dónde ocurrieron los sismos")
            st.caption("Mapa de calor: más color = más actividad en esa zona.")
            LIMIT_MAPA = 5000
            if len(df) > LIMIT_MAPA:
                df_map = df.sample(LIMIT_MAPA)
                st.caption(f"Visualizando {LIMIT_MAPA} puntos representativos.")
            else:
                df_map = df

            fig_map = go.Figure(go.Densitymapbox(
                lat=df_map.Latitude, lon=df_map.Longitude, z=df_map.Magnitude,
                radius=10, opacity=0.7, colorscale="Viridis"
            ))
            # Zonas críticas (eventos que superan umbrales de alerta)
            mag_above = st.session_state.get("alert_mag_above", 6.0)
            depth_below = st.session_state.get("alert_depth_below", 70.0)
            critical_mask = alerts_module.get_critical_mask(df_map, mag_above, depth_below)
            if critical_mask.any():
                df_crit = df_map[critical_mask]
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_crit.Latitude, lon=df_crit.Longitude, mode='markers',
                    marker=dict(size=12, color='red'), name='Zonas críticas'
                ))
            # Evento seleccionado resaltado
            if selected_event is not None and 'Latitude' in selected_event.index and 'Longitude' in selected_event.index:
                fig_map.add_trace(go.Scattermapbox(
                    lat=[selected_event["Latitude"]], lon=[selected_event["Longitude"]],
                    mode='markers', marker=dict(size=18, color='gold'),
                    name='Evento seleccionado'
                ))
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_center_lat=df_map.Latitude.mean(),
                mapbox_center_lon=df_map.Longitude.mean(),
                mapbox_zoom=2, height=500, margin=dict(l=0, r=0, t=0, b=0),
                template=plotly_template
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No se encontraron columnas de Latitud/Longitud para el mapa.")

    with tab2:
        if 'Time' in df.columns:
            st.subheader("Cómo cambió la actividad en el tiempo")
            st.caption("Línea en el tiempo: cada punto es un sismo (eje vertical = magnitud).")
            
            # Ordenamos por fecha para que la línea tenga sentido
            df_time = df.sort_values(by="Time")
            
            # Sampling temporal inteligente para gráficos de LINEA
            # Si hay demasiados puntos, el navegador muere al tratar de hacer SVG de una linea infinita.
            # Tomamos un resampleo si es muy grande.
            if len(df_time) > 5000:
                # Opción A: Slicing simple (rápido y efectivo para tendencias)
                step = len(df_time) // 5000
                df_plot = df_time.iloc[::step]
                st.caption(f"Visualizando tendencia con {len(df_plot)} puntos (resampleado por optimización).")
            else:
                df_plot = df_time
                
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=df_plot['Time'], y=df_plot['Magnitude'],
                mode='lines', 
                line=dict(width=1, color='#00CC96'), # CAMBIO DE COLOR: Cyan/Green brillante para contraste
                name='Sismicidad'
            ))
            if selected_event is not None and "Time" in selected_event.index:
                fig_time.add_trace(go.Scatter(
                    x=[selected_event["Time"]], y=[selected_event["Magnitude"]],
                    mode='markers', marker=dict(size=14, color='gold', symbol='star'),
                    name='Evento seleccionado'
                ))
            fig_time.update_layout(
                title="Histórico de Magnitudes (Sismograma)",
                xaxis_title="Fecha",
                yaxis_title="Magnitud",
                height=400,
                template=plotly_template,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No se encontró columna de Fecha/Tiempo ('Time', 't', 'Date', etc.) para graficar.")

    with tab4:
        if 'Depth' in df.columns:
            st.subheader("A qué profundidad ocurrieron")
            st.caption("Profundidad en km (0 = superficie; más abajo = más profundo).")
            if st.session_state.get("educational_mode"):
                st.caption("ℹ️ Sismos superficiales suelen sentirse más.")
            if len(df) > 5000:
                df_depth = df.sample(5000)
                st.caption("Visualizando muestra de 5,000 registros para optimizar.")
            else:
                df_depth = df

            fig_depth = go.Figure(data=[go.Histogram(x=df_depth["Depth"], nbinsx=30, marker_color='teal')])
            if selected_event is not None and "Depth" in selected_event.index and pd.notna(selected_event.get("Depth")):
                fig_depth.add_vline(x=float(selected_event["Depth"]), line_dash="dash", line_color="gold", annotation_text="Seleccionado")
            fig_depth.update_layout(title="Profundidad de los Sismos", xaxis_title="Profundidad (km)", yaxis_title="Frecuencia", height=400, template=plotly_template)
            st.plotly_chart(fig_depth, use_container_width=True)

            # Scatter Profundidad vs Magnitud (con correlación)
            fig_dvsm = go.Figure(go.Scatter(
                x=df_depth["Depth"], y=df_depth["Magnitude"], mode='markers',
                marker=dict(size=3, opacity=0.5)
            ))
            if selected_event is not None and "Depth" in selected_event.index and pd.notna(selected_event.get("Depth")):
                fig_dvsm.add_trace(go.Scatter(
                    x=[selected_event["Depth"]], y=[selected_event["Magnitude"]],
                    mode='markers', marker=dict(size=14, color='gold', symbol='star'),
                    name='Evento seleccionado'
                ))
            r_vals = df_depth[["Depth", "Magnitude"]].dropna()
            if len(r_vals) > 2:
                r_pearson, p_val = pearsonr(r_vals["Depth"], r_vals["Magnitude"])
                fig_dvsm.update_layout(
                    title=f"Correlación: Profundidad vs Magnitud (r = {r_pearson:.3f})",
                    xaxis_title="Profundidad (km)", yaxis_title="Magnitud", height=400,
                    template=plotly_template
                )
            else:
                fig_dvsm.update_layout(title="Correlación: Profundidad vs Magnitud", xaxis_title="Profundidad (km)", yaxis_title="Magnitud", height=400, template=plotly_template)
            st.plotly_chart(fig_dvsm, use_container_width=True)
        else:
            st.warning("No se encontró columna de Profundidad ('Depth').")

    with tab5:
        st.subheader("Números y estadísticas (para análisis detallado)")
        st.caption("Media, mediana, desviación y correlaciones. Útil para informes o tareas.")
        # Estadísticas descriptivas
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Magnitud: media", f"{df['Magnitude'].mean():.3f}")
            st.metric("Magnitud: mediana", f"{df['Magnitude'].median():.3f}")
        with c2:
            st.metric("Magnitud: desv. estándar", f"{df['Magnitude'].std():.3f}")
            q1, q3 = df['Magnitude'].quantile(0.25), df['Magnitude'].quantile(0.75)
            st.metric("Rango intercuartílico (Q1–Q3)", f"{q1:.2f} – {q3:.2f}")
        with c3:
            st.metric("Mínimo", f"{df['Magnitude'].min():.2f}")
            st.metric("Máximo", f"{df['Magnitude'].max():.2f}")
        if 'Depth' in df.columns:
            d = df[["Depth", "Magnitude"]].dropna()
            if len(d) > 2:
                r_pearson, p_val = pearsonr(d["Depth"], d["Magnitude"])
                st.markdown("#### Correlación Profundidad–Magnitud")
                st.markdown(f"**Coeficiente de Pearson:** r = {r_pearson:.4f} (p = {p_val:.4f}).")
                st.caption("Intervalos de confianza para r requieren bootstrap o transformación; p < 0.05 sugiere correlación significativa.")

    with tab6:
        st.caption("Compara dos regiones del mundo (ej. Chile vs Japón) y mira qué zona tiene más sismos.")
        geo_comparison_module.render_geo_comparison(df)

else:
    st.info("**Paso 1:** Sube un archivo CSV con datos de sismos en la barra lateral (←). Suele tener columnas como magnitud, fecha y opcionalmente latitud, longitud y profundidad.")
    st.stop()

# --- FASE 2: Procesamiento del Modelo ---

if model_option == "Regresión Temporal":
    if 'Time' in df.columns:
        df_year = df.copy()
        df_year["Year"] = df_year["Time"].dt.year
        yearly_counts = df_year.groupby("Year").size().reset_index(name="Eventos")
        
        if len(yearly_counts) >= 2:
            slope_t, intercept_t, r_t, _, _ = linregress(yearly_counts["Year"], yearly_counts["Eventos"])

            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=yearly_counts["Year"],
                y=yearly_counts["Eventos"],
                mode="markers",
                name="Datos Reales"
            ))

            fig_temp.add_trace(go.Scatter(
                x=yearly_counts["Year"],
                y=intercept_t + slope_t * yearly_counts["Year"],
                mode="lines",
                name="Modelo Lineal"
            ))

            fig_temp.update_layout(title="Regresión Temporal de Eventos por Año")
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.warning("Se requieren al menos 2 años de datos para realizar la regresión temporal.")
    else:
        st.error("No se encontró la columna 'Time' para el análisis temporal.")
    st.stop()

# Preparar datos comunes
magnitudes = np.sort(df["Magnitude"].values)[::-1]
N_obs = np.arange(1, len(magnitudes) + 1)
log_N_obs = np.log10(N_obs)

# Modelo Único: Regresión Lineal sobre log(N)
# Log10(N) = a - bM
slope, intercept, r, p, se = linregress(magnitudes, log_N_obs)
a = intercept
b = -slope

# Cálculo de Predicciones y Métricas para AMBOS escenarios

# 1. Escenario Estándar (Comparación en escala Lineal N)
# N = 10^(a - bM)
N_pred_linear = 10**(intercept + slope * magnitudes)
rmse_linear = np.sqrt(mean_squared_error(N_obs, N_pred_linear))
r2_linear = r2_score(N_obs, N_pred_linear)

# 2. Escenario Logarítmico (Comparación en escala Log N)
# log(N) = a - bM
log_N_pred = intercept + slope * magnitudes
rmse_log = np.sqrt(mean_squared_error(log_N_obs, log_N_pred))
r2_log = r2_score(log_N_obs, log_N_pred)

st.divider()
st.subheader("¿Qué tan bien explica el modelo tus datos?")
st.caption("Estos números resumen la calidad del ajuste (más cerca de 1 en R² = mejor).")

c_params, c_metrics_log, c_metrics_lin = st.columns(3)

with c_params:
    st.markdown("#### Parámetros del modelo")
    st.metric("Valor 'a' (sismicidad)", f"{a:.4f}")
    st.metric("Valor 'b' (proporción)", f"{b:.4f}")

with c_metrics_log:
    st.markdown("#### Ajuste en escala log")
    st.metric("R² (Log)", f"{r2_log:.3f}")
    st.metric("RMSE (Log)", f"{rmse_log:.4f}")

with c_metrics_lin:
    st.markdown("#### Ajuste en número de eventos")
    st.metric("R² (Lineal)", f"{r2_linear:.3f}")
    st.metric("RMSE (Eventos)", f"{rmse_linear:.1f}")

with st.expander("ℹ️ ¿Qué significan R² y RMSE?"):
    st.markdown("""
    **R²** indica qué parte de los datos explica el modelo (0 a 1; 1 = perfecto).  
    **RMSE** es el error típico (cuanto más bajo, mejor).  
    El modelo de Gutenberg-Richter trabaja en escala logarítmica; por eso se muestran ambas formas de medir el ajuste.
    """)

# --- FASE 3: Visualización Interactiva ---

fig = go.Figure()

# Optimización de renderizado
LIMIT_PUNTOS = 10000
if len(magnitudes) > LIMIT_PUNTOS:
    step = len(magnitudes) // LIMIT_PUNTOS
    mag_plot = magnitudes[::step]
    y_plot_obs = log_N_obs[::step] # Usaremos Log para el plot principal por defecto
    st.caption(f"ℹ️ Visualizando {len(mag_plot)} puntos representativos.")
else:
    mag_plot = magnitudes
    y_plot_obs = log_N_obs

# Datos observados
fig.add_trace(go.Scatter(
    x=mag_plot, y=y_plot_obs,
    mode='markers', name='Datos Observados (Log N)',
    marker=dict(size=5, color='blue', opacity=0.5)
))

# Línea de Ajuste
fig.add_trace(go.Scatter(
    x=magnitudes, y=log_N_pred,
    mode='lines', name=f'Modelo G-R (b={b:.2f})',
    line=dict(color='red', width=3)
))

plotly_template_main = "plotly_dark" if st.session_state.get("theme") == "dark" else "plotly_white"
fig.update_layout(
    title="Gráfico del modelo: cuántos sismos de cada magnitud (línea = predicción)",
    xaxis_title="Magnitud",
    yaxis_title="Cantidad acumulada (escala log)",
    template=plotly_template_main,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
_export_chart_ui(fig, "gutenberg_richter")

# --- Calculadora de probabilidad (lenguaje claro) ---
st.divider()
st.subheader("¿Qué probabilidad hay de un sismo fuerte en los próximos años?")
st.caption("El modelo estima la probabilidad según tus datos. Puedes cambiar magnitud y número de años.")

years_obs_default = 50.0
if "Time" in df.columns:
    try:
        min_date = df["Time"].min()
        max_date = df["Time"].max()
        diff = max_date - min_date
        years_obs_calculated = diff.days / 365.25
        if years_obs_calculated > 0:
            years_obs_default = years_obs_calculated
    except Exception:
        pass

c1, c2, c3 = st.columns(3)
with c1:
    mag_target = st.number_input("Magnitud que te interesa (ej. 6 = fuerte)", min_value=4.0, max_value=10.0, value=6.0, step=0.1)
with c2:
    time_horizon = st.number_input("Próximos cuántos años", min_value=1, max_value=100, value=10)
with c3:
    years_obs = st.number_input("Años que cubren tus datos", value=float(f"{years_obs_default:.2f}"), min_value=0.1)

# Cálculo corregido
# N_val es el número TOTAL de eventos > M esperados en el periodo de observación (years_obs)
# Por tanto lambda (tasa anual) = N_val / years_obs
N_val_total = 10**(a - b * mag_target)
lambd = N_val_total / years_obs # Tasa anual de eventos > M
prob = 1 - np.exp(-lambd * time_horizon)

st.success(f"Probabilidad de que ocurra al menos un sismo de magnitud ≥ {mag_target} en los próximos **{time_horizon}** años:")
st.markdown(f"## {prob*100:.2f}%")

if prob > 0.999:
    st.warning(f"Según el modelo, en promedio ocurren {lambd:.2f} sismos de esa magnitud o mayor por año.")
else:
    st.info(f"Tasa estimada: {lambd:.4f} sismos de esa magnitud (o mayor) por año.")

st.subheader("🚦 Resumen de riesgo")
if prob < 0.2:
    st.success("🟢 Riesgo bajo")
elif prob < 0.5:
    st.warning("🟡 Riesgo moderado")
else:
    st.error("🔴 Riesgo alto")
st.caption("Basado en la probabilidad que calculamos arriba.")

# --- Módulo de Alertas (configurables + mensajes contextuales) ---
alerts_module.render_alerts_section(df)