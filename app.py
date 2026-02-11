import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import storytelling_module

# Configuración de la página
st.set_page_config(page_title="Análisis Sísmico G-R", layout="wide")

st.title("🌍 Monitor de Predicción Sísmica: Modelo Gutenberg-Richter")
st.markdown("""
Esta herramienta ajusta la **Ley de Gutenberg-Richter** a datos sísmicos cargados dinámicamente.
Sube tu archivo `.csv` (debe contener una columna llamada `Magnitude`) para comenzar.
""")

# --- FASE 1: Carga de Datos ---
with st.sidebar:
    mode = st.radio("Modo de Visualización", ["Análisis General", "Storytelling"])
    st.divider()
    st.header("1. Configuración de Datos")
    uploaded_file = st.file_uploader("Cargar archivo CSV (Kaggle)", type=["csv"])
    
    # Selector de Modelo (Lineal vs Logarítmico según tu código)
    tipo_modelo = st.radio("Seleccionar enfoque de ajuste:", 
                           ["Ajuste G-R Estándar", "Regresión Lineal (Escala Log)"])

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
    
    # 2. Configuración de Columnas (Sidebar)
    st.sidebar.divider()
    st.sidebar.subheader("2. Mapeo de Columnas")
    
    with st.sidebar.expander("🛠️ Asignar Variables", expanded=True):
        st.info("Selecciona qué columna de tu archivo corresponde a cada variable.")
        
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
    
    st.success(f"Datos procesados: {len(df)} registros válidos.")
    
    # --- VISUALIZACIÓN POR PESTAÑAS ---
    if mode == "Storytelling":
        storytelling_module.render_storytelling(df)
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis General", "📈 Evolución Temporal", "🗺️ Mapa Geoespacial", "📉 Profundidad"])

    with tab1:
        st.subheader("Distribución de Magnitudes")
        # Optimization: Histogram sampling
        if len(df) > 10000:
            df_hist = df.sample(10000)
            st.caption("Visualizando muestra de 10,000 registros.")
        else:
            df_hist = df
            
        fig_hist = go.Figure(data=[go.Histogram(x=df_hist["Magnitude"], nbinsx=50, marker_color='indianred')])
        fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            st.subheader("Mapa de Calor Sísmico")
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
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox_center_lat=df_map.Latitude.mean(),
                mapbox_center_lon=df_map.Longitude.mean(),
                mapbox_zoom=2, height=500, margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No se encontraron columnas de Latitud/Longitud para el mapa.")

    with tab2:
        if 'Time' in df.columns:
            st.subheader("Sismograma: Evolución Temporal")
            
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
            fig_time.update_layout(
                title="Histórico de Magnitudes (Sismograma)",
                xaxis_title="Fecha", 
                yaxis_title="Magnitud", 
                height=400,
                template="plotly_dark", # TEMA OSCURO para resaltar el color brillante
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No se encontró columna de Fecha/Tiempo ('Time', 't', 'Date', etc.) para graficar.")

    with tab4:
        if 'Depth' in df.columns:
            st.subheader("Distribución de Profundidad")
            
            # Global Sampling for Depth Tab
            if len(df) > 5000:
                df_depth = df.sample(5000)
                st.caption("Visualizando muestra de 5,000 registros para optimizar.")
            else:
                df_depth = df
            
            fig_depth = go.Figure(data=[go.Histogram(x=df_depth["Depth"], nbinsx=30, marker_color='teal')])
            fig_depth.update_layout(title="Profundidad de los Sismos", xaxis_title="Profundidad (km)", yaxis_title="Frecuencia", height=400)
            st.plotly_chart(fig_depth, use_container_width=True)
            
            # Scatter Profundidad vs Magnitud
            fig_dvsm = go.Figure(go.Scatter(
                x=df_depth["Depth"], y=df_depth["Magnitude"], mode='markers',
                marker=dict(size=3, opacity=0.5)
            ))
            fig_dvsm.update_layout(title="Correlación: Profundidad vs Magnitud", xaxis_title="Profundidad", yaxis_title="Magnitud", height=400)
            st.plotly_chart(fig_dvsm, use_container_width=True)
        else:
            st.warning("No se encontró columna de Profundidad ('Depth').")

else:
    st.info("Esperando archivo CSV para ejecutar el modelo...")
    st.stop()

# --- FASE 2: Procesamiento del Modelo ---

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

# --- Dashboard de Métricas (UX: Resultados Clave) ---
st.divider()
st.subheader("2. Resultados del Ajuste (Comparativa)")

c_params, c_metrics_log, c_metrics_lin = st.columns(3)

with c_params:
    st.markdown("#### Parámetros G-R")
    st.metric("Valor 'a' (Sismicidad)", f"{a:.4f}")
    st.metric("Valor 'b' (Proporción)", f"{b:.4f}")

with c_metrics_log:
    st.markdown("#### Ajuste Logarítmico")
    st.caption("Ajuste sobre log10(N). Es lo que el modelo 've'.")
    st.metric("R² (Log)", f"{r2_log:.3f}")
    st.metric("RMSE (Log)", f"{rmse_log:.4f}")

with c_metrics_lin:
    st.markdown("#### Ajuste Lineal")
    st.caption("Ajuste sobre N (Eventos). Cómo se ve en la realidad.")
    st.metric("R² (Lineal)", f"{r2_linear:.3f}")
    st.metric("RMSE (Eventos)", f"{rmse_linear:.1f}")

with st.expander("ℹ️ ¿Por qué hay dos métricas?"):
    st.markdown("""
    La Ley de Gutenberg-Richter es logarítmica. 
    - **Ajuste Logarítmico**: Mide qué tan recta es la línea en el gráfico semilogarítmico. Generalmente es muy alto (>0.95).
    - **Ajuste Lineal**: Mide el error en el número real de terremotos. Puede ser menor porque pequeños errores en logaritmo significan grandes diferencias en cantidad de terremotos pequeños.
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

fig.update_layout(
    title="Curva de Gutenberg-Richter (Escala Logarítmica)",
    xaxis_title="Magnitud (M)",
    yaxis_title="log₁₀ N (Acumulado)",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- FASE 4: Predicción de Probabilidad ---
st.divider()
st.subheader("3. Calculadora de Probabilidad de Riesgo")

# Cálculo automático de años observados
years_obs_default = 50.0
if "Time" in df.columns:
    try:
        min_date = df["Time"].min()
        max_date = df["Time"].max()
        diff = max_date - min_date
        years_obs_calculated = diff.days / 365.25
        if years_obs_calculated > 0:
            years_obs_default = years_obs_calculated
    except:
        pass

c1, c2, c3 = st.columns(3)
with c1:
    mag_target = st.number_input("Magnitud objetivo (M)", min_value=4.0, max_value=10.0, value=6.0, step=0.1)
with c2:
    time_horizon = st.number_input("Horizonte de tiempo (años)", min_value=1, max_value=100, value=10)
with c3:
    st.markdown("###### Periodo de observación")
    years_obs = st.number_input("Años detectados en datos", value=float(f"{years_obs_default:.2f}"), min_value=0.1)

# Cálculo corregido
# N_val es el número TOTAL de eventos > M esperados en el periodo de observación (years_obs)
# Por tanto lambda (tasa anual) = N_val / years_obs
N_val_total = 10**(a - b * mag_target)
lambd = N_val_total / years_obs # Tasa anual de eventos > M
prob = 1 - np.exp(-lambd * time_horizon)

st.success(f"Probabilidad de ≥1 sismo de M ≥ {mag_target} en los próximos {time_horizon} años:")
st.markdown(f"## {prob*100:.2f}%")

if prob > 0.999:
    st.warning(f"⚠️ ¡Alta Probabilidad! Según el modelo, ocurren {lambd:.2f} sismos de M>={mag_target} por año promedio.")
else:
    st.info(f"Tasa anual estimada: {lambd:.4f} eventos/año.")