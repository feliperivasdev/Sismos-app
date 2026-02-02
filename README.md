# 🌍 Monitor de Predicción Sísmica: Modelo Gutenberg-Richter

Una aplicación web interactiva desarrollada en **Streamlit** para el análisis estadístico de catálogos sísmicos. Esta herramienta permite cargar datos de terremotos, ajustar la **Ley de Gutenberg-Richter** y calcular probabilidades de riesgo sísmico futuro.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🚀 Características Principales

### 📊 Análisis Estadístico

- **Ajuste Automático G-R**: Calcula los parámetros $a$ (sismicidad) y $b$ (relación de frecuencias) utilizando regresión lineal sobre la frecuencia acumulada.
- **Métricas de Error**: Evalúa la calidad del ajuste con $R^2$ y $RMSE$ tanto en escala lineal como logarítmica.

### 🗺️ Visualización Geoespacial

- **Mapa de Calor**: Visualización interactiva de la densidad de sismos.
- **Distribución 3D**: Gráficos de dispersión para analizar Magnitud vs. Profundidad.

### 🔮 Predicción de Riesgo

- **Calculadora de Probabilidad**: Estima la probabilidad de ocurrencia de un sismo de cierta magnitud en un horizonte de tiempo específico (Modelo de Poisson).
- **Proyección de Tasas**: Tasa anual de eventos esperados.

### 🛠️ Herramientas de Datos

- **Mapeo de Columnas Inteligente**: Interfaz para asignar manualmente columnas de archivos CSV con formatos desconocidos.
- **Filtrado Temporal**: Detección automática de fechas y columnas `Time`.

## 📦 Instalación y Requisitos

Asegúrate de tener Python 3.8+ instalado.

1. **Clona este repositorio**

   ```bash
   git clone https://github.com/tu-usuario/sismos-app.git
   cd sismos-app
   ```

2. **Instala las dependencias**
   Puedes hacerlo manualmente con pip:
   ```bash
   pip install streamlit pandas numpy scipy scikit-learn plotly
   ```

## 🖥️ Cómo Ejecutar

Para iniciar la aplicación en tu navegador local, ejecuta el siguiente comando en la terminal:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

## 📂 Formato de Datos

La aplicación acepta archivos `.csv` con al menos las siguientes columnas (los nombres pueden variar, la app permite mapearlos):

- **Magnitud**: (Ej: `Magnitude`, `mag`, `mw`)
- **Tiempo**: (Ej: `Time`, `Date`, `timestamp`) - O columnas separadas de Año/Mes/Día.
- **(Opcional) Latitud/Longitud**: Para mapas.
- **(Opcional) Profundidad**: Para análisis 3D.

## 🧮 Metodología

El modelo se basa en la relación empírica:

$$ \log\_{10} N = a - bM $$

Donde:

- $N$ es el número acumulado de sismos con magnitud $\ge M$.
- $a$ es la productividad sísmica total.
- $b$ es la pendiente que indica la proporción entre sismos grandes y pequeños (típicamente cercano a 1.0).

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

---

Desarrollado con ❤️ usando [Streamlit](https://streamlit.io/).
