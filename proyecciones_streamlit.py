import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(page_title="Proyecciones 2025-2026", layout="wide", initial_sidebar_state="collapsed")

# CSS para agrandar MUCHO más fuentes y números
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 28px;
    }
    h1 {
        font-size: 64px !important;
        font-weight: bold !important;
    }
    h2 {
        font-size: 56px !important;
        font-weight: bold !important;
    }
    h3 {
        font-size: 44px !important;
        font-weight: bold !important;
    }
    [data-testid="metric-container"] {
        font-size: 36px;
    }
    [data-testid="metric-container"] > div:nth-child(1) {
        font-size: 32px;
    }
    [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 56px !important;
        font-weight: bold !important;
    }
    .stDataFrame {
        font-size: 28px !important;
    }
    table {
        font-size: 28px !important;
    }
    [data-testid="stDataFrame"] thead {
        font-size: 32px !important;
        font-weight: bold !important;
    }
    [data-testid="stDataFrame"] tbody {
        font-size: 28px !important;
    }
    [data-testid="stDataFrame"] td {
        font-size: 28px !important;
        padding: 15px !important;
    }
    [data-testid="stDataFrame"] th {
        font-size: 30px !important;
        padding: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Proyecciones 2025-2026")
st.markdown("### Análisis de 3 escenarios de pronóstico para proyectar valores hasta 2026")
st.markdown("---")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@st.cache_data
def calculate_percentage_rmse(y_true, y_pred):
    rmse = calculate_rmse(y_true, y_pred)
    mean_y_true = np.mean(y_true)
    return (rmse / mean_y_true) * 100

@st.cache_data
def calculate_yearly_total(forecast_df, year):
    """Calcula el total anual sumando los valores mensuales"""
    yearly = forecast_df[forecast_df['ds'].dt.year == year]['yhat'].sum()
    return yearly

# ============================================================================
# CARGAR DATOS
# ============================================================================
st.sidebar.header("📁 Carga tus datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Leer archivo
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        df['Valor'] = df['Valor'].astype(str).str.replace(',', '').astype(float)
    else:
        df = pd.read_excel(uploaded_file)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
    
    # Generar fechas
    date_index = pd.date_range(start='2002-01-01', periods=len(df), freq='MS')
    df['ds'] = date_index
    df['y'] = df['Valor']
    df_prophet = df[['ds', 'y']].copy()
    
    st.sidebar.success(f"✅ {len(df_prophet)} registros cargados")
    st.sidebar.info(f"Período: {df_prophet['ds'].min().strftime('%B %Y')} a {df_prophet['ds'].max().strftime('%B %Y')}")
    
    # ============================================================================
    # SECCIÓN 1: DATOS HISTÓRICOS
    # ============================================================================
    st.header("1️⃣ Datos Históricos")
    st.markdown("Serie histórica de valores desde 2002 hasta septiembre 2025")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<h3>Total de registros</h3><h2>{len(df_prophet)}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3>Valor máximo</h3><h2>{df_prophet['y'].max():,.0f}</h2>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h3>Valor mínimo</h3><h2>{df_prophet['y'].min():,.0f}</h2>", unsafe_allow_html=True)
    
    # Gráfico histórico con Plotly
    fig_historico = go.Figure()
    fig_historico.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        name='Valor real',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_historico.update_layout(
        title="Serie Histórica 2002-2025",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_white",
        height=800,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36)))
    )
    st.plotly_chart(fig_historico, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================================================
    # ENTRENAR MODELOS
    # ============================================================================
    with st.spinner("⏳ Entrenando los 3 modelos..."):
        
        # MODELO 1: 2002
        df_train_2002 = df_prophet.copy()
        model_2002 = Prophet(seasonality_mode='additive')
        model_2002.fit(df_train_2002)
        future_2002 = model_2002.make_future_dataframe(periods=15, freq='MS')
        forecast_2002 = model_2002.predict(future_2002)
        
        forecast_2002_train = forecast_2002[forecast_2002['ds'].isin(df_train_2002['ds'])].copy()
        rmse_2002 = calculate_rmse(df_train_2002['y'].values, forecast_2002_train['yhat'].values)
        percentage_rmse_2002 = calculate_percentage_rmse(df_train_2002['y'].values, forecast_2002_train['yhat'].values)
        
        # MODELO 2: 2010
        df_train_2010 = df_prophet[df_prophet['ds'] >= '2010-01-01'].copy()
        model_2010 = Prophet(changepoint_prior_scale=1, seasonality_mode='multiplicative')
        model_2010.fit(df_train_2010)
        future_2010 = model_2010.make_future_dataframe(periods=15, freq='MS')
        forecast_2010 = model_2010.predict(future_2010)
        
        forecast_2010_train = forecast_2010[forecast_2010['ds'].isin(df_train_2010['ds'])].copy()
        rmse_2010 = calculate_rmse(df_train_2010['y'].values, forecast_2010_train['yhat'].values)
        percentage_rmse_2010 = calculate_percentage_rmse(df_train_2010['y'].values, forecast_2010_train['yhat'].values)
        
        # MODELO 3: 2021
        df_train_2021 = df_prophet[df_prophet['ds'] >= '2021-01-01'].copy()
        model_2021 = Prophet(changepoint_prior_scale=1.0, seasonality_mode='multiplicative')
        model_2021.fit(df_train_2021)
        future_2021 = model_2021.make_future_dataframe(periods=15, freq='MS')
        forecast_2021 = model_2021.predict(future_2021)
        
        forecast_2021_train = forecast_2021[forecast_2021['ds'].isin(df_train_2021['ds'])].copy()
        rmse_2021 = calculate_rmse(df_train_2021['y'].values, forecast_2021_train['yhat'].values)
        percentage_rmse_2021 = calculate_percentage_rmse(df_train_2021['y'].values, forecast_2021_train['yhat'].values)
        
        st.success("✅ Modelos entrenados correctamente")
    
    # ============================================================================
    # SECCIÓN 2: MODELO 2002
    # ============================================================================
    st.header("2️⃣ Modelo 2002 - Entrenamiento con todos los datos (Aditivo)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3>RMSE</h3><h2>{rmse_2002:,.0f}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3>%RMSE</h3><h2>{percentage_rmse_2002:.2f}%</h2>", unsafe_allow_html=True)
    
    # Gráfico pronóstico Modelo 2002
    fig_2002 = go.Figure()
    fig_2002.add_trace(go.Scatter(
        x=forecast_2002['ds'],
        y=forecast_2002['yhat'],
        mode='lines',
        name='Pronóstico',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_2002.add_trace(go.Scatter(
        x=forecast_2002['ds'],
        y=forecast_2002['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(31, 119, 180, 0)',
        showlegend=False
    ))
    fig_2002.add_trace(go.Scatter(
        x=forecast_2002['ds'],
        y=forecast_2002['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(31, 119, 180, 0)',
        name='Intervalo 95%',
        fillcolor='rgba(31, 119, 184, 0.2)'
    ))
    fig_2002.update_layout(
        title="Pronóstico Modelo 2002",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_white",
        height=800,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36)))
    )
    st.plotly_chart(fig_2002, use_container_width=True)
    
    # Componentes Modelo 2002
    fig_comp_2002 = model_2002.plot_components(forecast_2002, figsize=(14, 10))
    plt.tick_params(labelsize=16)
    for ax in fig_comp_2002.axes:
        ax.tick_params(labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)
    st.pyplot(fig_comp_2002)
    
    # Pronóstico 15 meses
    with st.expander("📋 Pronóstico 15 meses - Modelo 2002"):
        tabla_2002 = forecast_2002[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15).copy()
        tabla_2002['ds'] = tabla_2002['ds'].dt.strftime('%Y-%m-%d')
        st.dataframe(tabla_2002, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================================
    # SECCIÓN 3: MODELO 2010
    # ============================================================================
    st.header("3️⃣ Modelo 2010 - Entrenamiento desde 2010 (Multiplicativo)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3>RMSE</h3><h2>{rmse_2010:,.0f}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3>%RMSE</h3><h2>{percentage_rmse_2010:.2f}%</h2>", unsafe_allow_html=True)
    
    # Gráfico pronóstico Modelo 2010
    fig_2010 = go.Figure()
    fig_2010.add_trace(go.Scatter(
        x=forecast_2010['ds'],
        y=forecast_2010['yhat'],
        mode='lines',
        name='Pronóstico',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig_2010.add_trace(go.Scatter(
        x=forecast_2010['ds'],
        y=forecast_2010['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(255, 127, 14, 0)',
        showlegend=False
    ))
    fig_2010.add_trace(go.Scatter(
        x=forecast_2010['ds'],
        y=forecast_2010['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255, 127, 14, 0)',
        name='Intervalo 95%',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    fig_2010.update_layout(
        title="Pronóstico Modelo 2010",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_white",
        height=800,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36)))
    )
    st.plotly_chart(fig_2010, use_container_width=True)
    
    # Componentes Modelo 2010
    fig_comp_2010 = model_2010.plot_components(forecast_2010, figsize=(14, 10))
    plt.tick_params(labelsize=16)
    for ax in fig_comp_2010.axes:
        ax.tick_params(labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)
    st.pyplot(fig_comp_2010)
    
    # Pronóstico 15 meses
    with st.expander("📋 Pronóstico 15 meses - Modelo 2010"):
        tabla_2010 = forecast_2010[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15).copy()
        tabla_2010['ds'] = tabla_2010['ds'].dt.strftime('%Y-%m-%d')
        st.dataframe(tabla_2010, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================================
    # SECCIÓN 4: MODELO 2021
    # ============================================================================
    st.header("4️⃣ Modelo 2021 - Entrenamiento desde punto de quiebre (Multiplicativo)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3>RMSE</h3><h2>{rmse_2021:,.0f}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3>%RMSE</h3><h2>{percentage_rmse_2021:.2f}%</h2>", unsafe_allow_html=True)
    
    # Gráfico pronóstico Modelo 2021
    fig_2021 = go.Figure()
    fig_2021.add_trace(go.Scatter(
        x=forecast_2021['ds'],
        y=forecast_2021['yhat'],
        mode='lines',
        name='Pronóstico',
        line=dict(color='#2ca02c', width=2)
    ))
    fig_2021.add_trace(go.Scatter(
        x=forecast_2021['ds'],
        y=forecast_2021['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(44, 160, 44, 0)',
        showlegend=False
    ))
    fig_2021.add_trace(go.Scatter(
        x=forecast_2021['ds'],
        y=forecast_2021['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(44, 160, 44, 0)',
        name='Intervalo 95%',
        fillcolor='rgba(44, 160, 44, 0.2)'
    ))
    fig_2021.update_layout(
        title="Pronóstico Modelo 2021",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_white",
        height=800,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36)))
    )
    st.plotly_chart(fig_2021, use_container_width=True)
    
    # Componentes Modelo 2021
    fig_comp_2021 = model_2021.plot_components(forecast_2021, figsize=(14, 10))
    plt.tick_params(labelsize=16)
    for ax in fig_comp_2021.axes:
        ax.tick_params(labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)
    st.pyplot(fig_comp_2021)
    
    # Pronóstico 15 meses
    with st.expander("📋 Pronóstico 15 meses - Modelo 2021"):
        tabla_2021 = forecast_2021[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15).copy()
        tabla_2021['ds'] = tabla_2021['ds'].dt.strftime('%Y-%m-%d')
        st.dataframe(tabla_2021, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================================
    # SECCIÓN 5: COMPARACIÓN DE MODELOS
    # ============================================================================
    st.header("5️⃣ Comparación de Modelos")
    
    # Tabla comparativa RMSE - con formato
    comparacion = pd.DataFrame({
        'Modelo': ['Modelo 2002', 'Modelo 2010', 'Modelo 2021'],
        'RMSE': [f"{rmse_2002:,.1f}", f"{rmse_2010:,.1f}", f"{rmse_2021:,.1f}"],
        '%RMSE': [f"{percentage_rmse_2002:.1f}%", f"{percentage_rmse_2010:.1f}%", f"{percentage_rmse_2021:.1f}%"]
    })
    
    st.subheader("Métricas de Error")
    comparacion_html = comparacion.to_html(index=False, escape=False)
    st.markdown(f"""
    <div style="font-size: 32px; padding: 30px; line-height: 2.5;">
    {comparacion_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico comparativo
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        name='Valor real',
        line=dict(color='black', width=2.5)
    ))
    
    fig_comp.add_trace(go.Scatter(
        x=forecast_2002['ds'],
        y=forecast_2002['yhat'],
        mode='lines',
        name='Modelo 2002',
        line=dict(color='#1f77b4', dash='dash', width=1.5)
    ))
    
    fig_comp.add_trace(go.Scatter(
        x=forecast_2010['ds'],
        y=forecast_2010['yhat'],
        mode='lines',
        name='Modelo 2010',
        line=dict(color='#ff7f0e', dash='dash', width=1.5)
    ))
    
    fig_comp.add_trace(go.Scatter(
        x=forecast_2021['ds'],
        y=forecast_2021['yhat'],
        mode='lines',
        name='Modelo 2021 (Mejor)',
        line=dict(color='#d62728', width=2.5)
    ))
    
    fig_comp.update_layout(
        title="Comparación de los 3 Modelos vs Serie Real",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_white",
        height=900,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        hovermode='x unified',
        legend=dict(font=dict(size=28), x=0.02, y=0.98)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================================================
    # SECCIÓN 6: RESULTADO FINAL
    # ============================================================================
    st.header("6️⃣ Resultado Final - Proyecciones 2025-2026")
    
    # Seleccionar mejor modelo por %RMSE
    modelos = {
        'Modelo 2002': {'percentage_rmse': percentage_rmse_2002, 'forecast': forecast_2002, 'year': 2002},
        'Modelo 2010': {'percentage_rmse': percentage_rmse_2010, 'forecast': forecast_2010, 'year': 2010},
        'Modelo 2021': {'percentage_rmse': percentage_rmse_2021, 'forecast': forecast_2021, 'year': 2021}
    }
    
    mejor_modelo_nombre = min(modelos, key=lambda x: modelos[x]['percentage_rmse'])
    mejor_modelo_datos = modelos[mejor_modelo_nombre]
    mejor_forecast = mejor_modelo_datos['forecast']
    
    st.success(f"🏆 Mejor modelo seleccionado: {mejor_modelo_nombre} (Error relativo: {mejor_modelo_datos['percentage_rmse']:.2f}%)")
    
    # Calcular totales anuales
    total_2024_real = df_prophet[df_prophet['ds'].dt.year == 2024]['y'].sum()
    total_2025_mejor = calculate_yearly_total(mejor_forecast, 2025)
    total_2026_mejor = calculate_yearly_total(mejor_forecast, 2026)
    
    # Tabla de proyecciones - con formato
    proyecciones = pd.DataFrame({
        'Año': ['2024', '2025', '2026'],
        'Total': [f"{total_2024_real:,.1f}", f"{total_2025_mejor:,.1f}", f"{total_2026_mejor:,.1f}"],
        'Tipo': ['Real', 'Estimado', 'Estimado'],
        'Variación vs 2024': ['—', f"{((total_2025_mejor/total_2024_real - 1) * 100):+.1f}%", f"{((total_2026_mejor/total_2024_real - 1) * 100):+.1f}%"]
    })
    
    st.subheader("Totales Anuales")
    proyecciones_html = proyecciones.to_html(index=False, escape=False)
    st.markdown(f"""
    <div style="font-size: 32px; padding: 30px; line-height: 2.5;">
    {proyecciones_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico de proyecciones
    fig_proyecciones = go.Figure()
    
    años = ['2024', '2025', '2026']
    totales = [total_2024_real, total_2025_mejor, total_2026_mejor]
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig_proyecciones.add_trace(go.Bar(
        x=años,
        y=totales,
        marker=dict(color=colores, line=dict(color='black', width=3)),
        text=[f"<b>{v:,.1f}</b>" for v in totales],
        textposition='outside',
        textfont=dict(size=32)
    ))
    
    fig_proyecciones.update_layout(
        title="Totales Anuales: 2024-2026",
        xaxis_title="Año",
        yaxis_title="Total",
        template="plotly_white",
        height=800,
        font=dict(size=32),
        title_font_size=44,
        xaxis=dict(tickfont=dict(size=36), title=dict(font=dict(size=36))),
        yaxis=dict(tickfont=dict(size=32), title=dict(font=dict(size=36))),
        showlegend=False
    )
    st.plotly_chart(fig_proyecciones, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Nota:** El Modelo 2021 fue seleccionado por tener el menor error relativo (%RMSE), lo que indica mejor precisión en la predicción.")

else:
    st.info("👈 Carga tu archivo CSV o Excel en la barra lateral para comenzar el análisis")
