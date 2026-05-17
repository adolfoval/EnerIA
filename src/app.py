import streamlit as st
import plotly.express as px
from core import load_data, transform_data, logo_path, ejecutar_simulacion_horizonte
 
#----------------------------------------------------------------------------------------
# Interfaz
#----------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="EnerIA - Planificación de Obras UPME")
st.title("Proyecto:  🌟 EnerIA", text_alignment="center")

col1, col2, col3 = st.columns(3)
with col2:
    st.image(logo_path,width=400)
"""--------------------------------------------------------------------------------------"""

data = load_data()
st.write(f"Dimensiones de los datos: {data.shape[0]} filas y {data.shape[1]} columnas.")
st.dataframe(data.head(5))

st.write("Datos procesados, limpios y transformados.")
data_ann = transform_data(data)
st.write(data_ann.head(5))

# columns = load_model_columns()
# if not columns:
#     st.toast(f"No se encontró el archivo de columnas", icon="🚨")

st.markdown("### 📊 Simulación y Análisis de Ciclo de Vida del Proyecto (ETAPA 5)")

# ==============================================================================
# CONTROLES DE ENTRADA (Ajuste del horizonte de tiempo)
# ==============================================================================
st.subheader("🎛️ Parámetros de la Infraestructura Propuesta")
col_inputs_1, col_inputs_2 = st.columns(2)

with col_inputs_1:
    ano_operacion = st.slider(
        "Año de Entrada en Operación de la Obra:", 
        min_value=2024, max_value=2050, value=2033, step=1
    )

with col_inputs_2:
    vida_util = st.slider(
        "Horizonte de Vida Útil Estimado (Años):", 
        min_value=5, max_value=50, value=25, step=1
    )

ano_limite = ano_operacion + vida_util
st.info(f"⏳ **Periodo bajo análisis:** Evaluación del sistema eléctrico desde `{ano_operacion}` hasta `{ano_limite}`.")

# ==============================================================================
# EJECUCIÓN Y RENDERIZADO DE RESULTADOS
# ==============================================================================
st.markdown("---")

try:
    # Llamamos a la lógica corregida y emparejada
    df_res, dictamen, justificacion, estilo_alerta, alivio_final = ejecutar_simulacion_horizonte(ano_operacion, vida_util)
    
    # 1. Bloque de Métricas Resumen (Estilo corporativo original)
    st.subheader(f"📈 Comportamiento Proyectado al Final de la Vida Útil ({ano_limite})")
    val_sin = df_res.iloc[-1]["Cargabilidad Sin Obra (%)"]
    val_con = df_res.iloc[-1]["Cargabilidad Con Obra (%)"]
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Cargabilidad Estimada (Sin Obra)", value=f"{val_sin:.2f} %")
    with m2:
        st.metric(label="Cargabilidad Estimada (Con Obra)", value=f"{val_con:.2f} %", delta=f"-{alivio_final:.2f} %", delta_color="inverse")
    with m3:
        st.metric(label="Alivio Sostenido en la Red", value=f"{alivio_final:.2f} %")

    # 2. Gráfico Interactivo de Curva de Carga (Plotly)
    st.subheader("📉 Evolución de la Cargabilidad en el Tiempo")
    
    df_melted = df_res.melt(
        id_vars=["Año"], 
        value_vars=["Cargabilidad Sin Obra (%)", "Cargabilidad Con Obra (%)"],
        var_name="Escenario de Simulación", 
        value_name="Cargabilidad (%)"
    )
    
    fig = px.line(
        df_melted, x="Año", y="Cargabilidad (%)", color="Escenario de Simulación",
        title=f"Tendencia del Sistema Eléctrico ({ano_operacion} - {ano_limite})",
        markers=True,
        color_discrete_map={"Cargabilidad Sin Obra (%)": "#EF553B", "Cargabilidad Con Obra (%)": "#636EFA"}
    )
    fig.add_hline(y=90.0, line_dash="dash", line_color="red", annotation_text="Límite Crítico Operativo (90%)", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Cuadro de Dictamen Técnico Automatizado
    st.subheader("📋 Dictamen Técnico de Viabilidad")
    if estilo_alerta == "success":
        st.success(f"### **✅ {dictamen}**\n\n**Justificación:** {justificacion}")
    elif estilo_alerta == "warning":
        st.warning(f"### **⚠️ {dictamen}**\n\n**Justificación:** {justificacion}")
    else:
        st.error(f"### **❌ {dictamen}**\n\n**Justificación:** {justificacion}")

except Exception as e:
    st.error(f"Error al procesar la simulación de la ETAPA 5: {e}")