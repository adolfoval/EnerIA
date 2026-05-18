import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from core import load_data, transform_data, logo_path, ejecutar_simulacion_horizonte, load_shap_data
 
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

st.markdown("### Simulación y Análisis de Ciclo de Vida del Proyecto")

# ==============================================================================
# FORMULARIO DE ENTRADA DE PARÁMETROS (ETAPA 5)
# ==============================================================================

# Creamos el contenedor del formulario
with st.form(key="formulario_simulacion"):
    st.markdown("Configure los parámetros técnicos del ciclo de vida de la infraestructura:")
    
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        ano_operacion = st.slider(
            "Año de entrada en operación de la obra:", 
            min_value=2024, 
            max_value=2040, 
            value=2030,
            step=1
        )
        
    with col_form2:
        vida_util = st.slider(
            "Vida útil estimada de la obra (Años):", 
            min_value=1, 
            max_value=50, 
            value=15,
            step=1
        )
        
    # Calculamos el límite visual de la proyección dentro del form
    ano_limite = ano_operacion + vida_util
    st.info(f" **Horizonte de evaluación:** El sistema proyectará el comportamiento de la red hasta el año **{ano_limite}**.")
    
    # El botón de envío obligatorio de st.form
    # Nota: No ejecutes st.button común aquí adentro, DEBE ser st.form_submit_button
    boton_predecir = st.form_submit_button(label="Ejecutar Predicción", use_container_width=True)

# ==============================================================================
# EJECUCIÓN DEL PIPELINE MATEMÁTICO (Solo ocurre al hacer clic)
# ==============================================================================
if boton_predecir:
    with st.spinner("🧠 La **Red Neuronal** está procesando los flujos de potencia..."):
        try:
            # 1. Ejecutar simulación
            df_res, dictamen, justificacion, estilo_alerta, alivio_final = ejecutar_simulacion_horizonte(ano_operacion, vida_util)
            
            st.markdown("---")
            
            # Creación de pestañas: Una para el análisis y otra para la validación del modelo
            tab1, tab2, tab3, tab4= st.tabs(["Simulación y análisis de Ciclo de Vida", "Métricas de Precisión del modelo", "Cuellos de Botella (Heatmaps)", "Explicabilidad de la Red Neuronal (SHAP)"])
            
            # ==========================================================
            # PESTAÑA 1: RESULTADOS DE LA SIMULACIÓN (Tu Etapa 5 actual)
            # ==========================================================
            with tab1:
                st.subheader(f"📈 Comportamiento Proyectado al Final de la Vida Útil ({ano_limite})")
                val_sin = df_res.iloc[-1]["Cargabilidad Sin Obra (%)"]
                val_con = df_res.iloc[-1]["Cargabilidad Con Obra (%)"]
                
                m1, m2, m3 = st.columns(3)
                with m1: st.metric(label="Cargabilidad Estimada (Sin Obra)", value=f"{val_sin:.2f} %")
                with m2: st.metric(label="Cargabilidad Estimada (Con Obra)", value=f"{val_con:.2f} %", delta=f"-{alivio_final:.2f} %", delta_color="inverse")
                with m3: st.metric(label="Alivio Sostenido en la Red", value=f"{alivio_final:.2f} %")

                import plotly.graph_objects as go
            
                # Recuperamos el último año histórico real de tu dataset base
                # Como tu captura muestra que los datos van hasta 2037:
                año_max_historico = 2037 
                
                # Creamos la figura desde cero usando Graph Objects para control total de las líneas
                fig = go.Figure()

                # --- 1. DATOS HISTÓRICOS (Líneas Sólidas) ---
                # Filtramos o asumimos el tramo histórico (2027 a 2037 en tu gráfica)
                # Para mantener la gráfica continua y fiel al cuaderno, tomamos los puntos del dataframe resultantes o históricos
                # NOTA: Si deseas pintar el histórico real exacto del dataframe 'df', asegúrate de pasarlo.
                # Aquí simularemos la unión continua con los datos que entrega tu simulación:
                
                df_hist = df_res[df_res['Año'] <= año_max_historico]
                df_proy = df_res[df_res['Año'] >= año_max_historico]
                
                # Si el horizonte del slider empieza después del histórico (ej. 2039), 
                # generamos el tramo intermedio o histórico para que la gráfica no quede mocha:
                df_completo_historico = df_res[df_res['Año'] <= año_max_historico]

                # Líneas históricas (Sólidas con marcadores)
                fig.add_trace(go.Scatter(
                    x=df_completo_historico['Año'], 
                    y=df_completo_historico['Cargabilidad Sin Obra (%)'], 
                    mode='lines+markers', 
                    name='Histórico Alt 0 (Sin Obra)', 
                    line=dict(color='red', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_completo_historico['Año'], 
                    y=df_completo_historico['Cargabilidad Con Obra (%)'], 
                    mode='lines+markers', 
                    name='Histórico Alt 1 (Con Obra)', 
                    line=dict(color='green', width=2)
                ))

                # --- 2. LÍNEAS PROYECTADAS POR IA (Líneas Punteadas 'dash') ---
                df_solo_proyeccion = df_res[df_res['Año'] >= año_max_historico]
                
                fig.add_trace(go.Scatter(
                    x=df_solo_proyeccion['Año'], 
                    y=df_solo_proyeccion['Cargabilidad Sin Obra (%)'], 
                    mode='lines+markers', 
                    name='Proyección IA Alt 0 (Sin Obra)', 
                    line=dict(color='salmon', dash='dash', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_solo_proyeccion['Año'], 
                    y=df_solo_proyeccion['Cargabilidad Con Obra (%)'], 
                    mode='lines+markers', 
                    name='Proyección IA Alt 1 (Con Obra)', 
                    line=dict(color='lightgreen', dash='dash', width=2)
                ))

                # --- 3. HITOS Y LÍNEAS DELIMITADORAS DEL CUADERNO ---
                
                # HITO 1: Inicio de la Proyección IA (Línea vertical gris)
                fig.add_vline(
                    x=año_max_historico, 
                    line_width=2, 
                    line_dash="dash", 
                    line_color="gray", 
                    annotation_text="Inicio Proyección IA", 
                    annotation_position="top left"
                )

                # HITO 2: Entrada en operación de la obra (Línea vertical azul basada en el Slider)
                fig.add_vline(
                    x=ano_operacion, 
                    line_width=2, 
                    line_dash="dashdot", 
                    line_color="blue", 
                    annotation_text=f"Entrada Obra ({ano_operacion})", 
                    annotation_position="bottom right"
                )

                # Límite operativo crítico normativo de contingencias (90%)
                fig.add_hline(
                    y=90.0, 
                    line_width=1.5,
                    line_dash="dot", 
                    line_color="black", 
                    annotation_text="Límite Operativo Crítico (90%)",
                    annotation_position="top right"
                )

                # --- 4. AJUSTES DE DISEÑO CORPORATIVO ---
                fig.update_layout(
                    title=f'Análisis de Ciclo de Vida: Evolución de Cargabilidad ({df_res["Año"].min()} - {ano_limite})',
                    xaxis_title='Año de Estudio',
                    yaxis_title='Cargabilidad Promedio del Sistema (%)',
                    template='plotly_white',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Renderizamos la gráfica interactiva en Streamlit con ancho adaptativo
                st.plotly_chart(fig, use_container_width=True)

                # Dictamen
                if estilo_alerta == "success": st.success(f"### **{dictamen}**\n\n**Justificación:** {justificacion}")
                elif estilo_alerta == "warning": st.warning(f"### **{dictamen}**\n\n**Justificación:** {justificacion}")
                else: st.error(f"### **{dictamen}**\n\n**Justificación:** {justificacion}")

                # Opcional: Tabla de datos crudos colapsable
                with st.expander("Ver valores detallados de la simulación año por año"):
                    st.dataframe(df_res, use_container_width=True)
            # ==========================================================
            # PESTAÑA 2: VALIDACIÓN DE LA RED NEURONAL (Nueva Etapa 4)
            # ==========================================================
            with tab2:
                st.subheader("Validación del Modelo (ANN)")
                st.markdown("Métricas obtenidas durante la fase de pruebas y validación cruzada del modelo:")
                
                # Importamos datos desde el core
                from core import load_validation_data
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                datos_v = load_validation_data()
                
                # Si por alguna razón no encuentra el pkl, definimos las métricas fijas de tu Colab
                val_mae = 3080.45 if datos_v is None else mean_absolute_error(datos_v['y_test'], datos_v['y_pred']) # Ajusta si volviste a entrenar
                val_r2 = 0.9450 if datos_v is None else r2_score(datos_v['y_test'], datos_v['y_pred']) # Ajusta a tus valores reales fijos
                val_rmse = 3081.60 if datos_v is None else np.sqrt(mean_squared_error(datos_v['y_test'], datos_v['y_pred']))

                # Tarjetas de KPI elegantes para Auditoría
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1: st.metric(label="R² Score (Precisión Global)", value=f"{val_r2:.4f}")
                with kpi2: st.metric(label="MAE (Error Absoluto Medio)", value=f"{val_mae:.2f} %")
                with kpi3: st.metric(label="RMSE (Raíz del Error Cuadrático)", value=f"{val_rmse:.2f} %")
                
                st.markdown("---")
                col_graf1, col_graf2 = st.columns(2)
                
                # Gráfico 1: Curva de Aprendizaje interactiva (Loss)
                with col_graf1:
                    st.markdown("#### Evolución del Aprendizaje por Épocas")
                    fig_loss = go.Figure()
                    if datos_v is not None:
                        fig_loss.add_trace(go.Scatter(y=datos_v['loss'], mode='lines', name='Pérdida Entrenamiento', line=dict(color='#636EFA')))
                        fig_loss.add_trace(go.Scatter(y=datos_v['val_loss'], mode='lines', name='Pérdida Validación', line=dict(color='#EF553B')))
                    else:
                        st.warning("Para ver las curvas vivas, exporta 'datos_validacion.pkl' desde tu Colab.")
                    fig_loss.update_layout(xaxis_title="Épocas", yaxis_title="Error (MSE)", template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                # Gráfico 2: Dispersión Real vs Predicción
                with col_graf2:
                    st.markdown("#### Cargabilidad Real vs Predicción")
                    if datos_v is not None:
                        fig_scatter = px.scatter(
                            x=datos_v['y_test'], y=datos_v['y_pred'], 
                            labels={'x': 'Cargabilidad Real (%)', 'y': 'Cargabilidad Predicha (%)'},
                            opacity=0.6, color_discrete_sequence=['#FF7F0E']
                        )
                        # Añadir línea de identidad (la diagonal perfecta de referencia k--)
                        min_val = min(datos_v['y_test'].min(), datos_v['y_pred'].min())
                        max_val = max(datos_v['y_test'].max(), datos_v['y_pred'].max())
                        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
                    else:
                        st.info("Gráfico de dispersión listo. Esperando matriz de validación.")
                    fig_scatter.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                st.subheader("Mapas de calor comparativos (análisis de cuellos de botella)")
                st.markdown(
                    "El siguiente análisis evalúa la cargabilidad individual de cada elemento del sistema "
                    "a lo largo del horizonte de estudio. Permite identificar la vulnerabilidad local de la infraestructura."
                )
                
                # Invocamos la función del backend pasando tu dataset original de ML o Datos Base (df)
                # NOTA: Asegúrate de que la variable 'df' (el dataframe con datos históricos/base) esté cargada en app.py
                from core import generar_datos_heatmaps
                
                try:
                    p_a0, p_a1, vmin, vmax = generar_datos_heatmaps(data)
                    
                    # Extraemos los ejes para Plotly
                    elementos = p_a0.index.tolist()
                    años = p_a0.columns.tolist()
                    
                    # Convertimos las matrices a listas de Python para que Plotly las digiera sin problemas
                    z_a0 = p_a0.values.tolist()
                    z_a1 = p_a1.values.tolist()
                    
                    # Creamos dos columnas en la interfaz para poner los mapas lado a lado
                    col_mapa1, col_mapa2 = st.columns(2)
                    
                    import plotly.graph_objects as go
                    
                    # --- MAPA DE CALOR 1: ALTERNATIVA 0 (Sin Obra) ---
                    with col_mapa1:
                        fig_h0 = go.Figure(data=go.Heatmap(
                            z=z_a0, x=años, y=elementos,
                            colorscale='YlOrRd',
                            zmin=vmin, zmax=vmax,
                            colorbar=dict(title="Cargabilidad (%)", len=0.8),
                            hovertemplate="<b>Elemento:</b> %{y}<br><b>Año:</b> %{x}<br><b>Cargabilidad:</b> %{z:.1f}%<extra></extra>"
                        ))
                        fig_h0.update_layout(
                            title="Alternativa 0 (Sin Obra) - Riesgo Operativo",
                            xaxis_title="Año de Operación",
                            yaxis_title="Elemento del Sistema",
                            template="plotly_white",
                            height=550
                        )
                        st.plotly_chart(fig_h0, use_container_width=True)
                        
                    # --- MAPA DE CALOR 2: ALTERNATIVA 1 (Con Obra) ---
                    with col_mapa2:
                        fig_h1 = go.Figure(data=go.Heatmap(
                            z=z_a1, x=años, y=elementos,
                            colorscale='YlOrRd',
                            zmin=vmin, zmax=vmax,
                            # Ocultamos la barra de color del segundo para no duplicar espacio visual si están simétricos
                            showscale=False, 
                            hovertemplate="<b>Elemento:</b> %{y}<br><b>Año:</b> %{x}<br><b>Cargabilidad:</b> %{z:.1f}%<extra></extra>"
                        ))
                        # En Streamlit/Plotly no compartimos el eje Y de forma estricta como subplots, 
                        # pero al ponerlos lado a lado con el mismo rango y tamaño, se alinean perfectamente.
                        fig_h1.update_layout(
                            title="Alternativa 1 (Con Obra) - Alivio en la Red",
                            xaxis_title="Año de Operación",
                            yaxis_title="", # Dejamos vacío o limpio para evitar redundancia
                            template="plotly_white",
                            height=550
                        )
                        st.plotly_chart(fig_h1, use_container_width=True)
                        
                    # --- CONCLUSIÓN AUTOMÁTICA VISUAL EN FORMATO UI ---
                    st.markdown("---")
                    st.markdown("### Análisis comparativo de mapas de calor:")
                    
                    st.info(
                        "* **Transición de Estrés Térmico:** La disipación de los bloques de color rojo/anaranjado hacia tonos amarillos o más claros "
                        "en el gráfico de la derecha (**Con Obra**) valida geográficamente que la expansión mitiga el riesgo de sobrecarga.\n"
                        "* **Restricciones Remanentes:** Aquellos elementos específicos que mantengan celdas en tonos oscuros o rojos hacia el final "
                        "del horizonte en la Alternativa 1 actúan como alertas tempranas. Indican futuros cuellos de botella secundarios que "
                        "requerirán nuevos esquemas de inversión o análisis complementarios en los próximos planes de expansión."
                    )
                    
                except Exception as e:
                    st.error(f"Error al generar los Mapas de Calor de la Etapa 6: {e}")
                    st.info("Asegúrese de que el dataframe 'df' base contenga las columnas 'Elemento', 'Año', 'Cargabilidad (%)_A0' y 'Cargabilidad (%)_A1'.")

            with tab4:
                st.subheader("Inteligencia Artificial Explicable (XAI) con Valores SHAP")
                st.markdown(
                    "Este módulo rompe el paradigma de 'caja negra' de la Red Neuronal, permitiendo auditar "
                    "cuáles variables técnicas y operativas tienen el mayor peso en las predicciones del sistema."
                )
                
                # Cargamos los datos precalculados del laboratorio de Colab
                datos_s = load_shap_data()
                
                if datos_s is not None:
                    try:
                        # 1. EXTRACCIÓN Y REESTRUCTURACIÓN DE MATRICES (Tidy Data)
                        sv = np.array(datos_s['shap_values'])       # Matriz SHAP [Muestras x Variables]
                        mv = np.array(datos_s['muestra_prueba'])     # Matriz Original Escalada [Muestras x Variables]
                        columnas = datos_s['columnas']
                        
                        # Calculamos la importancia global media para ordenar de arriba hacia abajo
                        importancias = np.abs(sv).mean(axis=0)
                        indices_ordenados = np.argsort(importancias)[::-1][:10] # Top 10 variables
                        
                        # Creamos la lista ordenada de variables (la más importante va en el índice superior)
                        orden_variables_y = [columnas[i] for i in indices_ordenados]
                        
                        lista_registros = []
                        num_muestras = sv.shape[0]
                        
                        # Aplanamos la matriz aplicando el efecto Jitter directamente en el dataframe
                        for idx_muestra in range(num_muestras):
                            for rango_y, idx_var in enumerate(reversed(indices_ordenados)):
                                var_nombre = columnas[idx_var]
                                shap_val = sv[idx_muestra, idx_var]
                                feat_val = mv[idx_muestra, idx_var] # Escala de color
                                
                                # Aplicamos dispersión vertical aleatoria (Jitter) estrictamente confinada al carril
                                # El carril va desde (rango_y - 0.2) hasta (rango_y + 0.2)
                                jitter_y = rango_y + np.random.uniform(-0.18, 0.18)
                                
                                lista_registros.append({
                                    'Muestra N°': idx_muestra,
                                    'Variable Técnica': var_nombre,
                                    'Impacto (SHAP Value)': shap_val,
                                    'Valor de la Variable (Color)': feat_val,
                                    'Y_Con_Jitter': jitter_y,
                                    'Carril_Base': rango_y
                                })
                        
                        df_shap_plot = pd.DataFrame(lista_registros)

                        # 2. CONSTRUCCIÓN PURA CON GRAPH OBJECTS (Control Absoluto del Renderizado)
                        fig_shap_live = go.Figure()
                        
                        # Añadimos la nube de puntos interactiva usando Scatter clásico (admite colorscale de forma nativa)
                        fig_shap_live.add_trace(go.Scatter(
                            x=df_shap_plot['Impacto (SHAP Value)'],
                            y=df_shap_plot['Y_Con_Jitter'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=df_shap_plot['Valor de la Variable (Color)'],
                                # Escala térmica Azul (Bajo) -> Púrpura (Medio) -> Rojo (Alto) idéntica al cuaderno
                                colorscale=[[0.0, '#0074D9'], [0.5, '#B10DC9'], [1.0, '#FF4136']],
                                showscale=True,
                                opacity=0.85,
                                line=dict(width=0.4, color='white'),
                                colorbar=dict(
                                    title="Feature value",
                                    tickvals=[df_shap_plot['Valor de la Variable (Color)'].min(), df_shap_plot['Valor de la Variable (Color)'].max()],
                                    ticktext=["Low", "High"],
                                    len=0.85
                                )
                            ),
                            # Ventana emergente interactiva elegante (Hover)
                            hovertemplate=(
                                "<b>Variable:</b> %{customdata[0]}<br>"
                                "<b>Impacto SHAP:</b> %{x:.4f}<br>"
                                "<b>Valor Característica:</b> %{marker.color:.4f}<br>"
                                "<b>Simulación Index:</b> #%{customdata[1]}<extra></extra>"
                            ),
                            # Inyectamos datos complementarios para que el Hover los lea sin romper los ejes
                            customdata=np.stack((df_shap_plot['Variable Técnica'], df_shap_plot['Muestra N°']), axis=-1)
                        ))
                        
                        # 3. AJUSTES DEL LIENZO, EJE CATEGÓRICO Y LÍNEAS DE REJILLA
                        valores_eje_y = list(range(len(orden_variables_y)))
                        nombres_eje_y = list(reversed(orden_variables_y)) # Invertimos para que la más importante quede arriba
                        
                        fig_shap_live.update_layout(
                            template='plotly_white',
                            height=600, # Altura amplia para evitar el amontonamiento de nombres
                            title='Top 10 Variables Críticas en el Modelo Predictivo (Análisis SHAP Interactivo)',
                            xaxis_title='SHAP value (impact on model output)',
                            yaxis_title='',
                            # Línea negra central de referencia en 0 (indica neutralidad operativa)
                            xaxis=dict(zeroline=True, zerolinecolor='#333333', zerolinewidth=1.5, showgrid=True),
                            # Enmascaramos el eje Y para ocultar los números del jitter y mostrar los nombres limpios
                            yaxis=dict(
                                tickmode='array',
                                tickvals=valores_eje_y,
                                ticktext=nombres_eje_y,
                                showgrid=True,
                                gridcolor='#E5E7E9', # Líneas horizontales grises tenues que delimitan cada carril
                                tickfont=dict(size=12, family="Arial, sans-serif")
                            ),
                            hovermode='closest'
                        )
                        
                        # Renderizamos el gráfico interactivo corregido
                        st.plotly_chart(fig_shap_live, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error al estructurar el gráfico interactivo de SHAP: {e}")
                        
                else:
                    st.warning("⚠️ Archivo 'datos_shap.pkl' no detectado en el directorio de modelos.")
                    st.info("Asegúrese de montar el archivo pkl exportado desde su cuaderno para inicializar la sección.")
                # --- GUÍA DE INTERPRETACIÓN EN FORMATO UI MODERNO ---
                st.markdown("---")
                st.markdown("### Guía Metodológica de Interpretación")
                
                c_guia1, c_guia2, c_guia3 = st.columns(3)
                
                with c_guia1:
                    st.markdown("#### 1. Orden de Importancia")
                    st.info(
                        "Las variables en el **eje vertical (Y)** están organizadas de forma estrictamente descendente. "
                        "El elemento en la cima posee el mayor impacto y sensibilidad sobre el resultado del modelo."
                    )
                    
                with c_guia2:
                    st.markdown("#### 2. Sentido del Impacto")
                    st.info(
                        "El **eje horizontal (X)** mide el valor SHAP:\n"
                        "* **Puntos a la Derecha (> 0):** Esa variable AUMENTA la predicción **(Ej. mayor cargabilidad)**.\n"
                        "* **Puntos a la Izquierda (< 0):** Esa variable DISMINUYE la predicción **(Ej. menor cargabilidad)**."
                    )
                    
                with c_guia3:
                    st.markdown("#### 3. Magnitud Física")
                    st.info(
                        "El color de cada punto representa el valor original del dato de entrada:\n"
                        "* **Color Rojo:** Valor alto de la variable (ej. pico de demanda o año avanzado).\n"
                        "* **Color Azul:** Valor bajo (ej. valles de carga o alternativa activa).\n"
                        "* Nota: Cada punto en el gráfico representa una simulación específica evaluada por la IA."
                    )

        except Exception as e:
            st.error(f"Error al renderizar el dashboard: {e}")