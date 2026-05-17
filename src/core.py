import streamlit as st
import pandas as pd
import os
from tensorflow.keras.models import load_model
import numpy as np
import joblib

current_path = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_path, "assets", "logo.jpg")
data_path = os.path.join(current_path, "datasets", "Base_de_datos.xlsx")
# model_path = os.path.join(current_path, "models", "modelo_eneria_original.keras")
model_path = os.path.join(current_path, "models", "modelo_original_pipeline_eneria.keras")
# columns_path = os.path.join(current_path, "models", "columnas_entrenamiento.pkl")
# scaler_path = os.path.join(current_path, "models", "scaler_original.pkl")
x_base_path = os.path.join(current_path, "models", "X_original.pkl")

@st.cache_data(persist="disk")
def load_data():
    print("¡SI ESTE MENSAJE APARECE MAS DE UNA VEZ EN CONSOLA, NO ESTA CACHEANDO LA CARGA!")
    df = pd.read_excel(data_path)
    return df

@st.cache_data(persist="disk")
def transform_data(df):
    print("¡SI ESTE MENSAJE APARECE EN LA CONSOLA MAS DE UNA VEZ, NO ESTA CACHEANDO LA TRANSFORMACION!")
    # ==============================================================================
    # ETAPA 2: PROCESAMIENTO Y LIMPIEZA DE DATOS
    # ==============================================================================
    # 1. Variables de identificación (las que no cambian entre alternativas)
    id_vars = ['Escenario generacion', 'Escenario demanda', 'Año', 'Contingencia', 'Elemento']

    # 2. Transformar las columnas de cargabilidad A0 y A1 en una sola columna objetivo (Target)
    df_ml = pd.melt(df, id_vars=id_vars,
                    value_vars=['Cargabilidad (%)_A0', 'Cargabilidad (%)_A1'],
                    var_name='Alternativa',
                    value_name='Cargabilidad_Porcentaje')

    # 3. Convertir la columna de 'Alternativa' a formato binario: 0 (sin obra) y 1 (con obra)
    df_ml['Alternativa'] = df_ml['Alternativa'].map({'Cargabilidad (%)_A0': 0, 'Cargabilidad (%)_A1': 1})

    # 4. Eliminar valores nulos por seguridad (buenas prácticas en ciencia de datos)
    df_ml = df_ml.dropna()

    # 5. Convertir las variables de texto a numéricas (One-Hot Encoding) para la IA
    columnas_categoricas = ['Escenario generacion', 'Escenario demanda', 'Contingencia', 'Elemento']
    df_ml = pd.get_dummies(df_ml, columns=columnas_categoricas, drop_first=True)

    return df_ml

@st.cache_resource
def _load_ann_model():
    if os.path.exists(model_path):
            return load_model(model_path)
 
    raise FileNotFoundError("No se encontró el archivo del modelo.")

# @st.cache_resource
# def _load_model_columns():
#     if os.path.exists(columns_path):
#         return list(joblib.load(columns_path))
#     return []

@st.cache_resource
def _load_ann_components():
    """Carga de forma segura la matriz base"""
    # scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    X_base = joblib.load(x_base_path) if os.path.exists(x_base_path) else None
    return  X_base

def ejecutar_simulacion_horizonte(ano_operacion, vida_util):
    """
    Simulación optimizada con Pipeline de Keras unificado.
    Forza la conversión de datos a float32 para evitar el error de dtypes.
    """
    model = _load_ann_model()
    X_base = _load_ann_components()
    
    anos_horizonte = list(range(ano_operacion, ano_operacion + vida_util + 1))
    resultados = []
    
    for anyo in anos_horizonte:
        # --- ALTERNATIVA 0 (Sin Obra) ---
        df_sim_alt0 = X_base.copy()
        df_sim_alt0['Año'] = float(anyo)
        df_sim_alt0['Alternativa'] = 0.0
        
        # SOLUCIÓN CRÍTICA: Convertimos todo el bloque a float32 antes de pasarlo a Keras
        X_input_alt0 = df_sim_alt0.astype(np.float32).values
        pred_alt0 = model.predict(X_input_alt0, verbose=0).mean()
        
        # --- ALTERNATIVA 1 (Con Obra) ---
        df_sim_alt1 = X_base.copy()
        df_sim_alt1['Año'] = float(anyo)
        df_sim_alt1['Alternativa'] = 1.0
        
        # SOLUCIÓN CRÍTICA: Convertimos todo el bloque a float32 antes de pasarlo a Keras
        X_input_alt1 = df_sim_alt1.astype(np.float32).values
        pred_alt1 = model.predict(X_input_alt1, verbose=0).mean()
        
        resultados.append({
            "Año": anyo,
            "Cargabilidad Sin Obra (%)": round(float(pred_alt0), 2),
            "Cargabilidad Con Obra (%)": round(float(pred_alt1), 2),
            "Alivio Sostenido (%)": round(float(pred_alt0 - pred_alt1), 2)
        })
        
    df_resultados = pd.DataFrame(resultados)
    
    # --- COMPLETAMOS LA LÓGICA DE DICTÁMENES PARA EL RETORNO ---
    fila_final = df_resultados.iloc[-1]
    c0_final = fila_final["Cargabilidad Sin Obra (%)"]
    c1_final = fila_final["Cargabilidad Con Obra (%)"]
    alivio_final = fila_final["Alivio Sostenido (%)"]
    
    if alivio_final > 1.0:
        if c1_final < 90 and c0_final >= 90:
            dictamen = "EXPANSIÓN TÉCNICAMENTE VIABLE Y ROBUSTA"
            justificacion = f"La obra garantiza la confiabilidad del sistema durante sus {vida_util} años de operación. Previene de manera efectiva el colapso del sistema (superación del límite crítico del 90%)."
            estilo_alerta = "success"
        else:
            dictamen = "EXPANSIÓN TÉCNICAMENTE VIABLE (MEJORA OPERATIVA SOSTENIDA)"
            justificacion = f"Aporta un alivio del {alivio_final:.2f}%, asegurando holgura operativa y respaldando el criterio de contingencia N-1 a largo plazo."
            estilo_alerta = "success"
    elif alivio_final <= 1.0 and alivio_final > 0:
        dictamen = "VIABILIDAD MARGINAL AL FINAL DEL CICLO - REVISIÓN REQUERIDA"
        justificacion = f"El impacto de la obra se diluye casi por completo al final del horizonte (alivio de solo {alivio_final:.2f}%). Se sugiere revisar dimensiones de ingeniería."
        estilo_alerta = "warning"
    else:
        dictamen = "OBSOLESCENCIA PREMATURA - PROYECTO NO VIABLE"
        justificacion = f"Para el año evaluado, la obra ya no aporta beneficios de cargabilidad frente a la alternativa de no hacer nada."
        estilo_alerta = "error"
        
    return df_resultados, dictamen, justificacion, estilo_alerta, alivio_final

@st.cache_resource
def load_validation_data():
    """Carga los datos históricos de validación y métricas guardados desde Colab"""
    valid_path = os.path.join(current_path, "models", "datos_validacion.pkl")
    if os.path.exists(valid_path):
        return joblib.load(valid_path)
    return None