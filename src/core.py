import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_path, "assets", "logo.jpg")
data_path = os.path.join(current_path, "datasets", "Base_de_datos.xlsx")
model_path = os.path.join(current_path, "models", "modelo_Ann_eneria.pkl")
columns_path = os.path.join(current_path, "models", "columnas_entrenamiento.pkl")

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

@st.cache_resource()
def load_model_columns():
    if os.path.exists(columns_path):
        # st.toast(f"Archivo de columnas encontrado", icon="✅")
        return joblib.load(columns_path)
    # st.toast(f"No se encontró el archivo de columnas", icon="🚨")
    raise FileNotFoundError(f"No se encontró el archivo de columnas en: {columns_path}")


@st.cache_resource()
def load_ml_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

def ejecutar_simulacion_horizonte(ano_operacion, vida_util):
    """
    Lógica de la ETAPA 5: Proyecta año a año el comportamiento del sistema 
    con y sin obra, calculando el alivio y el dictamen final.
    """
    model = load_ml_model()
    model_columns = load_model_columns()
    
    # Crear la lista de años correspondientes al horizonte de la vida útil
    anos_horizonte = list(range(ano_operacion, ano_operacion + vida_util + 1))
    
    resultados = []
    
    # Nota: Para hacer la predicción masiva, simulamos sobre la estructura base de columnas
    # Si en tu entrenamiento guardaste las columnas, creamos un DataFrame plantilla con ceros
    if model_columns is not None:
        base_row = pd.DataFrame(0, index=[0], columns=model_columns)
    else:
        # Fallback si no encuentra el archivo de columnas
        base_row = pd.DataFrame([{"Año": ano_operacion, "Alternativa": 0}])

    for anyo in anos_horizonte:
        # 1. Simulación ALTERNATIVA 0 (Sin Obra)
        df_a0 = base_row.copy()
        if 'Año' in df_a0.columns: df_a0['Año'] = anyo
        if 'Alternativa' in df_a0.columns: df_a0['Alternativa'] = 0
        if model_columns is not None: df_a0 = df_a0[model_columns]
        
        pred_a0 = model.predict(df_a0)
        cargabilidad_a0 = float(np.mean(pred_a0)) # Promedio del sistema simulado
        
        # 2. Simulación ALTERNATIVA 1 (Con Obra)
        df_a1 = base_row.copy()
        if 'Año' in df_a1.columns: df_a1['Año'] = anyo
        if 'Alternativa' in df_a1.columns: df_a1['Alternativa'] = 1
        if model_columns is not None: df_a1 = df_a1[model_columns]
        
        pred_a1 = model.predict(df_a1)
        cargabilidad_a1 = float(np.mean(pred_a1))
        
        # Guardar registros de este año específico
        resultados.append({
            "Año": anyo,
            "Cargabilidad Sin Obra (%)": round(cargabilidad_a0, 2),
            "Cargabilidad Con Obra (%)": round(cargabilidad_a1, 2),
            "Alivio Sostenido (%)": round(cargabilidad_a0 - cargabilidad_a1, 2)
        })
        
    df_resultados = pd.DataFrame(resultados)
    
    # Extraer datos del año final de vida útil para el dictamen técnico
    fila_final = df_resultados.iloc[-1]
    cargabilidad_final_con_obra = fila_final["Cargabilidad Con Obra (%)"]
    alivio_final = fila_final["Alivio Sostenido (%)"]
    ano_final = anyo
    
    # Lógica de Dictamen Automatizado según tu cuaderno (umbral crítico del 90%)
    if cargabilidad_final_con_obra <= 90.0:
        dictamen = "EXPANSIÓN TÉCNICAMENTE VIABLE Y ROBUSTA"
        justificacion = (f"La obra garantiza la confiabilidad del sistema durante sus {vida_util} años de operación. "
                         f"Para el año {ano_final}, previene de manera efectiva el colapso del sistema, "
                         f"manteniendo el nivel de carga por debajo del límite crítico del 90%.")
        estilo_alerta = "success"
    else:
        dictamen = "EXPANSIÓN CON RIESGO O RESTRICCIONES A LARGO PLAZO"
        justificacion = (f"Aunque la obra genera un alivio inicial, la proyección al final de su vida útil ({ano_final}) "
                         f"indica que el sistema volverá a superar el umbral crítico del 90% ({cargabilidad_final_con_obra}%). "
                         f"Se recomienda evaluar un rediseño de capacidad o infraestructura complementaria.")
        estilo_alerta = "warning"
        
    return df_resultados, dictamen, justificacion, estilo_alerta, alivio_final