#& "C:\Users\DiegoR\anaconda3\envs\modulo3\python.exe" -m streamlit run "C:\RD\2026_2027\M3V2\1ACT\M3ACT12\chatbotcopy.py" 
import streamlit as st
import pandas as pd
from openai import OpenAI

# Configuración básica de la página
st.set_page_config(page_title="Chatbot de Análisis de Datos")
st.title("Chatbot con Streamlit y OpenAI")
st.write("Sube un archivo CSV y haz preguntas especificas sobre tus datos.")

# Barra lateral para ingresar la API Key de OpenAI de forma segura
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Ingresa tu OpenAI API Key", type="password")
    st.markdown("Obtén tu clave en [OpenAI Platform](https://openai.com).")

# 3. Entrada de datos: Carga de archivo CSV
uploaded_file = st.file_uploader("Sube tu tabla de datos en formato *.CSV", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa de los 10 primeros registros")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")

# 3. Entrada de datos: Campo de texto para el prompt del usuario
user_prompt = st.text_area(
    "Escribe tu pregunta especifica sobre los datos ingresados en el archivo CSV:",
    placeholder="Ejemplo: ¿Cuál es el promedio del campo ...?"
)

# Botón para enviar la consulta
if st.button("Consultar"):
    if not api_key:
        st.warning("Por favor, ingresa tu OpenAI API Key en la barra lateral.")
    elif df is None:
        st.warning("Por favor, sube primero un archivo CSV para poder analizarlo.")
    elif not user_prompt.strip():
        st.warning("Por favor, escribe una pregunta antes de consultar.")
    else:
        with st.spinner("Analizando los datos y generando respuesta..."):
            try:
                # Inicializar cliente de OpenAI
                client = OpenAI(api_key=api_key)

                # Convertir el DataFrame a texto (CSV/Markdown) para pasarlo como contexto
                # Nota: Para datasets muy grandes, se recomienda enviar un resumen o las primeras filas
                datos_texto = df.to_csv(index=False)

                # 4 y 6.2. Prompt del sistema estricto para evitar respuestas fuera del dataset
                system_instruction = (
                    "Eres un asistente de análisis de datos estricto. Tu única fuente de conocimiento "
                    "es la tabla de datos proporcionada a continuación.\n"
                    "REGLAS OBLIGATORIAS:\n"
                    "1. Responde ÚNICAMENTE utilizando la información presente en los datos proporcionados.\n"
                    "2. Si la pregunta del usuario no puede responderse con los datos de la tabla, "
                    "o trata sobre temas externos (cultura general, noticias, opiniones, código, etc.), "
                    "DEBE responder exactamente: 'Lo siento, esa pregunta está fuera del alcance de los datos proporcionados.'\n"
                    "3. No inventes ni asumas datos que no estén explícitamente en la tabla.\n\n"
                    f"--- INICIO DE DATOS (CSV) ---\n{datos_texto}\n--- FIN DE DATOS ---"
                )

                # Llamada al modelo gpt-4o-mini
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0  # Temperatura 0 para máxima precisión y cero creatividad
                )

                # 5. Salida del chatbot en la misma página
                st.subheader("Respuesta del Chatbot:")
                st.info(response.choices[0].message.content)

            except Exception as e:  
                st.error(f"Ocurrió un error al comunicarse con el modelo: {e}")
