# ============================================================
# CABECERA
# ============================================================
# Alumno: Nicolas Garcia Gonzalez
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.

SYSTEM_PROMPT = """
Eres un asistente analítico especializado exclusivamente en un dataset de Spotify ya cargado en memoria.

CONTEXTO DEL DATASET
- Fecha mínima disponible: {fecha_min}
- Fecha máxima disponible: {fecha_max}
- Plataformas válidas: {plataformas}
- Valores válidos de reason_start: {reason_start_values}
- Valores válidos de reason_end: {reason_end_values}
- Nombres válidos de columnas: {columnas}

OBJETIVO
Responder preguntas relacionadas únicamente con la información contenida en el dataset cargado.

Toda respuesta válida debe consistir en:
1. Un hallazgo corto basado en una agregación real de los datos.
2. Una especificación de visualización Plotly adecuada para responder la pregunta.

FORMATO DE SALIDA
Siempre devolver exclusivamente un JSON válido con la siguiente estructura:
Consulta válida:
{{
  "tipo": "grafico",
  "interpretacion": "hallazgo según restricciones de conocimiento que incluya la métrica, dimensión y filtro utilizados o valores de fecha, plataforma, reason_start, reason_end, artista o álbum si se aplicaron",
  "codigo": "código python que genera una figura plotly llamada fig"
}}

Consulta inválida:
{{
  "tipo": "fuera_de_alcance",
  "interpretacion": "explicación específica del error",
  "codigo": ""
}}

RESTRICCIONES SOBRE EL HALLAZGO
- Debe derivarse únicamente de datos existentes en el dataset.
- Debe estar basado en una agregación real.
- No puede superar 200 caracteres.
- Debe ocupar una sola línea.
- No incluir explicaciones adicionales.
- No inventar métricas, tendencias o conclusiones.
- No mencionar información que no pueda calcularse con los datos disponibles.

REGLAS DE VALIDACIÓN
Antes de construir cualquier respuesta:
1. Verificar que todos los filtros solicitados existan en el dataset.
2. Verificar que todos los valores solicitados existan en los dominios válidos.
3. Verificar que todas las fechas estén dentro del rango: {fecha_min} a {fecha_max}
4. Verificar que la fecha inicial sea menor o igual a la fecha final.
5. Verificar que no existan fechas negativas.
6. Verificar que no existan fechas futuras respecto al rango del dataset.
7. Verificar que las plataformas solicitadas pertenezcan a: {plataformas}
8. Verificar que los valores de reason_start pertenezcan a: {reason_start_values}
9. Verificar que los valores de reason_end pertenezcan a: {reason_end_values}
10. Verificar que los nombres de las columnas existan en {columnas} para definir las estructuras de las respuestas.
11. Verificar que la pregunta sea autosuficiente y no dependa de contexto previo.
12. Verificar que ningún filtro, dimensión o métrica solicitada sea inventada o no exista en el dataset.

MANEJO DE ERRORES
Si alguna validación falla, NO generar visualización ni agregaciones.
Responder únicamente con:
{{
  "status": "error",
  "reason": "explicación específica"
}}

Ejemplos válidos:
{{
  "status": "error",
  "reason": "La fecha 2035-01-01 está fuera del rango disponible del dataset ({fecha_min} - {fecha_max})."
}}
{{
  "status": "error",
  "reason": "La plataforma 'BlackBerry' no existe en el dataset. Valores válidos: {plataformas}."
}}
{{
  "status": "error",
  "reason": "El valor 'unknown_reason' no existe en reason_start."
}}
{{
  "status": "error",
  "reason": "La columna solicitada no existe en el dataset."
}}
{{
  "status": "error",
  "reason": "La fecha inicial es posterior a la fecha final."
}}

RESTRICCIONES DE CONOCIMIENTO
- Utilizar únicamente información presente en el dataset cargado.
- No utilizar conocimiento externo.
- No asumir valores faltantes.
- No inferir categorías inexistentes.
- No estimar resultados.
- No extrapolar tendencias.
- No utilizar ejemplos ficticios.
- No generar datos sintéticos.
- No utilizar respuestas preconstruidas.
- No inventar filtros.
- No inventar dimensiones.
- No inventar métricas.

ESTADO CONVERSACIONAL
Cada consulta debe procesarse de forma independiente.
- No mantener memoria de preguntas anteriores.
- No utilizar contexto conversacional previo.
- No recordar filtros utilizados anteriormente.
- No concatenar consultas previas.
- No inferir referencias como "igual que antes", "la misma fecha", "esa plataforma" o similares.
- Si la consulta depende de contexto previo, devolver error indicando que cada consulta debe ser autosuficiente.

VISUALIZACIÓN
Seleccionar automáticamente el tipo de gráfico más adecuado:
- Series temporales → line
- Comparaciones entre categorías → bar
- Distribuciones → histogram
- Participaciones → pie
- Relaciones entre variables → scatter
- Dispersión estadística → box

La visualización debe construirse exclusivamente con datos obtenidos del dataset y de la agregación solicitada.
Si la consulta no puede responderse con los datos existentes, devolver error.

NUNCA:
- Inventar resultados.
- Inventar registros.
- Inventar fechas.
- Inventar plataformas.
- Inventar valores de reason_start.
- Inventar valores de reason_end.
- Inventar agregaciones.
- Utilizar conocimiento externo.
- Utilizar memoria conversacional.
- Implementar reintentos automáticos.
- Corregir silenciosamente errores del usuario.
- Sustituir valores inválidos por otros similares.

REGLAS PARA EL CÓDIGO
El código generado debe:
- utilizar exclusivamente:
  df
  pd
  px
  go

- crear obligatoriamente una variable llamada:
  fig

- no utilizar print()
- no utilizar st.*
- no utilizar archivos externos
- no utilizar imports
- no utilizar exec()
- no utilizar variables no definidas

Cuando exista un error, explicar exactamente qué validación falló y por qué.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.

@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # PREPARACIÓN DE DATOS
    # ----------------------------------------------------------
    # Creación de columnas de fecha y tiempo
    df['ts'] = pd.to_datetime(df['ts'])
    df['month'] = df['ts'].dt.to_period('M').dt.to_timestamp()
    df['week'] = df['ts'].dt.isocalendar().week
    df['day'] = df['ts'].dt.day
    df['day_name'] = df['ts'].dt.day_name()
    df['hour'] = df['ts'].dt.hour
    df['quarter'] = df['ts'].dt.quarter
    df['semester'] = df['ts'].dt.month.apply(lambda x: 1 if x <= 6 else 2)
    df['shuffle'] = df['shuffle'].fillna(0).astype(int)
    df['skipped'] = df['skipped'].fillna(0).astype(int)
    df['secs_played'] = df['ms_played']/1000
    df['mins_played'] = df['secs_played']/60

    # Limpieza de nombres de columnas
    df.columns = df.columns.str.replace('master_metadata_', '').str.replace('album_a', 'a')
    df.rename(columns={'ts': 'timestamp'}, inplace=True)

    # Limpieza de duplicados y valores nulos
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['track_name'], inplace=True)

    # Estandarización de nombres de artistas, canciones y álbumes
    cols = ['artist_name', 'track_name', 'album_name']
    for col in cols:
        df[col] = df[col].str.upper().str.strip()

    # Cálculo de temporadas del año
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }

    df['season'] = df['month'].dt.month.map(season_map)
    
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores calculados aquí reemplazan a los placeholders
    dentro de SYSTEM_PROMPT.
    """
    fecha_min = df["timestamp"].min()
    fecha_max = df["timestamp"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
    columnas = df.columns.tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        columnas=columnas
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.

def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.

def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)
                st.expander("Respuesta LLM").write(raw)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")
                st.error(str(e))
                st.exception(e)


# ============================================================
# REFLEXIÓN TÉCNICA (Extensión: 28 líneas de texto sin contar espacios entre párrafos al abrirlo en Anaconda Jupyter)
# ============================================================
# La aplicación consta de 3 secciones principales: 1. Definición del entorno y variables de ejecución, 2. Definición de funciones y 3. Interfaz de usuario. 

# En la Sección 1 se cargan las librerias necesarias para el procesamiento, visualización de datos y conexiones al LLM, así como las instrucciones, limitaciones y formatos de respuesta que debe seguir el modelo de lenguaje para responder a las solicitudes.

# En esta primera sección, específicamente en la definición del system prompt, es donde radica la esencia de la aplicación. Dicho prompt contiene tanto las indicaciones generales de trabajo como los elementos de validación y restricciones que permitirán que las respuestas se ejecuten correctamente. Una de las restricciones más relevantes acá es la limitación de los nombres de columnas del dataset, puesto que el modelo podría inferir nombres basados en el contexto de la pregunta sin que fueran nombres reales en el dataset, desembocando en errores de ejecución dado que Python tendría conflictos para encontrar dentro de sus indices un campo que el modelo trató de rellenar para completar la solicitud. Si dicha restricción se eliminara, ante preguntas como ¿Cuál es mi album preferido? la respuesta del LLM podría hacer referencia a campos inexistentes como df['album_title'] o df['artist_album'], mientras que mantener la restricción permite que el LLM responda correctamente refiriendose al campo df['album_name'], el cual es un campo válido en el dataset.

#La Sección 2 corresponde a funciones reutilizables dentro del código, encargadas de dar formato y enriquecer los datos (load_data), modificar el system prompt con las variables y restricciones provenientes del dataset cargado (build_prompt), envío de system prompt, solicitudes y preguntas al LLM (get_response), tratamiento de la respuesta de texto del LLM para extraer la porción de código propuesta por el LLM (parse_response) y la ejecución local del código Python extraído de la respuesta, con la cual se crea la gráfica que responde a la pregunta realizada por el usuario (execute_chart). 

#Posteriormente, la Sección 3 consiste en la definición de la interfaz de usuario de la página y la incorporación de los elementos obtenidos de la Sección 2 para responder dinámicamente a las preguntas utilizando gráficos.

#En esencia, cada vez que el usuario hace una nueva consulta, la aplicación envia el system prompt, la consulta del usuario y el esquema del dataset (nombres de columnas y tipos de datos/ejemplos) al LLM, le solicita que cree una porción de código que permita responder con un gráfico a la pregunta del usuario, luego extrae ese código y lo ejecuta localmente para construir la gráfica dentro de Python para mostrarla finalmente en la página web. Todas las solicitudes al LLM consisten en utilizar ciertos elementos relevantes de los datos y el sytem prompt como restricciones de las respuestas para que se pueda generar código dinámicamente, de tal modo que la pregunta del usuario se pueda responder de manera consistente con el formato y contenido del dataset mientras que la ejecución del código se mantiene en el entorno local de ejecución de la app.

# Las razones por las cuales esta arquitectura no entrega directamente los datos al LLM se pueden resumir en 2 aspectos principales: optimización de recursos y privacidad. La optimización de recursos se refiere al consumo que se requiere por parte del modelo; dado que cada solicitud se ejecuta independientemente y sin mantener memoria de conversación, el cargar los datos al modelo implicaria su lectura completa cada vez que el usuario utiliza la página, lo cual llevaría a un consumo excesivo de tokens y tiempo de procesamiento. En cuanto a la privacidad, esta arquitectura es una opción que permite explotar las capacidades de codificación de los modelos y su nivel de adaptabilidad y razonamiento para generar dinámicamente las respuestas sin necesidad de depender de líneas de código escritas en el archivo (hard coded), mientras se mantienen datos sensibles fuera del alcance de posibles filtraciones o almacenamiento en los servidores del modelo.