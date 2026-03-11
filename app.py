import streamlit as st
import os
import logging
from eterna_core import (
    generar_respuesta,
    buscar_textos_similares,
    guardar_interaccion,
    planificar_y_ejecutar,
    reiniciar_indice_faiss,
    cargar_modelos_embedding,
    cargar_modelos_generacion,
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Cargar variables de entorno (para local) o usar st.secrets (para nube)
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Cargando variables desde .env")
else:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        if "EMAIL_SMTP_SERVER" in st.secrets:
            os.environ["EMAIL_SMTP_SERVER"] = st.secrets["EMAIL_SMTP_SERVER"]
            os.environ["EMAIL_SMTP_PORT"] = str(st.secrets["EMAIL_SMTP_PORT"])
            os.environ["EMAIL_REMITENTE"] = st.secrets["EMAIL_REMITENTE"]
            os.environ["EMAIL_PASSWORD"] = st.secrets["EMAIL_PASSWORD"]
        if "HA_URL" in st.secrets:
            os.environ["HA_URL"] = st.secrets["HA_URL"]
            os.environ["HA_TOKEN"] = st.secrets["HA_TOKEN"]
        logger.info("Variables cargadas desde secrets")
    except Exception as e:
        logger.error(f"Error cargando secrets: {e}")

# Inicializar modelos (solo una vez)
if 'modelos_cargados' not in st.session_state:
    with st.spinner("Cargando modelos..."):
        cargar_modelos_embedding()
        cargar_modelos_generacion()
        st.session_state['modelos_cargados'] = True
    logger.info("Modelos cargados en sesión")

st.set_page_config(page_title="ETERNA", layout="wide")
st.markdown("<style>body { background-color: #0e1117; color: #00ff00; }</style>", unsafe_allow_html=True)

# Diagnóstico
with st.expander("🔧 Diagnóstico de modelos", expanded=False):
    if st.button("Recargar modelos de embedding"):
        with st.spinner("Recargando..."):
            cargar_modelos_embedding()
        st.success("Modelos de embedding recargados")
    if st.button("Recargar modelos de generación"):
        with st.spinner("Recargando..."):
            cargar_modelos_generacion()
        st.success("Modelos de generación recargados")

# Mantenimiento
with st.expander("🛠️ Mantenimiento de memoria", expanded=False):
    if st.button("Reiniciar índice FAISS (borrar todos los vectores)"):
        resultado = reiniciar_indice_faiss()
        st.success(resultado)
        st.rerun()

# Inicializar historial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Háblame, Didier..."):
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    guardar_interaccion("user", prompt)

    with st.spinner("Buscando en mi memoria..."):
        contexto = buscar_textos_similares(prompt, k=3)

    palabras_clave = ["prepara", "organiza", "planifica", "secuencia", "automatiza", "haz todo lo necesario"]
    if any(palabra in prompt.lower() for palabra in palabras_clave):
        with st.spinner("ETERNA está planificando..."):
            respuesta = planificar_y_ejecutar(prompt)
    else:
        with st.spinner("ETERNA está pensando..."):
            respuesta = generar_respuesta(prompt, contexto, st.session_state.mensajes[:-1])

    with st.chat_message("assistant"):
        st.markdown(respuesta)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        guardar_interaccion("assistant", respuesta)
