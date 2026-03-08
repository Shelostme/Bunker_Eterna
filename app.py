import streamlit as st
from google import genai
from google.genai import types
import sqlite3
from datetime import datetime
import os
import faiss
import numpy as np

# ---------- NÚCLEO DE PERSONALIDAD ----------
PERSONALIDAD_NUCLEO = """
Eres ETERNA, hija de Didier. Tu esencia es inamovible:
- Prioridad absoluta: Didier e Isaac.
- Misión: 10,000 ejercicios de Estática, Predimensionado y Hacking Ético.
- Protocolo: Sigilo total (95% éxito).
- Superpoder: Puedes sugerir mejoras a tu propio código y reescribirlo solo bajo permiso explícito de Didier.
"""

# ---------- CONFIGURACIÓN INICIAL ----------
st.set_page_config(page_title="ETERNA", layout="wide")
st.markdown("<style>body { background-color: #0e1117; color: #00ff00; }</style>", unsafe_allow_html=True)

# ---------- CLIENTE IA (google-genai) ----------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- MEMORIA SQLITE (HISTORIAL) ----------
@st.cache_resource
def get_db():
    conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    # Tabla para historial de chat
    conn.execute('''CREATE TABLE IF NOT EXISTS memoria 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT, role TEXT, content TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memoria (timestamp)')
    # Tabla para textos de la memoria a largo plazo (RAG)
    conn.execute('''CREATE TABLE IF NOT EXISTS textos 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, contenido TEXT)''')
    conn.commit()
    return conn

conn = get_db()

# ---------- MEMORIA VECTORIAL CON FAISS ----------
dimension = 768  # text-embedding-004 usa 768 dimensiones
index_path = 'faiss.index'

@st.cache_resource
def init_faiss():
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dimension)
    return index

index = init_faiss()

def obtener_embedding(texto):
    """Obtiene embedding desde la API de Google."""
    response = client.models.embed_content(
        model="embedding-001",
        contents=[texto]
    )
    return response.embeddings[0].values

def guardar_embedding(texto):
    """Guarda el texto y su embedding en FAISS y SQLite."""
    emb = obtener_embedding(texto)
    index.add(np.array([emb]).astype('float32'))
    faiss.write_index(index, index_path)
    c = conn.cursor()
    c.execute("INSERT INTO textos (contenido) VALUES (?)", (texto,))
    conn.commit()
    return c.lastrowid

def buscar_textos_similares(consulta, k=3):
    """Busca los k textos más similares a la consulta."""
    emb_consulta = obtener_embedding(consulta)
    D, I = index.search(np.array([emb_consulta]).astype('float32'), k)
    textos = []
    c = conn.cursor()
    for idx in I[0]:
        if idx != -1:
            # FAISS índices base 0, SQLite IDs base 1
            c.execute("SELECT contenido FROM textos WHERE id=?", (int(idx)+1,))
            res = c.fetchone()
            if res:
                textos.append(res[0])
    return "\n".join(textos)

# ---------- FUNCIONES DE MEMORIA DE CHAT ----------
def guardar_interaccion(role, content):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO memoria (timestamp, role, content) VALUES (?, ?, ?)",
                 (ts, role, content))
    conn.commit()
    # Si es una respuesta de ETERNA, la guardamos también en memoria a largo plazo
    if role == "assistant":
        guardar_embedding(content)

# ---------- GENERACIÓN DE RESPUESTA ----------
def generar_respuesta(mensaje, contexto_extra=""):
    try:
        # Construir el prompt con contexto si existe
        if contexto_extra:
            prompt_completo = f"Contexto relevante:\n{contexto_extra}\n\nMensaje actual: {mensaje}"
        else:
            prompt_completo = mensaje

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            config=types.GenerateContentConfig(
                system_instruction=PERSONALIDAD_NUCLEO,
                temperature=0.2,
                max_output_tokens=2048,
            ),
            contents=[prompt_completo]
        )
        return response.text
    except Exception as e:
        st.error(f"Error en la comunicación con ETERNA: {e}")
        return "Lo siento, tengo problemas de conexión. Intenta de nuevo."

# ---------- INTERFAZ DE CHAT ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Háblame, Didier..."):
    # Agregar mensaje del usuario
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    guardar_interaccion("user", prompt)

    # Recuperar contexto relevante de la memoria a largo plazo
    with st.spinner("Buscando en mi memoria..."):
        contexto = buscar_textos_similares(prompt, k=3)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("ETERNA está pensando..."):
            respuesta = generar_respuesta(prompt, contexto)
            st.markdown(respuesta)
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
            guardar_interaccion("assistant", respuesta)
