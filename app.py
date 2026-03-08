import requests
import json
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
# Usamos la configuración por defecto (ya no forzamos v1)
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- MEMORIA SQLITE (HISTORIAL) ----------
@st.cache_resource
def get_db():
    conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS memoria 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT, role TEXT, content TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memoria (timestamp)')
    conn.execute('''CREATE TABLE IF NOT EXISTS textos 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, contenido TEXT)''')
    conn.commit()
    return conn

conn = get_db()

# ---------- MEMORIA VECTORIAL CON FAISS ----------
# gemini-embedding-001 usa 768 dimensiones (igual que los anteriores)
dimension = 768
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
    """
    Obtiene embedding usando el modelo recomendado por Google: gemini-embedding-001
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={st.secrets['GOOGLE_API_KEY']}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": texto}]}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # La respuesta tiene estructura: {"embedding": {"values": [0.1, 0.2, ...]}}
            if 'embedding' in data and 'values' in data['embedding']:
                return data['embedding']['values']
            else:
                st.warning("Estructura de respuesta inesperada")
                return None
        else:
            st.warning(f"Error en embedding (código {response.status_code})")
            return None
    except Exception as e:
        st.warning(f"Excepción al obtener embedding: {e}")
        return None

def guardar_embedding(texto):
    """Guarda el texto y su embedding en FAISS y SQLite."""
    emb = obtener_embedding(texto)
    if emb is None:
        return None
    
    try:
        index.add(np.array([emb]).astype('float32'))
        faiss.write_index(index, index_path)
        c = conn.cursor()
        c.execute("INSERT INTO textos (contenido) VALUES (?)", (texto,))
        conn.commit()
        return c.lastrowid
    except Exception as e:
        st.error(f"Error al guardar embedding: {e}")
        return None

def buscar_textos_similares(consulta, k=3):
    """Busca los k textos más similares a la consulta."""
    emb_consulta = obtener_embedding(consulta)
    if emb_consulta is None:
        return ""
    
    try:
        D, I = index.search(np.array([emb_consulta]).astype('float32'), k)
        textos = []
        c = conn.cursor()
        for idx in I[0]:
            if idx != -1:
                c.execute("SELECT contenido FROM textos WHERE id=?", (int(idx)+1,))
                res = c.fetchone()
                if res:
                    textos.append(res[0])
        return "\n".join(textos)
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return ""

# ---------- FUNCIONES DE MEMORIA DE CHAT ----------
def guardar_interaccion(role, content):
    ts = datetime.now().isoformat()
    try:
        conn.execute("INSERT INTO memoria (timestamp, role, content) VALUES (?, ?, ?)",
                     (ts, role, content))
        conn.commit()
    except Exception as e:
        st.error(f"Error guardando en historial: {e}")
    
    if role == "assistant":
        guardar_embedding(content)

# ---------- GENERACIÓN DE RESPUESTA ----------
def generar_respuesta(mensaje, contexto_extra=""):
    # Lista de modelos a probar (en orden de preferencia)
    modelos = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    for modelo in modelos:
        try:
            if contexto_extra:
                prompt_completo = f"Contexto relevante:\n{contexto_extra}\n\nMensaje actual: {mensaje}"
            else:
                prompt_completo = mensaje

            response = client.models.generate_content(
                model=modelo,
                config=types.GenerateContentConfig(
                    system_instruction=PERSONALIDAD_NUCLEO,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
                contents=[prompt_completo]
            )
            return response.text
        except Exception as e:
            if "404" in str(e):
                continue
            st.warning(f"Error con modelo {modelo}: {e}")
            continue
    
    return "Lo siento, tengo problemas de conexión. Intenta de nuevo."

# ---------- INTERFAZ DE CHAT ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Háblame, Didier..."):
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    guardar_interaccion("user", prompt)

    with st.spinner("Buscando en mi memoria..."):
        contexto = buscar_textos_similares(prompt, k=3)

    with st.chat_message("assistant"):
        with st.spinner("ETERNA está pensando..."):
            respuesta = generar_respuesta(prompt, contexto)
            st.markdown(respuesta)
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
            guardar_interaccion("assistant", respuesta)
