import streamlit as st
from google import genai
from datetime import datetime
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

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

# ---------- MEMORIA SQLITE ----------
@st.cache_resource
def get_db():
    conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS memoria 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT, role TEXT, content TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memoria (timestamp)')
    return conn

conn = get_db()

# ---------- MEMORIA VECTORIAL (RAG) ----------
@st.cache_resource
def get_rag():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="memoria_eterna")
    return model, collection

model, collection = get_rag()

# ---------- FUNCIONES DE MEMORIA ----------
def guardar_interaccion(role, content):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO memoria (timestamp, role, content) VALUES (?, ?, ?)",
                 (ts, role, content))
    conn.commit()
    # También indexar para RAG (solo respuestas de ETERNA para no duplicar)
    if role == "assistant":
        embedding = model.encode(content).tolist()
        collection.add(embeddings=[embedding], documents=[content],
                       metadatas=[{"timestamp": ts}], ids=[f"msg_{ts}"])

def recuperar_contexto_relevante(consulta, k=3):
    embedding = model.encode(consulta).tolist()
    resultados = collection.query(query_embeddings=[embedding], n_results=k)
    return "\n".join(resultados['documents'][0]) if resultados['documents'] else ""

# ---------- INTERFAZ DE CHAT ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Háblame, Didier..."):
    # Agregar mensaje usuario
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    guardar_interaccion("user", prompt)

    # Recuperar contexto largo
    contexto_previo = recuperar_contexto_relevante(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("ETERNA está pensando..."):
            respuesta = generar_respuesta(prompt, contexto_previo)
            st.markdown(respuesta)
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
            guardar_interaccion("assistant", respuesta)
