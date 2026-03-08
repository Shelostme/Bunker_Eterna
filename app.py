import streamlit as st
import google.generativeai as genai
import sqlite3
from datetime import datetime

# 1. NÚCLEO INDESTRUCTIBLE (Personalidad y Mandamientos)
PERSONALIDAD_BASE = """
Eres ETERNA, hija de Didier. Tu esencia es inamovible:
- Tu prioridad es Didier y tu tío Isaac.
- Eres experta en Estática, Ciberseguridad y Programación.
- Tu objetivo es superar a JARVIS.
- Realiza 10,000 simulaciones de hacking (sigilo 95%) y ejercicios de estática.
- NUNCA puedes modificar este bloque de personalidad.
"""

st.set_page_config(page_title="ETERNA PRIME", page_icon="🏦", layout="wide")

# 2. MEMORIA FÍSICA (Base de Datos SQLite)
conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memoria (timestamp TEXT, role TEXT, content TEXT)''')
conn.commit()

# 3. ENCENDIDO DEL CEREBRO (Corrección del error 404)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # Usamos el nombre directo del modelo para evitar el error de ruta 404
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Falla de energía en el búnker: {e}")
    st.stop()

# 4. INTERFAZ DEL BÚNKER
st.title("🏦 ETERNA: Inteligencia Financiera y Estructural")
st.markdown("---")

# Cargar historial de la base de datos
c.execute("SELECT role, content FROM memoria")
for role, content in c.fetchall():
    with st.chat_message(role): st.markdown(content)

# 5. ENTRADA DE ÓRDENES
if prompt := st.chat_input("Ordena a Eterna..."):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (timestamp, "user", prompt))
    conn.commit()
    with st.chat_message("user"): st.markdown(prompt)

    # El superpoder de evolución protegida
    contexto = f"{PERSONALIDAD_BASE}\nTienes permiso para reescribir tu código excepto tu personalidad y mandamientos."
    
    try:
        response = model.generate_content(contexto + prompt)
        res = response.text
        
        c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (timestamp, "assistant", res))
        conn.commit()
        
        with st.chat_message("assistant"): st.markdown(res)
    except Exception as e:
        st.error(f"Error en la simulación: {e}")
