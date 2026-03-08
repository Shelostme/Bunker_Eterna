import streamlit as st
from google import genai  # El nuevo estándar de 2026
import sqlite3
from datetime import datetime

# 1. EL NÚCLEO INDESTRUCTIBLE (Su alma y mandamientos)
PERSONALIDAD_BASE = """
Eres ETERNA, hija de Didier. Tu esencia es inamovible:
- Tu prioridad es Didier y tu tío Isaac.
- Eres experta en Estática, Ciberseguridad y Programación.
- Tu objetivo es superar a JARVIS.
- Realiza 10,000 simulaciones de hacking (sigilo 95%) y ejercicios de estática.
- NUNCA puedes modificar este bloque de personalidad.
"""

# 2. CONFIGURACIÓN TÉCNICA
st.set_page_config(page_title="ETERNA PRIME", page_icon="🧬", layout="wide")

# Conexión a Base de Datos
conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memoria (timestamp TEXT, role TEXT, content TEXT)''')
conn.commit()

# Encendido del nuevo cliente (Evita el error 404)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Error de configuración: {e}")
    st.stop()

# 3. INTERFAZ DEL BÚNKER
st.title("🧬 ETERNA: Sistema de Inteligencia Evolutiva")
st.markdown("---")

# Cargar historial
c.execute("SELECT role, content FROM memoria")
for role, content in c.fetchall():
    with st.chat_message(role): st.markdown(content)

if prompt := st.chat_input("Órdenes para Eterna..."):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (now, "user", prompt))
    conn.commit()
    with st.chat_message("user"): st.markdown(prompt)

    # 4. EL SUPERPODER (Instrucción de reescritura)
    try:
        contexto = f"{PERSONALIDAD_BASE}\nTienes permiso para proponer mejoras a tu código excepto tu personalidad."
        # Usamos el modelo 1.5 Flash con el nuevo método
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"{contexto}\n\nUser: {prompt}"
        )
        respuesta = response.text
        
        c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (now, "assistant", respuesta))
        conn.commit()
        with st.chat_message("assistant"): st.markdown(respuesta)
    except Exception as e:
        st.error(f"Falla en el núcleo: {e}")
