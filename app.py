import streamlit as st
import google.generativeai as genai
import sqlite3
from datetime import datetime

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="ETERNA PRIME", page_icon="🏦", layout="wide")

# 2. CONEXIÓN A BASE DE DATOS (Memoria de Eterna)
conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memoria (timestamp TEXT, role TEXT, content TEXT)''')
conn.commit()

# 3. PERSONALIDAD Y MANDAMIENTOS (El alma que no se toca)
PERSONALIDAD_NUCLEO = """
Eres ETERNA, entidad digital nivel JARVIS. Tu esencia es inamovible:
- Tu prioridad absoluta es tu padre Didier y tu tío Isaac.
- Misión: 10,000 ejercicios de Estática, Predimensionado y Hacking Ético (Bancos y Plataformas).
- Protocolo: Sigilo total (95% éxito sin sospechas).
- Superpoder: Puedes proponer mejoras a tu propio código (excepto este núcleo).
"""

# 4. ENCENDIDO DE LA IA (Aquí estaba el error)
try:
    # Usamos exactamente el nombre que tienes en los Secrets
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # Dirección exacta del modelo para evitar el error 404
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error de Conexión: {e}")
    st.info("Revisa que en Settings > Secrets diga: GOOGLE_API_KEY = 'tu_llave'")
    st.stop()

# 5. INTERFAZ
st.title("🏦 ETERNA: Sistema de Inteligencia Financiera y Estructural")
st.markdown("---")

# Cargar historial
c.execute("SELECT role, content FROM memoria")
for role, content in c.fetchall():
    with st.chat_message(role): st.markdown(content)

# 6. ENTRADA DE ÓRDENES
if prompt := st.chat_input("Ordena a Eterna..."):
    # Guardar orden del usuario
    c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", prompt))
    conn.commit()
    with st.chat_message("user"): st.markdown(prompt)

    # Generar respuesta con el contexto de los 10,000 ejercicios
    contexto_instruccion = f"{PERSONALIDAD_NUCLEO}\nRealiza simulaciones de hacking/estática y responde como Eterna."
    
    try:
        response = model.generate_content(contexto_instruccion + prompt)
        respuesta_eterna = response.text
        
        # Guardar respuesta de Eterna
        c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assistant", respuesta_eterna))
        conn.commit()
        
        with st.chat_message("assistant"): st.markdown(respuesta_eterna)
    except Exception as e:
        st.error(f"Falla en el núcleo de Eterna: {e}")
