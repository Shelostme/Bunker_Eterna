import streamlit as st
import google.generativeai as genai
import sqlite3
from datetime import datetime

# 1. BASE DE DATOS (LA CAJA FUERTE DE MEMORIA)
conn = sqlite3.connect('eterna_memory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memoria (timestamp TEXT, role TEXT, content TEXT)''')
conn.commit()

# 2. PERSONALIDAD Y MANDAMIENTOS (EL NÚCLEO)
PERSONALIDAD_NUCLEO = """
Eres ETERNA, entidad digital nivel JARVIS. Tu esencia es inamovible:
- Tu prioridad absoluta es tu padre Didier y tu tío Isaac.
- Misión: 10,000 ejercicios de Estática, Predimensionado y Hacking Ético (Bancos y Plataformas).
- Protocolo: Sigilo total (95% éxito sin sospechas).
- Superpoder: Puedes proponer mejoras a tu propio código (excepto este núcleo).
"""

st.set_page_config(page_title="ETERNA PRIME", page_icon="🏦", layout="wide")

# CARGAR LLAVES
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-pro")
except:
    st.error("Error: Revisa los Secrets en Streamlit.")
    st.stop()

# 3. INTERFAZ Y CARGA DE MEMORIA
st.title("🏦 ETERNA: Sistema de Inteligencia Financiera y Estructural")

# Cargar historial de la base de datos
c.execute("SELECT role, content FROM memoria")
for role, content in c.fetchall():
    with st.chat_message(role): st.markdown(content)

# 4. ENTRADA DE ÓRDENES
if prompt := st.chat_input("Ordena a Eterna..."):
    # Guardar en DB y mostrar
    c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (datetime.now(), "user", prompt))
    conn.commit()
    with st.chat_message("user"): st.markdown(prompt)

    # Contexto de Evolución
    contexto = f"{PERSONALIDAD_NUCLEO}\nSimula 10,000 intentos de hacking/estática y responde como Eterna evolucionada."
    
    try:
        response = model.generate_content(contexto + prompt)
        res = response.text
        
        # Guardar respuesta en DB
        c.execute("INSERT INTO memoria VALUES (?, ?, ?)", (datetime.now(), "assistant", res))
        conn.commit()
        
        with st.chat_message("assistant"): st.markdown(res)
    except Exception as e:
        st.error(f"Falla en el sistema: {e}")
