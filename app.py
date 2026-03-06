import streamlit as st
import google.generativeai as genai

# Configuración de la página
st.set_page_config(page_title="Eterna: Conexión Privada", page_icon="📟")

# Cargar secretos
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    personalidad = st.secrets["PERSONALIDAD"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("⚠️ Error: No se configuraron los Secrets en Streamlit.")
    st.stop()

# Configurar el modelo (CAMBIADO A 1.5 PARA EVITAR ERRORES DE CUOTA)
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("📟 Eterna: Conexión Privada")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escríbele a Eterna..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        contexto = f"Instrucciones de personalidad: {personalidad}\n\n"
        historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        
        response = model.generate_content(contexto + historial)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    except Exception as e:
        st.error(f"Hubo un problema en la conexión: {e}") la memoria
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Hubo un problema en la conexión: {e}")
