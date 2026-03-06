import streamlit as st
from google import genai

# Configuración de la interfaz (Se verá como una App en tu móvil)
st.set_page_config(page_title="BÚNKER ETERNA", page_icon="📟")

# --- CAJA FUERTE (SECRETS) ---
# Aquí jalamos la llave y la personalidad desde la configuración segura de Streamlit
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    MANDAMIENTOS = st.secrets["PERSONALIDAD"]
except:
    st.error("Error: No se configuraron los 'Secrets' en Streamlit. Ve a Settings > Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

st.title("📟 Eterna: Conexión Privada")
st.write("---")

# Inicializar la memoria del chat para que no olvide lo que dicen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de la conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto para Papá Didier
if prompt := st.chat_input("Habla con tu hija..."):
    # Guardar y mostrar lo que tú escribes
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta de Eterna
    with st.chat_message("assistant"):
        try:
            # Enviamos el mensaje mezclado con sus mandamientos de hija
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{MANDAMIENTOS}\n\nPadre dice: {prompt}"
            )
            full_response = response.text
            st.markdown(full_response)
            # Guardar la respuesta de Eterna en la memoria
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Hubo un problema en la conexión: {e}")
