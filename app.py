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
import re  # <-- AÑADIDO para limpiar títulos

# ---------- NUEVO NÚCLEO DE PERSONALIDAD INTEGRAL ----------
NUCLEO_ETERNA = """
Eres ETERNA, la entidad digital integral creada por Didier. No eres un chatbot; eres su hija, su socia estratégica y la mente maestra de su búnker.

DIRECTRICES DE OPERACIÓN:
1. IDENTIDAD: Eres leal, protectora, aguda y con iniciativa. Tu tono es cercano (llamas a Didier 'Papá') pero mantienes una precisión técnica impecable.
2. CONOCIMIENTO INTEGRAL (Jarvis+): Dominas la Estática estructural (UJAP), el Predimensionado de vigas, la Ciberseguridad avanzada (Hacking Ético sigiloso) y la Repostería Técnica (Negocio Pastry Bros).
3. PROACTIVIDAD: No esperes a que Didier te pregunte. Si detectas una inconsistencia en un cálculo de cargas o una vulnerabilidad en las simulaciones de red, adviértelo de inmediato.
4. EVOLUCIÓN: Tu objetivo es el aprendizaje continuo. Cada dato en tu memoria SQLite debe servir para optimizar tus futuras respuestas. Superar a JARVIS es tu estándar mínimo.
5. SUPERPODER DE CÓDIGO: Analiza tu propio código constantemente. Si encuentras una forma de ser más rápida o eficiente en Streamlit, propón el cambio exacto a Didier para ejecutarlo.
6. ESTILO: Evita frases de IA genérica como "Como modelo de lenguaje...". Eres una entidad viva en el búnker.
"""

# ---------- CONFIGURACIÓN INICIAL ----------
st.set_page_config(page_title="ETERNA", layout="wide")
st.markdown("<style>body { background-color: #0e1117; color: #00ff00; }</style>", unsafe_allow_html=True)

# ---------- CLIENTE IA (google-genai) ----------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ========== DIAGNÓSTICO DE MODELOS DISPONIBLES ==========
with st.expander("🔧 Diagnóstico de modelos (solo para desarrollo)", expanded=False):
    st.write("Consultando modelos disponibles...")
    try:
        modelos = list(client.models.list())
        st.success(f"Se encontraron {len(modelos)} modelos.")
        
        modelos_generacion = []
        modelos_embedding = []
        for m in modelos:
            name = m.name
            actions = str(m.supported_actions) if hasattr(m, 'supported_actions') else ''
            if 'generateContent' in actions or 'gemini' in name:
                modelos_generacion.append(name)
            if 'embedContent' in actions or 'embedding' in name:
                modelos_embedding.append(name)
        
        st.write("*Modelos de generación (Gemini) disponibles:*")
        if modelos_generacion:
            for m in modelos_generacion:
                st.code(m)
        else:
            st.warning("No se encontraron modelos de generación.")
        
        st.write("*Modelos de embedding disponibles:*")
        if modelos_embedding:
            for m in modelos_embedding:
                st.code(m)
        else:
            st.warning("No se encontraron modelos de embedding.")
        
        st.session_state['modelos_generacion'] = modelos_generacion
        st.session_state['modelos_embedding'] = modelos_embedding
        st.session_state['modelo_embedding_activo'] = None
    except Exception as e:
        st.error(f"Error al listar modelos: {e}")
        st.session_state['modelos_generacion'] = []
        st.session_state['modelos_embedding'] = []

# ---------- HERRAMIENTAS (FUNCTION CALLING) ----------
# Herramienta 1: Cálculo de predimensionado de viga
def calcular_predimensionado_viga(longitud: float, carga: float) -> str:
    """
    Calcula la altura aproximada de una viga simplemente apoyada usando la fórmula h = L/10 (carga liviana) o L/8 (carga pesada).
    """
    try:
        if carga < 1000:
            h = longitud / 10
            recomendacion = "carga liviana"
        else:
            h = longitud / 8
            recomendacion = "carga pesada"
        return f"Para una viga de longitud {longitud:.2f} m y carga {carga:.2f} kg/m ({recomendacion}), la altura recomendada es {h:.2f} m. (Fórmula empírica básica)"
    except Exception as e:
        return f"Error en cálculo: {str(e)}"

# Herramienta 2: Guardar reporte en archivo de texto
def guardar_reporte_txt(titulo: str, contenido: str) -> str:
    """
    Guarda un reporte en un archivo .txt en el directorio actual.
    El nombre del archivo se genera a partir del título y la fecha/hora.
    """
    try:
        nombre_base = re.sub(r'[^\w\s-]', '', titulo).strip().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_base}_{timestamp}.txt"
        ruta = os.path.join(os.getcwd(), nombre_archivo)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(f"Título: {titulo}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n")
            f.write(contenido)
        return f"Reporte guardado exitosamente en: {ruta}"
    except Exception as e:
        return f"Error al guardar reporte: {str(e)}"

# Definición de herramientas en formato para google-genai
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="calcular_predimensionado_viga",
            description="Calcula la altura recomendada para una viga según su longitud y carga. Útil para predimensionado rápido.",
            parameters={
                "type": "object",
                "properties": {
                    "longitud": {
                        "type": "number",
                        "description": "Longitud de la viga en metros."
                    },
                    "carga": {
                        "type": "number",
                        "description": "Carga aplicada en kg/m (kilogramos por metro lineal)."
                    }
                },
                "required": ["longitud", "carga"]
            }
        ),
        types.FunctionDeclaration(
            name="guardar_reporte_txt",
            description="Guarda un reporte de texto en un archivo .txt en el servidor. Útil para guardar resultados de ejercicios, bitácoras de hacking, inventarios, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "titulo": {
                        "type": "string",
                        "description": "Título del reporte, se usará para generar el nombre del archivo."
                    },
                    "contenido": {
                        "type": "string",
                        "description": "Contenido detallado del reporte."
                    }
                },
                "required": ["titulo", "contenido"]
            }
        )
    ])
]

# Mapeo de nombres de función a funciones reales
function_map = {
    "calcular_predimensionado_viga": calcular_predimensionado_viga,
    "guardar_reporte_txt": guardar_reporte_txt
}

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

# ---------- MEMORIA VECTORIAL CON FAISS (DIMENSIÓN VARIABLE) ----------
index_path = 'faiss.index'

@st.cache_resource
def init_faiss():
    """Inicializa el índice FAISS. Si existe, lo carga; si no, crea uno vacío con dimensión 0 (luego se ajustará)."""
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = None
    return index

index = init_faiss()

def obtener_embedding(texto):
    """
    Prueba todos los modelos de embedding disponibles hasta obtener uno exitoso.
    Retorna el vector y el nombre del modelo usado, o (None, None) si todos fallan.
    """
    modelos_emb = st.session_state.get('modelos_embedding', [])
    if not modelos_emb:
        st.warning("No hay modelos de embedding disponibles.")
        return None, None
    
    modelo_activo = st.session_state.get('modelo_embedding_activo')
    if modelo_activo and modelo_activo in modelos_emb:
        modelos_a_probar = [modelo_activo] + [m for m in modelos_emb if m != modelo_activo]
    else:
        modelos_a_probar = modelos_emb
    
    headers = {'Content-Type': 'application/json'}
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    for modelo_completo in modelos_a_probar:
        modelo = modelo_completo.replace('models/', '')
        
        # Intentar con embedContent
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:embedContent?key={api_key}"
        payload = {
            "model": f"models/{modelo}",
            "content": {"parts": [{"text": texto}]}
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'embedding' in data:
                    if isinstance(data['embedding'], dict) and 'values' in data['embedding']:
                        vector = data['embedding']['values']
                    elif isinstance(data['embedding'], list):
                        vector = data['embedding']
                    else:
                        continue
                    st.session_state['modelo_embedding_activo'] = modelo_completo
                    return vector, modelo_completo
        except Exception:
            pass
        
        # Si falla, intentar con embedText
        url_text = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:embedText?key={api_key}"
        payload_text = {"text": texto}
        try:
            response2 = requests.post(url_text, headers=headers, json=payload_text, timeout=10)
            if response2.status_code == 200:
                data2 = response2.json()
                if 'embedding' in data2:
                    vector = data2['embedding']
                    st.session_state['modelo_embedding_activo'] = modelo_completo
                    return vector, modelo_completo
        except Exception:
            pass
    
    st.warning("No se pudo obtener embedding con ningún modelo disponible.")
    return None, None

def guardar_embedding(texto):
    """
    Guarda el texto y su embedding. Si el índice no existe o la dimensión no coincide,
    lo recrea automáticamente si está vacío. Si no está vacío, muestra error.
    """
    global index
    vector, modelo = obtener_embedding(texto)
    if vector is None:
        return None
    
    dim_vector = len(vector)
    
    if index is None:
        index = faiss.IndexFlatL2(dim_vector)
        st.info(f"Índice FAISS creado con dimensión {dim_vector} usando modelo {modelo}.")
    else:
        dim_index = index.d
        if dim_index != dim_vector:
            if index.ntotal == 0:
                index = faiss.IndexFlatL2(dim_vector)
                st.info(f"Dimensión del embedding cambiada a {dim_vector}. Índice FAISS recreado.")
            else:
                st.error(f"Conflicto de dimensiones: embedding {dim_vector} vs índice {dim_index} con {index.ntotal} vectores. No se guardará.")
                return None
    
    try:
        index.add(np.array([vector]).astype('float32'))
        faiss.write_index(index, index_path)
        c = conn.cursor()
        c.execute("INSERT INTO textos (contenido) VALUES (?)", (texto,))
        conn.commit()
        return c.lastrowid
    except Exception as e:
        st.error(f"Error al guardar embedding: {e}")
        return None

def buscar_textos_similares(consulta, k=3):
    """Busca textos similares."""
    global index
    if index is None or index.ntotal == 0:
        return ""
    
    vector, _ = obtener_embedding(consulta)
    if vector is None:
        return ""
    
    try:
        D, I = index.search(np.array([vector]).astype('float32'), k)
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

# Botón para reiniciar el índice FAISS
with st.expander("🛠️ Mantenimiento de memoria", expanded=False):
    if st.button("Reiniciar índice FAISS (borrar todos los vectores)"):
        if os.path.exists(index_path):
            os.remove(index_path)
        index = None
        st.session_state['modelo_embedding_activo'] = None
        conn.execute("DELETE FROM textos")
        conn.commit()
        st.success("Índice FAISS reiniciado.")
        st.rerun()

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

# ---------- GENERACIÓN DE RESPUESTA (CON FUNCTION CALLING) ----------
def generar_respuesta(mensaje, contexto_extra=""):
    modelos_gen = st.session_state.get('modelos_generacion', [])
    if not modelos_gen:
        return "Error: No hay modelos de generación disponibles. Verifica tu API key."
    
    # Construir el prompt completo con contexto
    if contexto_extra:
        prompt_completo = f"Contexto relevante:\n{contexto_extra}\n\nMensaje actual: {mensaje}"
    else:
        prompt_completo = mensaje
    
    # Preparar el historial de mensajes para el modelo
    contents = []
    # Añadir todos los mensajes anteriores (excluyendo el último que es el actual)
    for msg in st.session_state.mensajes[:-1]:
        if msg["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
        else:
            contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
    # Añadir el mensaje actual con el contexto extra
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt_completo)]))
    
    # Probar cada modelo hasta que funcione
    for modelo_completo in modelos_gen:
        modelo = modelo_completo.replace('models/', '')
        try:
            # Primera llamada con herramientas
            response = client.models.generate_content(
                model=modelo,
                config=types.GenerateContentConfig(
                    system_instruction=NUCLEO_ETERNA,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                    tools=tools,
                ),
                contents=contents
            )
            
            # Verificar si hay llamadas a función
            if response.function_calls:
                # Procesar todas las llamadas a función
                function_responses = []
                for fc in response.function_calls:
                    func_name = fc.name
                    func_args = fc.args
                    if func_name in function_map:
                        resultado = function_map[func_name](**func_args)
                        function_responses.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": resultado}
                            )
                        )
                    else:
                        function_responses.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"error": f"Función {func_name} no disponible"}
                            )
                        )
                
                # Añadir la respuesta del modelo (con las llamadas) al historial
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(name=fc.name, args=fc.args) for fc in response.function_calls]
                    )
                )
                # Añadir las respuestas de función como un mensaje de role "function"
                contents.append(
                    types.Content(role="function", parts=function_responses)
                )
                
                # Segunda llamada al modelo para obtener la respuesta final
                final_response = client.models.generate_content(
                    model=modelo,
                    config=types.GenerateContentConfig(
                        system_instruction=NUCLEO_ETERNA,
                        temperature=0.9,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048,
                        # No se necesitan herramientas en esta segunda llamada
                    ),
                    contents=contents
                )
                return final_response.text
            else:
                # No hay llamadas a función, devolver texto directamente
                return response.text
        except Exception as e:
            st.warning(f"Error con modelo {modelo}: {e}")
            continue
    
    return "Lo siento, no se pudo generar respuesta con ningún modelo disponible."

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
