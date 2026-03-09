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
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess

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

# Herramienta 3: Consultar clima actual (vía wttr.in)
def consultar_clima(ciudad: str) -> str:
    try:
        url = f"https://wttr.in/{ciudad}?format=%C+%t+%w&m"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"Clima en {ciudad}: {response.text.strip()}"
        else:
            return f"No se pudo obtener el clima para {ciudad} (código {response.status_code})"
    except Exception as e:
        return f"Error consultando clima: {str(e)}"

# Herramienta 4: Agregar tarea a la lista pendiente (SQLite)
def agregar_tarea(descripcion: str, fecha_limite: str = "") -> str:
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tareas 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     descripcion TEXT,
                     fecha_limite TEXT,
                     completada INTEGER DEFAULT 0,
                     creada TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute("INSERT INTO tareas (descripcion, fecha_limite) VALUES (?, ?)",
                  (descripcion, fecha_limite))
        conn.commit()
        return f"Tarea '{descripcion}' agregada correctamente."
    except Exception as e:
        return f"Error al agregar tarea: {str(e)}"

# Herramienta 5: Listar tareas pendientes
def listar_tareas() -> str:
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tareas 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     descripcion TEXT,
                     fecha_limite TEXT,
                     completada INTEGER DEFAULT 0,
                     creada TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute("SELECT id, descripcion, fecha_limite, creada FROM tareas WHERE completada=0 ORDER BY creada DESC")
        filas = c.fetchall()
        if not filas:
            return "No hay tareas pendientes."
        resultado = "Tareas pendientes:\n"
        for id, desc, fl, creada in filas:
            fl_str = f" (para: {fl})" if fl else ""
            resultado += f"- [{id}] {desc}{fl_str} (creada: {creada[:10]})\n"
        return resultado
    except Exception as e:
        return f"Error al listar tareas: {str(e)}"

# Herramienta 6: Ejecutar comando seguro (whitelist)
def ejecutar_comando_seguro(comando: str) -> str:
    comandos_permitidos = {
        'date': ['date'],
        'uptime': ['uptime'],
        'df': ['df', '-h'],
        'free': ['free', '-h'],
        'ls': ['ls', '-la'],
        'whoami': ['whoami'],
        'uname': ['uname', '-a'],
    }
    if comando not in comandos_permitidos:
        return f"Comando '{comando}' no permitido. Los comandos válidos son: {', '.join(comandos_permitidos.keys())}"
    try:
        resultado = subprocess.run(comandos_permitidos[comando], capture_output=True, text=True, timeout=5)
        if resultado.returncode == 0:
            return f"\n{resultado.stdout}\n"
        else:
            return f"Error: {resultado.stderr}"
    except Exception as e:
        return f"Excepción al ejecutar comando: {str(e)}"

# Herramienta 7: Enviar email real (SMTP)
def enviar_email_real(destinatario: str, asunto: str, cuerpo: str) -> str:
    try:
        smtp_server = st.secrets["EMAIL_SMTP_SERVER"]
        smtp_port = st.secrets["EMAIL_SMTP_PORT"]
        remitente = st.secrets["EMAIL_REMITENTE"]
        password = st.secrets["EMAIL_PASSWORD"]

        msg = MIMEMultipart()
        msg['From'] = remitente
        msg['To'] = destinatario
        msg['Subject'] = asunto
        msg.attach(MIMEText(cuerpo, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(remitente, password)
        server.send_message(msg)
        server.quit()
        return f"Email enviado correctamente a {destinatario}."
    except Exception as e:
        return f"Error al enviar email: {str(e)}"

# Herramienta 8: Controlar dispositivo en Home Assistant
def controlar_dispositivo(entidad: str, accion: str) -> str:
    """
    Controla un dispositivo a través de Home Assistant.
    Requiere HA_URL y HA_TOKEN en secrets.
    Ejemplo de entidad: "light.sala"
    """
    try:
        ha_url = st.secrets["HA_URL"].rstrip('/')
        ha_token = st.secrets["HA_TOKEN"]
    except KeyError:
        return "Error: Configura HA_URL y HA_TOKEN en secrets para usar control de dispositivos."

    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    if accion.lower() in ["encender", "on"]:
        service = "turn_on"
    elif accion.lower() in ["apagar", "off"]:
        service = "turn_off"
    else:
        return f"Acción '{accion}' no soportada. Usa 'encender'/'on' o 'apagar'/'off'."

    try:
        domain = entidad.split('.')[0]
    except:
        return "Formato de entidad inválido. Debe ser 'dominio.nombre' (ej. 'light.sala')."

    url = f"{ha_url}/api/services/{domain}/{service}"
    payload = {"entity_id": entidad}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            return f"Comando '{accion}' enviado a {entidad} correctamente."
        else:
            return f"Error en Home Assistant: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error conectando con Home Assistant: {str(e)}"

# Definición de herramientas en formato para google-genai
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="calcular_predimensionado_viga",
            description="Calcula la altura recomendada para una viga según su longitud y carga.",
            parameters={
                "type": "object",
                "properties": {
                    "longitud": {"type": "number", "description": "Longitud de la viga en metros."},
                    "carga": {"type": "number", "description": "Carga aplicada en kg/m."}
                },
                "required": ["longitud", "carga"]
            }
        ),
        types.FunctionDeclaration(
            name="guardar_reporte_txt",
            description="Guarda un reporte de texto en un archivo .txt en el servidor.",
            parameters={
                "type": "object",
                "properties": {
                    "titulo": {"type": "string", "description": "Título del reporte."},
                    "contenido": {"type": "string", "description": "Contenido detallado."}
                },
                "required": ["titulo", "contenido"]
            }
        ),
        types.FunctionDeclaration(
            name="consultar_clima",
            description="Obtiene el clima actual de una ciudad.",
            parameters={
                "type": "object",
                "properties": {
                    "ciudad": {"type": "string", "description": "Nombre de la ciudad (ej. 'Caracas')."}
                },
                "required": ["ciudad"]
            }
        ),
        types.FunctionDeclaration(
            name="agregar_tarea",
            description="Agrega una nueva tarea a la lista de pendientes.",
            parameters={
                "type": "object",
                "properties": {
                    "descripcion": {"type": "string", "description": "Descripción de la tarea."},
                    "fecha_limite": {"type": "string", "description": "Fecha límite opcional (formato YYYY-MM-DD)."}
                },
                "required": ["descripcion"]
            }
        ),
        types.FunctionDeclaration(
            name="listar_tareas",
            description="Lista todas las tareas pendientes.",
            parameters={"type": "object", "properties": {}}
        ),
        types.FunctionDeclaration(
            name="ejecutar_comando_seguro",
            description="Ejecuta un comando del sistema de una lista segura (date, uptime, df, free, ls, whoami, uname).",
            parameters={
                "type": "object",
                "properties": {
                    "comando": {"type": "string", "description": "Nombre del comando a ejecutar (ej. 'date')."}
                },
                "required": ["comando"]
            }
        ),
        types.FunctionDeclaration(
            name="enviar_email_real",
            description="Envía un email real vía SMTP. Requiere configuración en secrets.",
            parameters={
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string", "description": "Dirección de correo del destinatario."},
                    "asunto": {"type": "string", "description": "Asunto del email."},
                    "cuerpo": {"type": "string", "description": "Cuerpo del mensaje."}
                },
                "required": ["destinatario", "asunto", "cuerpo"]
            }
        ),
        types.FunctionDeclaration(
            name="controlar_dispositivo",
            description="Controla un dispositivo (luces, enchufes, etc.) a través de Home Assistant. Requiere configuración en secrets.",
            parameters={
                "type": "object",
                "properties": {
                    "entidad": {"type": "string", "description": "ID de la entidad en Home Assistant (ej. 'light.sala')."},
                    "accion": {"type": "string", "description": "Acción: 'encender'/'on' o 'apagar'/'off'."}
                },
                "required": ["entidad", "accion"]
            }
        )
    ])
]

# Mapeo de nombres de función a funciones reales
function_map = {
    "calcular_predimensionado_viga": calcular_predimensionado_viga,
    "guardar_reporte_txt": guardar_reporte_txt,
    "consultar_clima": consultar_clima,
    "agregar_tarea": agregar_tarea,
    "listar_tareas": listar_tareas,
    "ejecutar_comando_seguro": ejecutar_comando_seguro,
    "enviar_email_real": enviar_email_real,
    "controlar_dispositivo": controlar_dispositivo
}

# ---------- MÓDULO DE PLANIFICACIÓN AUTÓNOMA (ReAct) ----------
def planificar_y_ejecutar(objetivo):
    """
    Toma un objetivo complejo, lo descompone en pasos y los ejecuta usando las herramientas disponibles.
    """
    plan_prompt = f"""
Eres ETERNA, una agente autónoma. Tu objetivo es: "{objetivo}"

Descompón este objetivo en una secuencia de pasos. Cada paso debe ser una acción que puedas realizar con tus herramientas disponibles:
- controlar_dispositivo(entidad, accion)
- calcular_predimensionado_viga(longitud, carga)
- guardar_reporte_txt(titulo, contenido)
- consultar_clima(ciudad)
- agregar_tarea(descripcion, fecha_limite)
- listar_tareas()
- ejecutar_comando_seguro(comando)
- enviar_email_real(destinatario, asunto, cuerpo)

Devuelve los pasos en formato JSON, así:
[
    {{"herramienta": "controlar_dispositivo", "argumentos": {{"entidad": "light.sala", "accion": "apagar"}}}},
    {{"herramienta": "consultar_clima", "argumentos": {{"ciudad": "Caracas"}}}}
]
Si no es posible descomponer, responde con un JSON vacío [].
"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(temperature=0.2),
            contents=[plan_prompt]
        )
        plan_text = response.text.strip()
        # Extraer JSON
        start = plan_text.find('[')
        end = plan_text.rfind(']') + 1
        if start != -1 and end != 0:
            plan_json = plan_text[start:end]
            pasos = json.loads(plan_json)
        else:
            pasos = []
    except Exception as e:
        return f"Error al generar plan: {e}"

    if not pasos:
        return "No pude descomponer el objetivo en pasos. Intenta ser más específico."

    resultados = []
    for paso in pasos:
        herramienta = paso.get("herramienta")
        args = paso.get("argumentos", {})
        if herramienta in function_map:
            try:
                resultado = function_map[herramienta](**args)
                resultados.append(f"Paso '{herramienta}': {resultado}")
            except Exception as e:
                resultados.append(f"Error en paso '{herramienta}': {e}")
        else:
            resultados.append(f"Herramienta '{herramienta}' no disponible")

    return "\n".join(resultados)

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
    conn.execute('''CREATE TABLE IF NOT EXISTS tareas 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     descripcion TEXT,
                     fecha_limite TEXT,
                     completada INTEGER DEFAULT 0,
                     creada TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

conn = get_db()

# ---------- MEMORIA VECTORIAL CON FAISS (DIMENSIÓN VARIABLE) ----------
index_path = 'faiss.index'

@st.cache_resource
def init_faiss():
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = None
    return index

index = init_faiss()

def obtener_embedding(texto):
    modelos_emb = st.session_state.get('modelos_embedding', [])
    if not modelos_emb:
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
    
    return None, None

def guardar_embedding(texto):
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

# ---------- GENERACIÓN DE RESPUESTA (CON FUNCTION CALLING Y PLANIFICACIÓN) ----------
def generar_respuesta(mensaje, contexto_extra=""):
    # Detectar si el mensaje es un objetivo complejo (puedes ajustar la lógica)
    palabras_clave = ["prepara", "organiza", "planifica", "secuencia", "automatiza", "haz todo lo necesario"]
    if any(palabra in mensaje.lower() for palabra in palabras_clave):
        resultado_plan = planificar_y_ejecutar(mensaje)
        if resultado_plan and not resultado_plan.startswith("Error") and resultado_plan != "No pude descomponer el objetivo en pasos. Intenta ser más específico.":
            return resultado_plan
        # Si falla, cae en el flujo normal

    modelos_gen = st.session_state.get('modelos_generacion', [])
    if not modelos_gen:
        return "Error: No hay modelos de generación disponibles. Verifica tu API key."
    
    if contexto_extra:
        prompt_completo = f"Contexto relevante:\n{contexto_extra}\n\nMensaje actual: {mensaje}"
    else:
        prompt_completo = mensaje
    
    contents = []
    for msg in st.session_state.mensajes[:-1]:
        if msg["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
        else:
            contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt_completo)]))
    
    for modelo_completo in modelos_gen:
        modelo = modelo_completo.replace('models/', '')
        try:
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
            
            if response.function_calls:
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
                
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(name=fc.name, args=fc.args) for fc in response.function_calls]
                    )
                )
                contents.append(
                    types.Content(role="function", parts=function_responses)
                )
                
                final_response = client.models.generate_content(
                    model=modelo,
                    config=types.GenerateContentConfig(
                        system_instruction=NUCLEO_ETERNA,
                        temperature=0.9,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048,
                    ),
                    contents=contents
                )
                return final_response.text
            else:
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
