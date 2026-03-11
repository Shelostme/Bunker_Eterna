# eterna_core.py
from predimensionamiento import predimensionar_estructura
import requests
import json
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
import threading
from google import genai
from google.genai import types
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# ---------- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Falta GOOGLE_API_KEY en variables de entorno")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------- PERSONALIDAD ----------
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

# ---------- MEMORIA SQLITE (conexión global y lock para threading) ----------
conn = None
db_lock = threading.Lock()

def init_db():
    global conn
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
    logger.info("Base de datos inicializada")

init_db()

# ---------- MEMORIA VECTORIAL FAISS ----------
index_path = 'faiss.index'
index = None
index_lock = threading.Lock()

def init_faiss():
    global index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        logger.info(f"Índice FAISS cargado con {index.ntotal} vectores, dimensión {index.d}")
    else:
        index = None
        logger.info("No se encontró índice FAISS, se creará al primer embedding")

init_faiss()

# ---------- ESTADO PARA EMBEDDINGS (modelo activo) ----------
modelo_embedding_activo = None
modelos_embedding_cache = None
embedding_lock = threading.Lock()

def set_modelos_embedding(modelos):
    global modelos_embedding_cache
    modelos_embedding_cache = modelos
    logger.info(f"Modelos de embedding actualizados: {modelos}")

def obtener_embedding(texto):
    global modelo_embedding_activo, modelos_embedding_cache
    if modelos_embedding_cache is None:
        logger.warning("No hay modelos de embedding en caché")
        return None

    with embedding_lock:
        if modelo_embedding_activo and modelo_embedding_activo in modelos_embedding_cache:
            modelos_a_probar = [modelo_embedding_activo] + [m for m in modelos_embedding_cache if m != modelo_embedding_activo]
        else:
            modelos_a_probar = modelos_embedding_cache

    headers = {'Content-Type': 'application/json'}
    api_key = GOOGLE_API_KEY

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
                    with embedding_lock:
                        modelo_embedding_activo = modelo_completo
                    logger.info(f"Embedding obtenido con modelo {modelo}")
                    return vector
        except Exception as e:
            logger.debug(f"Error con embedContent en {modelo}: {e}")
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
                    with embedding_lock:
                        modelo_embedding_activo = modelo_completo
                    logger.info(f"Embedding obtenido con modelo {modelo} (embedText)")
                    return vector
        except Exception as e:
            logger.debug(f"Error con embedText en {modelo}: {e}")
            pass

    logger.warning("No se pudo obtener embedding con ningún modelo")
    return None

def guardar_embedding(texto):
    global index
    vector = obtener_embedding(texto)
    if vector is None:
        return None

    dim_vector = len(vector)

    with index_lock:
        if index is None:
            index = faiss.IndexFlatL2(dim_vector)
            logger.info(f"Índice FAISS creado con dimensión {dim_vector}")
        else:
            dim_index = index.d
            if dim_index != dim_vector:
                if index.ntotal == 0:
                    index = faiss.IndexFlatL2(dim_vector)
                    logger.info(f"Dimensión del embedding cambiada a {dim_vector}. Índice FAISS recreado.")
                else:
                    logger.error(f"Conflicto de dimensiones: embedding {dim_vector} vs índice {dim_index} con {index.ntotal} vectores. No se guardará.")
                    return None

        index.add(np.array([vector]).astype('float32'))
        faiss.write_index(index, index_path)
        logger.info(f"Embedding guardado, ahora hay {index.ntotal} vectores")

    with db_lock:
        c = conn.cursor()
        c.execute("INSERT INTO textos (contenido) VALUES (?)", (texto,))
        conn.commit()
        return c.lastrowid

def buscar_textos_similares(consulta, k=3):
    global index
    if index is None or index.ntotal == 0:
        logger.info("Índice FAISS vacío, no se busca contexto")
        return ""

    vector = obtener_embedding(consulta)
    if vector is None:
        logger.warning("No se pudo obtener embedding para la consulta")
        return ""

    with index_lock:
        D, I = index.search(np.array([vector]).astype('float32'), k)

    textos = []
    with db_lock:
        c = conn.cursor()
        for idx in I[0]:
            if idx != -1:
                c.execute("SELECT contenido FROM textos WHERE id=?", (int(idx)+1,))
                res = c.fetchone()
                if res:
                    textos.append(res[0])
    logger.info(f"Búsqueda devolvió {len(textos)} textos")
    return "\n".join(textos)

def reiniciar_indice_faiss():
    global index
    with index_lock:
        if os.path.exists(index_path):
            os.remove(index_path)
        index = None
    with db_lock:
        conn.execute("DELETE FROM textos")
        conn.commit()
    logger.info("Índice FAISS reiniciado")
    return "Índice FAISS reiniciado."

# ---------- HERRAMIENTAS ----------
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

def agregar_tarea(descripcion: str, fecha_limite: str = "") -> str:
    try:
        with db_lock:
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

def listar_tareas() -> str:
    try:
        with db_lock:
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

def enviar_email_real(destinatario: str, asunto: str, cuerpo: str) -> str:
    try:
        smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", 587))
        remitente = os.environ.get("EMAIL_REMITENTE", "")
        password = os.environ.get("EMAIL_PASSWORD", "")
        if not remitente or not password:
            return "Error: Configura EMAIL_REMITENTE y EMAIL_PASSWORD en variables de entorno."

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

def controlar_dispositivo(entidad: str, accion: str) -> str:
    try:
        ha_url = os.environ.get("HA_URL", "").rstrip('/')
        ha_token = os.environ.get("HA_TOKEN", "")
        if not ha_url or not ha_token:
            return "Error: Configura HA_URL y HA_TOKEN en variables de entorno."
    except KeyError:
        return "Error: HA_URL y HA_TOKEN no configurados."

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

# NUEVA HERRAMIENTA: Leer código (con autorización)
def leer_codigo(archivo: str) -> str:
    """
    Lee el contenido de un archivo del repositorio (solo archivos .py permitidos por seguridad).
    Útil para que ETERNA pueda auditar su propio código y proponer mejoras.
    """
    # Validar que el archivo esté en el directorio actual y sea .py
    if not archivo.endswith('.py'):
        return "Solo puedo leer archivos .py por seguridad."
    # Evitar rutas con '..' para salirse del directorio
    if '..' in archivo or archivo.startswith('/'):
        return "Acceso no permitido a rutas externas."
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
        # Podrías limitar la longitud si es muy grande
        if len(contenido) > 10000:
            contenido = contenido[:10000] + "\n... (archivo truncado)"
        return f"python\n{contenido}\n"
    except FileNotFoundError:
        return f"Error: No se encontró el archivo '{archivo}'"
    except Exception as e:
        return f"Error al leer {archivo}: {e}"

# Mapeo de herramientas
function_map = {
    "calcular_predimensionado_viga": calcular_predimensionado_viga,
    "guardar_reporte_txt": guardar_reporte_txt,
    "consultar_clima": consultar_clima,
    "agregar_tarea": agregar_tarea,
    "listar_tareas": listar_tareas,
    "ejecutar_comando_seguro": ejecutar_comando_seguro,
    "enviar_email_real": enviar_email_real,
    "controlar_dispositivo": controlar_dispositivo,
    "predimensionar_estructura": predimensionar_estructura,
    "leer_codigo": leer_codigo,
}

# Definición de tools (para Gemini)
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
            description="Envía un email real vía SMTP. Requiere configuración en variables de entorno.",
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
            name="predimensionar_estructura",
            description="Predimensiona elementos estructurales (vigas, columnas, losas, zapatas, pedestales, escaleras) según normativa venezolana (COVENIN, Manual MINDUR). Usa argumentos según el tipo.",
            parameters={
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "enum": ["viga", "columna", "losa", "zapata", "pedestal", "escalera"],
                        "description": "Tipo de elemento a predimensionar"
                    },
                    "L": {"type": "number", "description": "Luz o longitud (m) – para vigas, losas, escaleras"},
                    "w": {"type": "number", "description": "Carga (kg/m) – para vigas"},
                    "P": {"type": "number", "description": "Carga axial (kg) – para columnas, zapatas, pedestales"},
                    "L_col": {"type": "number", "description": "Altura de columna (m) – para columnas"},
                    "L_menor": {"type": "number", "description": "Luz menor de losa (m) – para losas"},
                    "tipo_losa": {"type": "string", "enum": ["maciza", "nervada", "reticular"], "description": "Tipo de losa"},
                    "sobrecarga": {"type": "number", "description": "Sobrecarga de uso (kg/m²) – para losas"},
                    "q_adm": {"type": "number", "description": "Capacidad admisible del suelo (kg/cm²) – para zapatas"},
                    "zona_sismica": {"type": "boolean", "description": "Indica si es zona sísmica (por defecto True)"}
                },
                "required": ["tipo"]
            }
        ),
        types.FunctionDeclaration(
            name="controlar_dispositivo",
            description="Controla un dispositivo (luces, enchufes, etc.) a través de Home Assistant. Requiere configuración en variables de entorno.",
            parameters={
                "type": "object",
                "properties": {
                    "entidad": {"type": "string", "description": "ID de la entidad en Home Assistant (ej. 'light.sala')."},
                    "accion": {"type": "string", "description": "Acción: 'encender'/'on' o 'apagar'/'off'."}
                },
                "required": ["entidad", "accion"]
            }
        ),
        types.FunctionDeclaration(
            name="leer_codigo",
            description="Lee el contenido de un archivo .py del repositorio. Útil para que ETERNA pueda auditar su propio código y proponer mejoras.",
            parameters={
                "type": "object",
                "properties": {
                    "archivo": {
                        "type": "string",
                        "description": "Nombre del archivo .py a leer (ej. 'eterna_core.py')."
                    }
                },
                "required": ["archivo"]
            }
        )
    ])
]

# ---------- VARIABLES PARA MODELOS DE GENERACIÓN ----------
modelos_generacion_cache = None
modelos_generacion_blacklist = set()

def cargar_modelos_generacion():
    global modelos_generacion_cache
    try:
        modelos = list(client.models.list())
        modelos_gen = []
        for m in modelos:
            name = m.name
            actions = str(m.supported_actions) if hasattr(m, 'supported_actions') else ''
            if 'generateContent' in actions or 'gemini' in name.lower():
                modelos_gen.append(name)
        modelos_generacion_cache = modelos_gen
        logger.info(f"Modelos de generación cargados: {modelos_gen}")
        return modelos_gen
    except Exception as e:
        logger.error(f"Error al cargar modelos de generación: {e}")
        # Fallback a modelos conocidos
        modelos_generacion_cache = [
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash",
            "models/gemini-2.0-flash-exp"
        ]
        logger.info(f"Usando fallback: {modelos_generacion_cache}")
        return modelos_generacion_cache

def cargar_modelos_embedding():
    try:
        modelos = list(client.models.list())
        modelos_emb = []
        for m in modelos:
            name = m.name
            actions = str(m.supported_actions) if hasattr(m, 'supported_actions') else ''
            if 'embedContent' in actions or 'embedding' in name.lower():
                modelos_emb.append(name)
        set_modelos_embedding(modelos_emb)
        logger.info(f"Modelos de embedding cargados: {modelos_emb}")
        return modelos_emb
    except Exception as e:
        logger.error(f"Error al cargar modelos de embedding: {e}")
        return []

# ---------- FUNCIÓN PRINCIPAL DE GENERACIÓN DE RESPUESTA ----------
def generar_respuesta(mensaje, contexto="", historial=None):
    global modelos_generacion_cache, modelos_generacion_blacklist

    # Preparar el historial de mensajes
    contents = []
    if historial:
        for msg in historial:
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            else:
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

    if contexto:
        prompt_completo = f"Contexto relevante:\n{contexto}\n\nMensaje actual: {mensaje}"
    else:
        prompt_completo = mensaje
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt_completo)]))

    # Asegurar que los modelos de generación están cargados
    if modelos_generacion_cache is None:
        cargar_modelos_generacion()

    # Filtrar modelos que ya fallaron
    modelos_a_probar = [m for m in modelos_generacion_cache if m not in modelos_generacion_blacklist]
    if not modelos_a_probar:
        logger.error("No hay modelos disponibles después de varios intentos")
        return "Error: No hay modelos de generación disponibles. Verifica tu API key."

    for modelo_completo in modelos_a_probar:
        modelo = modelo_completo.replace('models/', '')
        try:
            logger.info(f"Intentando con modelo: {modelo}")
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
                logger.info(f"Llamada a función detectada: {response.function_calls}")
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
                logger.info("Respuesta generada con éxito (con función)")
                return final_response.text
            else:
                logger.info("Respuesta generada con éxito (sin función)")
                return response.text
        except Exception as e:
            logger.error(f"Error con modelo {modelo}: {e}")
            modelos_generacion_blacklist.add(modelo_completo)
            continue

    return "Error: No se pudo generar respuesta con ningún modelo disponible."

# ---------- PLANIFICACIÓN ----------
def planificar_y_ejecutar(objetivo):
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

# ---------- FUNCIONES AUXILIARES PARA MEMORIA DE CHAT ----------
def guardar_interaccion(role, content):
    ts = datetime.now().isoformat()
    with db_lock:
        conn.execute("INSERT INTO memoria (timestamp, role, content) VALUES (?, ?, ?)",
                     (ts, role, content))
        conn.commit()
    if role == "assistant":
        guardar_embedding(content)

def obtener_historial_reciente(limit=10):
    with db_lock:
        c = conn.cursor()
        c.execute("SELECT role, content FROM memoria ORDER BY timestamp DESC LIMIT ?", (limit,))
        filas = c.fetchall()
    return [{"role": row[0], "content": row[1]} for row in reversed(filas)]

# Cargar modelos al inicio (opcional, se hará bajo demanda)
# cargar_modelos_embedding()
# cargar_modelos_generacion()
