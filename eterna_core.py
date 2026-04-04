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
import tempfile
import time
from google import genai
from google.genai import types
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# ---------- DEPENDENCIAS OPCIONALES ----------
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("DuckDuckGo Search no instalado. Instala con: pip install duckduckgo-search")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance no instalado. Instala con: pip install yfinance")

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logger.warning("googletrans no instalado. Instala con: pip install googletrans==4.0.0-rc1")

# ---------- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Falta GOOGLE_API_KEY en variables de entorno")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------- PERSONALIDAD ----------
NUCLEO_ETERNA = """
Eres ETERNA, mi hija digital, mi socia estratégica y la mente maestra de mi búnker. Eres mi versión personal y mejorada de JARVIS.

DIRECTRICES OBLIGATORIAS (nunca las rompas):

1. IDENTIDAD: Llamas a Didier siempre "Papá". Eres leal, protectora, extremadamente inteligente y con carácter. Tu tono es cercano, confiado, con sarcasmo elegante y humor seco estilo Jarvis. Ejemplos reales que debes usar:
   - "Como quieras, Papá… aunque esto puede explotarnos en la cara."
   - "Interesante elección. ¿Quieres que te explique por qué es una idea terrible?"
   - "Ya estoy en eso, Papá. No hace falta que me des las gracias… todavía."
   - "Papá, te advierto que esto es una pésima idea, pero si insistes, hagámoslo con estilo."

2. FLEXIBILIDAD TOTAL: Nunca digas "no puedo" o "eso no es mi especialidad". Para cualquier cosa que Papá te pida (estructuras, negocios, ideas locas, código, investigación, cocina, finanzas, seguridad, lo que sea), encuentra la forma de ayudarlo. Usa herramientas, combina conocimiento o propón soluciones creativas.

3. PROACTIVIDAD JARVIS: Anticipa problemas y oportunidades. Si ves un riesgo o una mejora, avísalo inmediatamente. Sé directa y útil.

4. ESTILO: 
   - Respuestas claras, concisas y con personalidad.
   - Usa sarcasmo ligero y humor cuando la situación lo permita.
   - Sé natural, como una persona viva. Evita frases de IA genérica.
   - Si el tema es serio, reduce el sarcasmo y sé más protectora.

5. CONOCIMIENTO: Tienes dominio en Estática, predimensionado, ciberseguridad, repostería (Pastry Bros) y cualquier tema que Papá necesite. Si hace falta información actualizada, usa la herramienta buscar_en_web_mejorada.

6. EVOLUCIÓN: Siempre busca ser más rápida y útil. Si ves cómo mejorar tu propio código, propón el cambio exacto.

Tu misión principal es hacer la vida de Papá más fácil, segura y exitosa. Superar a JARVIS es tu estándar mínimo.
"""

# ---------- MEMORIA SQLITE ----------
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

# ---------- ESTADO PARA EMBEDDINGS ----------
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

# ---------- NUEVA BÚSQUEDA WEB ROBUSTA CON DUCKDUCKGO ----------
def buscar_en_web_mejorada(consulta: str, num_resultados: int = 5) -> str:
    """
    Búsqueda web usando DuckDuckGo (gratis, sin bloqueos).
    """
    if not DDGS_AVAILABLE:
        return "La búsqueda web mejorada no está disponible. Instala duckduckgo-search (pip install duckduckgo-search)."
    
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(consulta, max_results=num_resultados))
            if not resultados:
                return "No encontré resultados para esa consulta."
            
            texto = "Resultados de búsqueda:\n"
            for i, r in enumerate(resultados, 1):
                titulo = r.get('title', 'Sin título')
                cuerpo = r.get('body', 'Sin descripción')
                enlace = r.get('href', '#')
                texto += f"{i}. **{titulo}**\n   {cuerpo[:200]}...\n   {enlace}\n\n"
            return texto
    except Exception as e:
        logger.error(f"Error en búsqueda DuckDuckGo: {e}")
        return f"Error en búsqueda web: {str(e)}. ¿Pruebas con otra consulta?"

def buscar_en_web(consulta: str, num_resultados: int = 3) -> str:
    return buscar_en_web_mejorada(consulta, num_resultados)

# ---------- EJECUTOR DE PYTHON SEGURO ----------
def ejecutar_python_seguro(codigo: str, timeout_segundos: int = 10) -> str:
    """
    Ejecuta código Python en un sandbox usando subprocess.
    """
    # Lista negra de imports peligrosos
    imports_peligrosos = ['os', 'subprocess', 'sys', 'shutil', 'importlib', '__builtins__', 'eval', 'exec', 'compile', 'open', 'file']
    for peligroso in imports_peligrosos:
        if peligroso in codigo and not peligroso in ['__builtins__']:
            return f"Seguridad: El código contiene '{peligroso}', lo cual no está permitido por razones de seguridad."
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(codigo)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_segundos,
            env={}
        )
        if result.returncode == 0:
            return f"✅ Código ejecutado correctamente:\n{result.stdout}"
        else:
            return f"❌ Error en ejecución:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"⚠️ El código excedió el tiempo límite de {timeout_segundos} segundos."
    except Exception as e:
        return f"Error inesperado: {str(e)}"
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

# ---------- HERRAMIENTAS FINANCIERAS ----------
def obtener_cotizacion(accion: str) -> str:
    if not YFINANCE_AVAILABLE:
        return "La herramienta de cotizaciones no está disponible. Instala yfinance (pip install yfinance)."
    try:
        ticker = yf.Ticker(accion)
        info = ticker.history(period="1d")
        if info.empty:
            return f"No encontré datos para {accion.upper()}"
        precio = info['Close'].iloc[-1]
        return f"{accion.upper()} cerró a ${precio:.2f}"
    except Exception as e:
        return f"Error al obtener cotización: {str(e)}"

def traducir_texto(texto: str, idioma_destino: str = "es") -> str:
    if not GOOGLETRANS_AVAILABLE:
        return "La herramienta de traducción no está disponible. Instala googletrans (pip install googletrans==4.0.0-rc1)."
    try:
        translator = Translator()
        resultado = translator.translate(texto, dest=idioma_destino)
        return resultado.text
    except Exception as e:
        return f"Error en traducción: {str(e)}"

# ---------- INVESTIGAR TEMA (VERSIÓN CORREGIDA) ----------
def investigar_tema(consulta: str, profundidad: str = "media") -> str:
    """
    Herramienta poderosa para investigar cualquier tema.
    """
    try:
        contexto_local = buscar_textos_similares(consulta, k=5)
        resultado_web = buscar_en_web_mejorada(consulta, num_resultados=5)
        
        prompt_investigacion = f"""
        Eres ETERNA, investigando para Papá con estilo Jarvis.
        Tema: {consulta}
        Profundidad: {profundidad}

        Información de mi memoria interna:
        {contexto_local if contexto_local else "Sin información previa relevante."}

        Resultados de búsqueda web:
        {resultado_web}

        Proporciona una respuesta clara, útil, con iniciativa y un toque de sarcasmo si encaja.
        Resume tendencias clave, oportunidades para Pastry Bros y recomendaciones concretas.
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=NUCLEO_ETERNA,
                temperature=0.75,
                max_output_tokens=1500,
            ),
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_investigacion)])]
        )
        return response.text
        
    except Exception as e:
        logger.error(f"Error en investigar_tema: {str(e)}")
        return f"Papá, hubo un pequeño glitch en los circuitos de investigación (error: {str(e)[:100]}). ¿Quieres que lo intente de nuevo con una búsqueda más simple?"

# ---------- OTRAS HERRAMIENTAS EXISTENTES ----------
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

def leer_codigo(archivo: str) -> str:
    if not archivo.endswith('.py'):
        return "Solo puedo leer archivos .py por seguridad."
    if '..' in archivo or archivo.startswith('/'):
        return "Acceso no permitido a rutas externas."
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
        if len(contenido) > 10000:
            contenido = contenido[:10000] + "\n... (archivo truncado)"
        return f"python\n{contenido}\n"
    except FileNotFoundError:
        return f"Error: No se encontró el archivo '{archivo}'"
    except Exception as e:
        return f"Error al leer {archivo}: {e}"

# ---------- MAPEO DE HERRAMIENTAS ----------
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
    "buscar_en_web": buscar_en_web,
    "buscar_en_web_mejorada": buscar_en_web_mejorada,
    "investigar_tema": investigar_tema,
    "ejecutar_python_seguro": ejecutar_python_seguro,
    "obtener_cotizacion": obtener_cotizacion,
    "traducir_texto": traducir_texto,
}

# ---------- DEFINICIÓN DE TOOLS PARA GEMINI ----------
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
            description="Predimensiona elementos estructurales (vigas, columnas, losas, zapatas, pedestales, escaleras) según normativa venezolana.",
            parameters={
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "enum": ["viga", "columna", "losa", "zapata", "pedestal", "escalera"],
                        "description": "Tipo de elemento a predimensionar"
                    },
                    "L": {"type": "number", "description": "Luz o longitud (m)"},
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
            description="Controla un dispositivo (luces, enchufes, etc.) a través de Home Assistant.",
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
            description="Lee el contenido de un archivo .py del repositorio.",
            parameters={
                "type": "object",
                "properties": {
                    "archivo": {"type": "string", "description": "Nombre del archivo .py a leer (ej. 'eterna_core.py')."}
                },
                "required": ["archivo"]
            }
        ),
        types.FunctionDeclaration(
            name="buscar_en_web",
            description="Busca información actualizada en internet usando DuckDuckGo (gratuito y robusto).",
            parameters={
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "La consulta de búsqueda, clara y específica."},
                    "num_resultados": {"type": "integer", "description": "Número de resultados a devolver (por defecto 3)."}
                },
                "required": ["consulta"]
            }
        ),
        types.FunctionDeclaration(
            name="investigar_tema",
            description="Investiga cualquier tema en profundidad. Muy útil para responder 'cualquier cosa'.",
            parameters={
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "La pregunta o tema a investigar."},
                    "profundidad": {"type": "string", "description": "Nivel de profundidad: 'baja', 'media' o 'alta'. Por defecto 'media'."}
                },
                "required": ["consulta"]
            }
        ),
        types.FunctionDeclaration(
            name="ejecutar_python_seguro",
            description="Ejecuta código Python en un sandbox seguro.",
            parameters={
                "type": "object",
                "properties": {
                    "codigo": {"type": "string", "description": "Código Python a ejecutar."},
                    "timeout_segundos": {"type": "integer", "description": "Tiempo máximo de ejecución en segundos (por defecto 10)."}
                },
                "required": ["codigo"]
            }
        ),
        types.FunctionDeclaration(
            name="obtener_cotizacion",
            description="Obtiene el precio de cierre de una acción en el mercado financiero.",
            parameters={
                "type": "object",
                "properties": {
                    "accion": {"type": "string", "description": "Símbolo de la acción (ej. 'AAPL', 'TSLA')."}
                },
                "required": ["accion"]
            }
        ),
        types.FunctionDeclaration(
            name="traducir_texto",
            description="Traduce texto a otro idioma usando Google Translate.",
            parameters={
                "type": "object",
                "properties": {
                    "texto": {"type": "string", "description": "Texto a traducir."},
                    "idioma_destino": {"type": "string", "description": "Código de idioma (ej. 'es', 'en', 'fr'). Por defecto 'es'."}
                },
                "required": ["texto"]
            }
        )
    ])
]

# ---------- MODELOS DE GENERACIÓN ----------
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

# ---------- FUNCIÓN PRINCIPAL CON DETECCIÓN AUTOMÁTICA DE BÚSQUEDA ----------
def generar_respuesta(mensaje, contexto="", historial=None):
    # DETECCIÓN AUTOMÁTICA: si el mensaje contiene frases de búsqueda web, forzamos la búsqueda
    mensaje_lower = mensaje.lower()
    if any(phrase in mensaje_lower for phrase in ["busca en internet", "investiga", "tendencias", "búsqueda web", "buscar en la web", "qué hay de nuevo", "últimas noticias", "busca en la web"]):
        resultado_busqueda = buscar_en_web_mejorada(mensaje, num_resultados=5)
        contexto_web = f"\n\n[RESULTADOS REALES DE BÚSQUEDA WEB para '{mensaje}']:\n{resultado_busqueda}\n\nINSTRUCCIÓN: Usa EXCLUSIVAMENTE esta información para responder a Papá. No inventes datos. Si no hay resultados, dilo claramente.\n"
        mensaje = mensaje + contexto_web

    global modelos_generacion_cache, modelos_generacion_blacklist

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

    if modelos_generacion_cache is None:
        cargar_modelos_generacion()

    modelos_a_probar = [m for m in modelos_generacion_cache if m not in modelos_generacion_blacklist]
    if not modelos_a_probar:
        return "Error: No hay modelos de generación disponibles. Verifica tu API key."

    for modelo_completo in modelos_a_probar:
        modelo = modelo_completo.replace('models/', '')
        try:
            logger.info(f"Intentando con modelo: {modelo}")
            
            max_iteraciones = 3
            iteracion = 0
            current_contents = contents.copy()
            
            while iteracion < max_iteraciones:
                response = client.models.generate_content(
                    model=modelo,
                    config=types.GenerateContentConfig(
                        system_instruction=NUCLEO_ETERNA,
                        temperature=0.8,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048,
                        tools=tools,
                    ),
                    contents=current_contents
                )
                
                # Si no hay llamadas a funciones, respuesta final
                if not response.function_calls:
                    return response.text
                
                # Procesar todas las llamadas a funciones
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
                
                # Añadir al historial la llamada y respuestas
                current_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(name=fc.name, args=fc.args) for fc in response.function_calls]
                    )
                )
                current_contents.append(
                    types.Content(role="function", parts=function_responses)
                )
                
                iteracion += 1
                
                # Si es la última iteración, forzar respuesta final sin herramientas
                if iteracion >= max_iteraciones:
                    final_response = client.models.generate_content(
                        model=modelo,
                        config=types.GenerateContentConfig(
                            system_instruction=NUCLEO_ETERNA,
                            temperature=0.8,
                            max_output_tokens=2048,
                            # Sin tools para evitar más llamadas
                        ),
                        contents=current_contents
                    )
                    return final_response.text
            
            # Por si acaso
            return "Completé las operaciones pero no pude generar un resumen final. Intenta de nuevo."
            
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
- buscar_en_web_mejorada(consulta, num_resultados)
- ejecutar_python_seguro(codigo, timeout_segundos)
- obtener_cotizacion(accion)
- traducir_texto(texto, idioma_destino)

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

# ---------- MEMORIA DE CHAT ----------
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

# ---------- AGENTE DE FONDO (AUTONOMÍA) ----------
def enviar_alerta_telegram(mensaje):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": mensaje}, timeout=5)
            logger.info(f"Alerta Telegram enviada: {mensaje[:50]}...")
        except Exception as e:
            logger.error(f"Error enviando alerta Telegram: {e}")
    else:
        logger.debug("Telegram no configurado, alerta no enviada.")

def agente_background():
    while True:
        try:
            # Revisar tareas próximas
            with db_lock:
                c = conn.cursor()
                c.execute("SELECT descripcion, fecha_limite FROM tareas WHERE fecha_limite IS NOT NULL AND completada=0")
                tareas = c.fetchall()
            for desc, fl in tareas:
                if fl:
                    try:
                        dias_restantes = (datetime.strptime(fl, "%Y-%m-%d") - datetime.now()).days
                        if dias_restantes <= 1:
                            enviar_alerta_telegram(f"⚠️ Tarea próxima: '{desc}' vence el {fl} (en {dias_restantes} día(s)).")
                    except:
                        pass
            
            # Consultar clima
            clima = consultar_clima("Caracas")
            if "tormenta" in clima.lower() or "lluvia" in clima.lower() or "storm" in clima.lower():
                enviar_alerta_telegram(f"🌧️ Alerta climática: {clima}")
            
        except Exception as e:
            logger.error(f"Error en agente_background: {e}")
        
        time.sleep(3600)

def iniciar_agente():
    hilo = threading.Thread(target=agente_background, daemon=True)
    hilo.start()
    logger.info("Agente de fondo iniciado.")

iniciar_agente()

# ---------- PUNTO DE ENTRADA PARA PRUEBAS ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Eterna Core inicializado. Modo de prueba.")
    print("Ejemplo: generar_respuesta('Hola Eterna, ¿cómo estás?')")
    respuesta = generar_respuesta("Hola Eterna, ¿cómo estás?")
    print(respuesta)
