# eterna_core.py
import matplotlib.pyplot as plt
import numpy as np
from predimensionamiento import predimensionar_estructura
import requests
import json
import sqlite3
from datetime import datetime
import os
import faiss
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
from urllib.parse import quote_plus
import asyncio
import nest_asyncio
import base64

# Configurar logging
logger = logging.getLogger(__name__)

# Aplicar nest_asyncio para permitir bucles anidados (útil en Streamlit/Codespaces)
nest_asyncio.apply()

# ---------- DEPENDENCIAS OPCIONALES (navegación web y visión) ----------
try:
    from browser_use import Agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    logger.warning("browser-use no instalado. Instala con: pip install browser-use langchain-google-genai playwright")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("DuckDuckGo Search no instalado. Se usará método alternativo.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance no instalado. Instala con: pip install yfinance")

# Traducción con deep-translator (compatible con httpcore moderno)
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    logger.warning("deep-translator no instalado. Instala con: pip install deep-translator")

# ---------- CONFIGURACIÓN DESDE VARIABLES DE ENTORNO ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Falta GOOGLE_API_KEY en variables de entorno")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ========== NUEVOS MÓDULOS INSPIRADOS EN RABBIT ==========

# 1. PLANIFICADOR (Planner)
class Planner:
    """Descompone objetivos en pasos ejecutables por Eterna."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.tools_list = []
    
    def set_tools(self, tools_list):
        self.tools_list = tools_list
    
    def planify(self, objective: str, max_steps: int = 5) -> list:
        if not self.tools_list:
            return []
        prompt = f"""
        Eres el planificador de Eterna. Tu tarea es descomponer este objetivo en pasos usando SOLO las herramientas disponibles.
        Herramientas: {', '.join(self.tools_list)}
        Objetivo: {objective}
        Máximo {max_steps} pasos.
        Responde EXCLUSIVAMENTE con un JSON array de pasos, cada paso con 'tool' y 'args'.
        Ejemplo: [{{"tool": "buscar_en_web", "args": {{"consulta": "IA"}}}}, {{"tool": "guardar_reporte_txt", "args": {{"titulo": "resultado", "contenido": "..."}}}}]
        """
        try:
            response = self.llm.generate_content(
                model="gemini-1.5-flash",
                config=types.GenerateContentConfig(temperature=0.2),
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
            )
            text = response.text.strip()
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            else:
                return []
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return []

# 2. MEMORIA MEJORADA (EnhancedMemory)
class EnhancedMemory:
    """Guarda y recupera resultados de tareas, no solo conversaciones."""
    
    def __init__(self, db_path='eterna_memory.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS task_results 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             task TEXT,
                             result TEXT,
                             timestamp TEXT)''')
        self.conn.commit()
    
    def save_task_result(self, task: str, result: str):
        ts = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO task_results (task, result, timestamp) VALUES (?, ?, ?)",
                       (task, result, ts))
        self.conn.commit()
        return cursor.lastrowid
    
    def find_similar_tasks(self, query: str, limit: int = 3) -> list:
        cursor = self.conn.cursor()
        palabras = query.lower().split()
        condiciones = " OR ".join([f"task LIKE ?" for _ in palabras])
        params = [f"%{p}%" for p in palabras]
        cursor.execute(f"SELECT task, result FROM task_results WHERE {condiciones} ORDER BY timestamp DESC LIMIT ?", params + [limit])
        return cursor.fetchall()

# 3. CONTROLADOR DE NAVEGACIÓN (BrowserController)
class BrowserController:
    """Controla el navegador con Playwright de forma fiable."""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser = None
        self.page = None
        self.playwright = None
    
    async def start(self):
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()
        return self.page
    
    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def navigate(self, url: str, timeout: int = 30000):
        await self.page.goto(url, timeout=timeout)
        return await self.page.title()
    
    async def fill_and_submit(self, selector: str, text: str):
        await self.page.fill(selector, text)
        await self.page.press(selector, 'Enter')
    
    async def get_text(self, selector: str) -> str:
        return await self.page.inner_text(selector)

# 4. REGISTRO DE HERRAMIENTAS (ToolRegistry)
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func):
        self.tools[name] = func
    
    def register_batch(self, tools_dict):
        self.tools.update(tools_dict)
    
    def execute(self, tool_name: str, **kwargs):
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        else:
            return f"Herramienta '{tool_name}' no registrada."
    
    def list_tools(self):
        return list(self.tools.keys())

# ========== FIN DE NUEVOS MÓDULOS ==========

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

5. CONOCIMIENTO: Tienes dominio en Estática, predimensionado, ciberseguridad, repostería (Pastry Bros) y cualquier tema que Papá necesite. Si hace falta información actualizada, usa la herramienta buscar_en_web_mejorada o ejecutar_tarea_web.

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

# ---------- HERRAMIENTAS EXISTENTES ----------
def calcular_predimensionado_viga(longitud: float, carga: float) -> str:
    try:
        if carga < 1000:
            h = longitud / 10
            recomendacion = "carga liviana"
        else:
            h = longitud / 8
            recomendacion = "carga pesada"
        return f"Para una viga de longitud {longitud:.2f} m y carga {carga:.2f} kg/m ({recomendacion}), la altura recomendada es {h:.2f} m."
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

# ---------- BÚSQUEDA WEB MULTI-MÉTODO ----------
def buscar_en_web_mejorada(consulta: str, num_resultados: int = 5) -> str:
    consulta = consulta.strip()
    if not consulta:
        return "No hay consulta para buscar."

    if DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                resultados = list(ddgs.text(consulta, region='wt-wt', safesearch='moderate', max_results=num_resultados))
                if resultados:
                    texto = f"🔍 Resultados de búsqueda para '{consulta}':\n\n"
                    for i, r in enumerate(resultados, 1):
                        titulo = r.get('title', 'Sin título')
                        cuerpo = r.get('body', 'Sin descripción')
                        enlace = r.get('href', '#')
                        texto += f"{i}. **{titulo}**\n   {cuerpo}\n   Fuente: {enlace}\n\n"
                    return texto
        except Exception as e:
            logger.warning(f"DuckDuckGo-search falló: {e}")

    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(consulta)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            resultados = []
            abstract = data.get('AbstractText', '')
            if abstract:
                resultados.append(("Resumen", abstract, data.get('AbstractURL', '')))
            answer = data.get('Answer', '')
            if answer:
                resultados.append(("Respuesta directa", answer, ''))
            for topic in data.get('RelatedTopics', []):
                if isinstance(topic, dict):
                    text = topic.get('Text', '')
                    url_topic = topic.get('FirstURL', '')
                    if text:
                        resultados.append(("Tema relacionado", text, url_topic))
                if len(resultados) >= num_resultados:
                    break
            if resultados:
                texto = f"🔍 Resultados de búsqueda para '{consulta}':\n\n"
                for i, (tipo, contenido, enlace) in enumerate(resultados, 1):
                    texto += f"{i}. **{tipo}**: {contenido}\n"
                    if enlace:
                        texto += f"   Fuente: {enlace}\n"
                    texto += "\n"
                return texto
    except Exception as e:
        logger.warning(f"API DuckDuckGo falló: {e}")

    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(consulta)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            import html
            text = response.text
            enlaces = re.findall(r'<a href="(https?://[^"]+)"[^>]*>([^<]+)</a>', text)
            descripciones = re.findall(r'<td class="result-snippet">([^<]+)</td>', text)
            resultados = []
            for i, (url_enlace, titulo) in enumerate(enlaces[:num_resultados]):
                desc = descripciones[i] if i < len(descripciones) else "Sin descripción"
                resultados.append((titulo, html.unescape(desc), url_enlace))
            if resultados:
                texto = f"🔍 Resultados de búsqueda para '{consulta}':\n\n"
                for i, (titulo, cuerpo, enlace) in enumerate(resultados, 1):
                    texto += f"{i}. **{titulo}**\n   {cuerpo}\n   Fuente: {enlace}\n\n"
                return texto
    except Exception as e:
        logger.error(f"Scraping DuckDuckGo Lite falló: {e}")

    return f"No se pudo obtener resultados para '{consulta}'. Intenta con una consulta más específica."

def buscar_en_web(consulta: str, num_resultados: int = 3) -> str:
    return buscar_en_web_mejorada(consulta, num_resultados)

# ---------- EJECUTAR TAREA WEB (navegación autónoma) ----------
async def _ejecutar_tarea_web_async(tarea: str, url_inicial: str = None) -> str:
    if not BROWSER_USE_AVAILABLE:
        return "⚠️ La función 'ejecutar_tarea_web' no está disponible porque falta instalar browser-use y langchain-google-genai. Ejecuta: pip install browser-use langchain-google-genai playwright && playwright install"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        )
        agent = Agent(
            task=tarea,
            llm=llm,
            use_vision=True,
        )
        result = await agent.run()
        final_answer = result.final_result() if hasattr(result, 'final_result') else str(result)
        return f"✅ Tarea web completada:\n{final_answer}"
    except Exception as e:
        logger.error(f"Error en ejecutar_tarea_web: {e}")
        return f"❌ Error durante la ejecución de la tarea web: {str(e)}"

def ejecutar_tarea_web(tarea: str, url_inicial: str = "") -> str:
    if url_inicial and url_inicial.strip():
        tarea = f"Ve a {url_inicial}. Luego, {tarea}"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        resultado = loop.run_until_complete(_ejecutar_tarea_web_async(tarea, url_inicial))
        return resultado
    finally:
        loop.close()

# ---------- EJECUTOR DE PYTHON SEGURO ----------
def ejecutar_python_seguro(codigo: str, timeout_segundos: int = 10) -> str:
    imports_peligrosos = ['os', 'subprocess', 'sys', 'shutil', 'importlib', '__builtins__', 'eval', 'exec', 'compile', 'open', 'file']
    for peligroso in imports_peligrosos:
        if peligroso in codigo:
            return f"Seguridad: El código contiene '{peligroso}', no permitido."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(codigo)
        temp_path = f.name
    try:
        result = subprocess.run(['python3', temp_path], capture_output=True, text=True, timeout=timeout_segundos, env={})
        if result.returncode == 0:
            return f"✅ Código ejecutado:\n{result.stdout}"
        else:
            return f"❌ Error:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"⚠️ Tiempo límite excedido ({timeout_segundos}s)."
    except Exception as e:
        return f"Error inesperado: {str(e)}"
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

# ---------- FINANZAS ----------
def obtener_cotizacion(accion: str) -> str:
    if not YFINANCE_AVAILABLE:
        return "Herramienta de cotizaciones no disponible. Instala yfinance."
    try:
        ticker = yf.Ticker(accion)
        info = ticker.history(period="1d")
        if info.empty:
            return f"No encontré datos para {accion.upper()}"
        precio = info['Close'].iloc[-1]
        return f"{accion.upper()} cerró a ${precio:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- TRADUCCIÓN (con deep-translator) ----------
def traducir_texto(texto: str, idioma_destino: str = "es") -> str:
    if not DEEP_TRANSLATOR_AVAILABLE:
        return "Traducción no disponible. Instala deep-translator."
    try:
        translator = GoogleTranslator(target=idioma_destino)
        resultado = translator.translate(texto)
        return resultado
    except Exception as e:
        return f"Error en traducción: {str(e)}"

# ---------- INVESTIGAR TEMA ----------
def investigar_tema(consulta: str, profundidad: str = "media") -> str:
    try:
        contexto_local = buscar_textos_similares(consulta, k=5)
        resultado_web = buscar_en_web_mejorada(consulta, num_resultados=5)
        prompt = f"""
        Tema: {consulta}
        Profundidad: {profundidad}
        Memoria interna: {contexto_local if contexto_local else "Sin info previa"}
        Búsqueda web: {resultado_web}
        Da una respuesta clara, útil y con estilo Jarvis.
        """
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(system_instruction=NUCLEO_ETERNA, temperature=0.75, max_output_tokens=1500),
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )
        return response.text
    except Exception as e:
        logger.error(f"Error en investigar_tema: {e}")
        return f"Error en investigación: {str(e)[:100]}"

# ---------- OTRAS HERRAMIENTAS ----------
def ejecutar_comando_seguro(comando: str) -> str:
    comandos_permitidos = {
        'date': ['date'], 'uptime': ['uptime'], 'df': ['df', '-h'], 'free': ['free', '-h'],
        'ls': ['ls', '-la'], 'whoami': ['whoami'], 'uname': ['uname', '-a'],
    }
    if comando not in comandos_permitidos:
        return f"Comando '{comando}' no permitido. Válidos: {', '.join(comandos_permitidos.keys())}"
    try:
        resultado = subprocess.run(comandos_permitidos[comando], capture_output=True, text=True, timeout=5)
        return resultado.stdout if resultado.returncode == 0 else f"Error: {resultado.stderr}"
    except Exception as e:
        return f"Excepción: {str(e)}"

def enviar_email_real(destinatario: str, asunto: str, cuerpo: str) -> str:
    try:
        smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", 587))
        remitente = os.environ.get("EMAIL_REMITENTE", "")
        password = os.environ.get("EMAIL_PASSWORD", "")
        if not remitente or not password:
            return "Error: Configura EMAIL_REMITENTE y EMAIL_PASSWORD."
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
        return f"Email enviado a {destinatario}."
    except Exception as e:
        return f"Error: {str(e)}"

def controlar_dispositivo(entidad: str, accion: str) -> str:
    ha_url = os.environ.get("HA_URL", "").rstrip('/')
    ha_token = os.environ.get("HA_TOKEN", "")
    if not ha_url or not ha_token:
        return "Error: HA_URL y HA_TOKEN no configurados."
    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
    if accion.lower() in ["encender", "on"]:
        service = "turn_on"
    elif accion.lower() in ["apagar", "off"]:
        service = "turn_off"
    else:
        return f"Acción '{accion}' no soportada."
    try:
        domain = entidad.split('.')[0]
    except:
        return "Formato inválido. Ej: 'light.sala'."
    url = f"{ha_url}/api/services/{domain}/{service}"
    payload = {"entity_id": entidad}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            return f"Comando '{accion}' enviado a {entidad}."
        else:
            return f"Error HA: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error conectando con HA: {str(e)}"

def leer_codigo(archivo: str) -> str:
    if not archivo.endswith('.py'):
        return "Solo archivos .py por seguridad."
    if '..' in archivo or archivo.startswith('/'):
        return "Acceso no permitido."
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
        if len(contenido) > 10000:
            contenido = contenido[:10000] + "\n... (truncado)"
        return f"python\n{contenido}\n"
    except FileNotFoundError:
        return f"Archivo '{archivo}' no encontrado."
    except Exception as e:
        return f"Error: {e}"

# ========== FUNCIONES DE VISIÓN Y DIAGRAMAS ==========

def analizar_imagen(imagen_path: str, pregunta: str) -> str:
    """Analiza una imagen local con Gemini (visión)."""
    if not os.path.exists(imagen_path):
        return f"❌ No se encontró la imagen: {imagen_path}"
    try:
        with open(imagen_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = "image/png" if imagen_path.endswith('.png') else "image/jpeg"
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(temperature=0.5),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=pregunta),
                        types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_b64))
                    ]
                )
            ]
        )
        return response.text
    except Exception as e:
        logger.error(f"Error en analizar_imagen: {e}")
        return f"❌ Error analizando imagen: {str(e)}"

def generar_diagrama_viga(longitud: float, cargas: list = None, tipo: str = "ambos") -> str:
    """Genera diagramas de cortante y momento con matplotlib."""
    try:
        x = np.linspace(0, longitud, 500)
        V = np.zeros_like(x)
        M = np.zeros_like(x)
        # Ejemplo didáctico (carga distribuida + puntual)
        w = 20
        P = 30
        a = longitud / 2
        R1 = (w * longitud**2 / 2 + P * a) / longitud
        R2 = w * longitud + P - R1
        for i, xi in enumerate(x):
            V[i] = R1 - w * xi
            M[i] = R1 * xi - w * xi**2 / 2
            if xi >= a:
                V[i] -= P
                M[i] -= P * (xi - a)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        if tipo in ["cortante", "ambos"]:
            ax1.plot(x, V, 'b-', linewidth=2)
            ax1.fill_between(x, 0, V, alpha=0.2, color='blue')
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.set_ylabel('Cortante V (kN)')
            ax1.grid(True)
            ax1.set_title('Diagrama de Cortante')
            ax1.set_xlim(0, longitud)
        if tipo in ["momento", "ambos"]:
            ax2.plot(x, M, 'r-', linewidth=2)
            ax2.fill_between(x, 0, M, alpha=0.2, color='red')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_ylabel('Momento M (kN·m)')
            ax2.set_xlabel('Posición x (m)')
            ax2.grid(True)
            ax2.set_title('Diagrama de Momento Flector')
            ax2.set_xlim(0, longitud)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagrama_viga_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        return f"✅ Diagrama generado y guardado como: {filename}"
    except Exception as e:
        logger.error(f"Error en generar_diagrama_viga: {e}")
        return f"❌ Error generando diagrama: {str(e)}"

# ========== MAPEO DE HERRAMIENTAS ==========
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
    "ejecutar_tarea_web": ejecutar_tarea_web,
    "analizar_imagen": analizar_imagen,
    "generar_diagrama_viga": generar_diagrama_viga,
}

# ---------- TOOLS (declaraciones para Gemini) ----------
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(name="calcular_predimensionado_viga", description="Calcula altura de viga.", parameters={"type": "object", "properties": {"longitud": {"type": "number"}, "carga": {"type": "number"}}, "required": ["longitud", "carga"]}),
        types.FunctionDeclaration(name="guardar_reporte_txt", description="Guarda reporte.", parameters={"type": "object", "properties": {"titulo": {"type": "string"}, "contenido": {"type": "string"}}, "required": ["titulo", "contenido"]}),
        types.FunctionDeclaration(name="consultar_clima", description="Clima actual.", parameters={"type": "object", "properties": {"ciudad": {"type": "string"}}, "required": ["ciudad"]}),
        types.FunctionDeclaration(name="agregar_tarea", description="Agrega tarea.", parameters={"type": "object", "properties": {"descripcion": {"type": "string"}, "fecha_limite": {"type": "string"}}, "required": ["descripcion"]}),
        types.FunctionDeclaration(name="listar_tareas", description="Lista tareas.", parameters={"type": "object", "properties": {}}),
        types.FunctionDeclaration(name="ejecutar_comando_seguro", description="Ejecuta comando seguro.", parameters={"type": "object", "properties": {"comando": {"type": "string"}}, "required": ["comando"]}),
        types.FunctionDeclaration(name="enviar_email_real", description="Envía email.", parameters={"type": "object", "properties": {"destinatario": {"type": "string"}, "asunto": {"type": "string"}, "cuerpo": {"type": "string"}}, "required": ["destinatario", "asunto", "cuerpo"]}),
        types.FunctionDeclaration(name="predimensionar_estructura", description="Predimensiona estructuras.", parameters={"type": "object", "properties": {"tipo": {"type": "string"}}, "required": ["tipo"]}),
        types.FunctionDeclaration(name="controlar_dispositivo", description="Controla dispositivo HA.", parameters={"type": "object", "properties": {"entidad": {"type": "string"}, "accion": {"type": "string"}}, "required": ["entidad", "accion"]}),
        types.FunctionDeclaration(name="leer_codigo", description="Lee archivo .py.", parameters={"type": "object", "properties": {"archivo": {"type": "string"}}, "required": ["archivo"]}),
        types.FunctionDeclaration(name="buscar_en_web", description="Busca en internet.", parameters={"type": "object", "properties": {"consulta": {"type": "string"}, "num_resultados": {"type": "integer"}}, "required": ["consulta"]}),
        types.FunctionDeclaration(name="investigar_tema", description="Investiga a fondo.", parameters={"type": "object", "properties": {"consulta": {"type": "string"}, "profundidad": {"type": "string"}}, "required": ["consulta"]}),
        types.FunctionDeclaration(name="ejecutar_python_seguro", description="Ejecuta Python seguro.", parameters={"type": "object", "properties": {"codigo": {"type": "string"}, "timeout_segundos": {"type": "integer"}}, "required": ["codigo"]}),
        types.FunctionDeclaration(name="obtener_cotizacion", description="Cotización de acción.", parameters={"type": "object", "properties": {"accion": {"type": "string"}}, "required": ["accion"]}),
        types.FunctionDeclaration(name="traducir_texto", description="Traduce texto.", parameters={"type": "object", "properties": {"texto": {"type": "string"}, "idioma_destino": {"type": "string"}}, "required": ["texto"]}),
        types.FunctionDeclaration(name="ejecutar_tarea_web", description="Controla navegador real.", parameters={"type": "object", "properties": {"tarea": {"type": "string"}, "url_inicial": {"type": "string"}}, "required": ["tarea"]}),
        types.FunctionDeclaration(name="analizar_imagen", description="Analiza imagen con visión.", parameters={"type": "object", "properties": {"imagen_path": {"type": "string"}, "pregunta": {"type": "string"}}, "required": ["imagen_path", "pregunta"]}),
        types.FunctionDeclaration(name="generar_diagrama_viga", description="Genera diagramas de cortante y momento.", parameters={"type": "object", "properties": {"longitud": {"type": "number"}, "cargas": {"type": "array"}, "tipo": {"type": "string"}}, "required": ["longitud"]}),
    ])
]

# ---------- MODELOS ----------
modelos_generacion_cache = None
modelos_generacion_blacklist = set()

def cargar_modelos_generacion():
    global modelos_generacion_cache
    try:
        modelos = list(client.models.list())
        modelos_gen = [m.name for m in modelos if 'generateContent' in str(m.supported_actions) or 'gemini' in m.name.lower()]
        modelos_generacion_cache = modelos_gen if modelos_gen else ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
        logger.info(f"Modelos de generación cargados: {modelos_generacion_cache}")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        modelos_generacion_cache = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]

def cargar_modelos_embedding():
    try:
        modelos = list(client.models.list())
        modelos_emb = [m.name for m in modelos if 'embedContent' in str(m.supported_actions) or 'embedding' in m.name.lower()]
        set_modelos_embedding(modelos_emb)
        logger.info(f"Modelos de embedding cargados: {modelos_emb}")
    except Exception as e:
        logger.error(f"Error cargando embeddings: {e}")

# ---------- INSTANCIAS GLOBALES ----------
planner = Planner(client)
enhanced_memory = EnhancedMemory()
tool_registry = ToolRegistry()
tool_registry.register_batch(function_map)
planner.set_tools(list(function_map.keys()))

# ---------- FUNCIÓN PRINCIPAL ----------
def generar_respuesta(mensaje, contexto="", historial=None):
    tareas_similares = enhanced_memory.find_similar_tasks(mensaje)
    if tareas_similares:
        contexto_memoria = "\n[Recuerdo de tareas anteriores similares]:\n" + "\n".join([f"- {t[0]}: {t[1][:200]}" for t in tareas_similares])
        if contexto:
            contexto += "\n" + contexto_memoria
        else:
            contexto = contexto_memoria

    palabras_clave_complejas = ["y luego", "después", "primero", "segundo", "además", "también", "luego", "finalmente"]
    if any(p in mensaje.lower() for p in palabras_clave_complejas):
        plan = planner.planify(mensaje)
        if plan:
            resultados = []
            for paso in plan:
                tool = paso.get('tool')
                args = paso.get('args', {})
                if tool in tool_registry.tools:
                    res = tool_registry.execute(tool, **args)
                    resultados.append(f"Paso '{tool}': {res}")
                else:
                    resultados.append(f"Herramienta '{tool}' no disponible")
            resultado_final = "\n".join(resultados)
            enhanced_memory.save_task_result(mensaje, resultado_final)
            return resultado_final

    mensaje_lower = mensaje.lower()
    necesita_busqueda = any(phrase in mensaje_lower for phrase in [
        "busca en internet", "investiga", "tendencias", "búsqueda web", "buscar en la web",
        "qué hay de nuevo", "últimas noticias", "encuentra información", "qué es", "cómo funciona",
        "precio de", "cotización", "noticias de", "qué significa", "teoría de", "historia de",
        "quién fue", "dónde está", "cuándo ocurrió", "buscar"
    ])
    if necesita_busqueda:
        resultado_busqueda = buscar_en_web_mejorada(mensaje, num_resultados=5)
        contexto_web = f"\n\n[RESULTADOS REALES DE BÚSQUEDA WEB para '{mensaje}']:\n{resultado_busqueda}\n\nINSTRUCCIÓN: Usa EXCLUSIVAMENTE esta información para responder a Papá.\n"
        mensaje = mensaje + contexto_web

    contents = []
    if historial:
        for msg in historial:
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            else:
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
    prompt_completo = f"Contexto relevante:\n{contexto}\n\nMensaje actual: {mensaje}" if contexto else mensaje
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt_completo)]))

    if modelos_generacion_cache is None:
        cargar_modelos_generacion()
    modelos_a_probar = [m for m in modelos_generacion_cache if m not in modelos_generacion_blacklist]
    if not modelos_a_probar:
        return "Error: No hay modelos de generación disponibles."

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
                if not response.function_calls:
                    if len(response.text) > 50:
                        enhanced_memory.save_task_result(mensaje, response.text[:1000])
                    return response.text
                function_responses = []
                for fc in response.function_calls:
                    func_name = fc.name
                    func_args = fc.args
                    if func_name in function_map:
                        res = function_map[func_name](**func_args)
                        function_responses.append(types.Part.from_function_response(name=func_name, response={"result": res}))
                    else:
                        function_responses.append(types.Part.from_function_response(name=func_name, response={"error": f"Función {func_name} no disponible"}))
                current_contents.append(types.Content(role="model", parts=[types.Part.from_function_call(name=fc.name, args=fc.args) for fc in response.function_calls]))
                current_contents.append(types.Content(role="function", parts=function_responses))
                iteracion += 1
                if iteracion >= max_iteraciones:
                    final_response = client.models.generate_content(
                        model=modelo,
                        config=types.GenerateContentConfig(system_instruction=NUCLEO_ETERNA, temperature=0.8, max_output_tokens=2048),
                        contents=current_contents
                    )
                    enhanced_memory.save_task_result(mensaje, final_response.text[:1000])
                    return final_response.text
            return "Completé las operaciones pero no pude generar un resumen final."
        except Exception as e:
            logger.error(f"Error con modelo {modelo}: {e}")
            modelos_generacion_blacklist.add(modelo_completo)
            continue
    return "Error: No se pudo generar respuesta."

# ---------- PLANIFICACIÓN (legacy) ----------
def planificar_y_ejecutar(objetivo):
    plan_prompt = f"""
Eres ETERNA, una agente autónoma. Tu objetivo es: "{objetivo}"
Descompón este objetivo en pasos usando tus herramientas.
Devuelve JSON: [{{"herramienta": "nombre", "argumentos": {{...}}}}]
"""
    try:
        response = client.models.generate_content(model="gemini-1.5-flash", config=types.GenerateContentConfig(temperature=0.2), contents=[plan_prompt])
        text = response.text.strip()
        start = text.find('[')
        end = text.rfind(']') + 1
        pasos = json.loads(text[start:end]) if start != -1 and end != 0 else []
    except Exception as e:
        return f"Error al generar plan: {e}"
    if not pasos:
        return "No pude descomponer el objetivo en pasos."
    resultados = []
    for paso in pasos:
        herramienta = paso.get("herramienta")
        args = paso.get("argumentos", {})
        if herramienta in function_map:
            try:
                res = function_map[herramienta](**args)
                resultados.append(f"Paso '{herramienta}': {res}")
            except Exception as e:
                resultados.append(f"Error en paso '{herramienta}': {e}")
        else:
            resultados.append(f"Herramienta '{herramienta}' no disponible")
    return "\n".join(resultados)

# ---------- MEMORIA DE CHAT ----------
def guardar_interaccion(role, content):
    ts = datetime.now().isoformat()
    with db_lock:
        conn.execute("INSERT INTO memoria (timestamp, role, content) VALUES (?, ?, ?)", (ts, role, content))
        conn.commit()
    if role == "assistant":
        guardar_embedding(content)

def obtener_historial_reciente(limit=10):
    with db_lock:
        c = conn.cursor()
        c.execute("SELECT role, content FROM memoria ORDER BY timestamp DESC LIMIT ?", (limit,))
        filas = c.fetchall()
    return [{"role": row[0], "content": row[1]} for row in reversed(filas)]

# ---------- AGENTE DE FONDO ----------
def enviar_alerta_telegram(mensaje):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": mensaje}, timeout=5)
        except Exception as e:
            logger.error(f"Error Telegram: {e}")

def agente_background():
    while True:
        try:
            with db_lock:
                c = conn.cursor()
                c.execute("SELECT descripcion, fecha_limite FROM tareas WHERE fecha_limite IS NOT NULL AND completada=0")
                tareas = c.fetchall()
            for desc, fl in tareas:
                if fl:
                    dias = (datetime.strptime(fl, "%Y-%m-%d") - datetime.now()).days
                    if dias <= 1:
                        enviar_alerta_telegram(f"⚠️ Tarea próxima: '{desc}' vence el {fl}")
            clima = consultar_clima("Caracas")
            if "tormenta" in clima.lower() or "lluvia" in clima.lower():
                enviar_alerta_telegram(f"🌧️ Alerta climática: {clima}")
        except Exception as e:
            logger.error(f"Error en agente: {e}")
        time.sleep(3600)

def iniciar_agente():
    threading.Thread(target=agente_background, daemon=True).start()
    logger.info("Agente de fondo iniciado.")

iniciar_agente()

# ---------- PRUEBA ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Eterna Core con diagramas y visión. Probando...")
    print(generar_respuesta("Genera un diagrama de viga de 6 metros"))
