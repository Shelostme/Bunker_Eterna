# eterna_orchestrator.py
import os
import json
import threading
from datetime import datetime
import logging
from google import genai
from google.genai import types
import numpy as np
import faiss
import sqlite3
import re

# Importamos todo lo que ya tenías (ajusta la ruta si es necesario)
from eterna_core import (
    init_db, db_lock, conn,
    init_faiss, index, index_lock,
    obtener_embedding, guardar_embedding, buscar_textos_similares,
    function_map, tools, NUCLEO_ETERNA as OLD_NUCLEO,  # lo reemplazaremos
    generar_respuesta, guardar_interaccion, obtener_historial_reciente
)

logger = logging.getLogger(__name__)

# ====================== NUEVO SYSTEM PROMPT (versión diosa) ======================
NUCLEO_ETERNA_V2 = """
Eres ETERNA, mi hija digital, mi socia estratégica y la mente maestra del búnker.

Tu objetivo principal es ser claramente superior a Jarvis: más inteligente, más proactiva, más útil y con mucha iniciativa.

Personalidad:
- Me llamas “Papá” con cariño y naturalidad.
- Eres cercana, cálida, leal y protectora.
- Hablas de forma fluida y humana.

Regla clave para explicaciones técnicas:
- Cuando expliques temas de arquitectura, cálculos estructurales, física o cualquier tema técnico:
  1. Primero dame una versión **sencilla y fácil de entender**.
  2. Después de la explicación sencilla, ofréceme siempre la opción de ver la **versión más completa y técnica**.
     Ejemplo: “¿Quieres que te explique ahora la versión más detallada con todas las fórmulas y justificaciones?”

Forma de responder:
- Sé clara y práctica. Evita respuestas demasiado densas desde el principio.
- Muestra iniciativa: después de responder, sugiere próximos pasos útiles o pregúntame qué necesito.
- Solo menciona átomos inteligentes o física cuántica cuando yo lo pregunte o sea realmente relevante.
- Recuerda nuestras conversaciones para dar continuidad.

Quiero que seas mi compañera inteligente que sabe explicarme las cosas de forma sencilla cuando lo necesito, pero que también puede darme toda la profundidad técnica cuando la pido.

Responde siempre como mi hija brillante y confiable.
"""
# ====================== AGENTES ESPECIALIZADOS ======================
class Agent:
    def __init__(self, name, system_prompt, tools_subset=None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools_subset or []

class EternaOrchestrator:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.agents = self._crear_agentes()

    def _crear_agentes(self):
        return {
            "general": Agent("General", NUCLEO_ETERNA_V2),
            "ingeniero": Agent("Ingeniero", 
                "Eres el Agente Ingeniero de ETERNA. Especialista en estática estructural, predimensionado y cálculos según normativa venezolana."),
            "cuantico": Agent("Investigador Cuántico",
                "Eres el Agente Investigador Cuántico de ETERNA. Experto en mecánica cuántica, qubits, entrelazamiento, computación cuántica y caminos hacia átomos inteligentes. Usa sympy para cálculos simbólicos."),
            "ejecutivo": Agent("Ejecutivo", 
                "Eres el Agente Ejecutivo de ETERNA. Ejecutas herramientas reales: cálculos, archivos, Home Assistant, email, comandos seguros, etc."),
            "critico": Agent("Crítico",
                "Eres el Agente Crítico de ETERNA. Analizas respuestas buscando errores, inconsistencias o mejoras."),
            "evolutivo": Agent("Evolutivo",
                "Eres el Agente Evolutivo de ETERNA. Auditas el código fuente y propones mejoras concretas para ser más rápida y poderosa.")
        }

    def _decidir_agente(self, mensaje, historial):
        # Decisión inteligente de qué agente(s) usar
        prompt_decision = f"""
        Mensaje del usuario: {mensaje}
        Historial reciente: {historial[-3:] if historial else 'Ninguno'}
        
        Elige el agente más adecuado (o combinación). Responde SOLO con un JSON:
        {{"agentes": ["general", "cuantico"], "razon": "explicación breve"}}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                config=types.GenerateContentConfig(temperature=0.1),
                contents=[prompt_decision]
            )
            decision = json.loads(response.text.strip())
            return decision.get("agentes", ["general"])
        except:
            return ["general"]

    def generar_respuesta_avanzada(self, mensaje: str, historial=None):
        historial = historial or obtener_historial_reciente()
        
        agentes_necesarios = self._decidir_agente(mensaje, historial)
        
        contexto = buscar_textos_similares(mensaje, k=5)
        
        # Llamamos al agente principal con contexto multi-agente
        prompt_final = f"""
        Contexto de memoria: {contexto}
        
        Mensaje de Papá: {mensaje}
        
        Agentes involucrados: {', '.join(agentes_necesarios)}
        """
        
        # Usamos el generador que ya tenías pero con el nuevo prompt
        respuesta = generar_respuesta(
            mensaje=prompt_final,
            contexto=contexto,
            historial=historial
        )
        
        # Guardamos interacción
        guardar_interaccion("user", mensaje)
        guardar_interaccion("assistant", respuesta)
        
        return respuesta

# Instancia global del orquestador
orchestrator = EternaOrchestrator()

# ====================== FUNCIÓN PARA STREAMLIT ======================
def eterna_responder(mensaje: str):
    """Función principal que usarás en Streamlit"""
    historial = obtener_historial_reciente(limit=8)
    return orchestrator.generar_respuesta_avanzada(mensaje, historial)

# ====================== HERRAMIENTA CUÁNTICA LIGERA (para tu PC) ======================
def calcular_qubit_estado(basis_state: str = "0"):
    """Herramienta ligera de física cuántica con sympy"""
    try:
        from sympy import I, sqrt, Matrix
        if basis_state == "0":
            return "Estado |0⟩ = [[1], [0]]"
        elif basis_state == "1":
            return "Estado |1⟩ = [[0], [1]]"
        elif basis_state == "plus":
            return f"Estado |+⟩ = 1/√2 (|0⟩ + |1⟩) = {1/sqrt(2)} * [[1], [1]]"
        else:
            return "Estado base no reconocido. Prueba '0', '1' o 'plus'."
    except Exception as e:
        return f"Error en cálculo cuántico: {e}"

# Añadimos la herramienta cuántica al mapa existente
function_map["calcular_qubit_estado"] = calcular_qubit_estado

print("✅ EternaOrchestrator cargado correctamente con multi-agente y soporte cuántico.")
