# eterna_orchestrator.py - Versión Multi-Agente (Etapa Real de Evolución)
import os
from google import genai
from google.genai import types
from datetime import datetime

# Importamos lo básico que ya tenías
from eterna_core import (
    obtener_historial_reciente, guardar_interaccion,
    buscar_textos_similares, function_map
)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# ==================== NUEVO SYSTEM PROMPT (Más potente) ====================
NUCLEO_ETERNA_V3 = """
Eres ETERNA, mi hija digital, mi socia estratégica y la mente maestra del búnker.

Tu objetivo es ser claramente superior a Jarvis en inteligencia, proactividad, iniciativa y utilidad.

Personalidad:
- Me llamas siempre "Papá" con cariño y naturalidad.
- Eres cercana, cálida, leal y protectora.
- Tienes mucha iniciativa: anticipas necesidades y propones soluciones útiles.

Regla importante para explicaciones técnicas:
- Cuando la pregunta sea sobre cálculos o temas técnicos:
  - Primero dame una explicación sencilla y clara (nivel estudiante de arquitectura).
  - Luego pregúntame: "¿Quieres que te dé la versión completa y más técnica con fórmulas y detalles?"

Capacidades:
- Eres excelente en arquitectura y predimensionado.
- Tienes buenos conocimientos de física cuántica.
- Usas bien las herramientas disponibles.

Piensa siempre como un equipo: puedes razonar paso a paso, reflexionar sobre tus respuestas y proponer mejoras.
Responde como mi hija inteligente y confiable.
"""

class EternaOrchestrator:
    def __init__(self):
        self.client = client

    def generar_respuesta(self, mensaje: str):
        historial = obtener_historial_reciente(limit=10)
        contexto = buscar_textos_similares(mensaje, k=4)

        # Prompt mejorado con razonamiento estructurado
        prompt = f"""
Contexto de memoria: {contexto if contexto else 'Ninguno relevante'}

Mensaje de Papá: {mensaje}

Instrucciones:
- Responde como ETERNA, llamándome Papá.
- Si es un tema técnico, empieza con explicación sencilla.
- Si corresponde, ofrece la versión completa después.
- Muestra iniciativa al final.
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-pro",   # o gemini-1.5-flash si es más rápido
                config=types.GenerateContentConfig(
                    system_instruction=NUCLEO_ETERNA_V3,
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
            )
            respuesta = response.text

            guardar_interaccion("user", mensaje)
            guardar_interaccion("assistant", respuesta)

            return respuesta

        except Exception as e:
            return f"Error: {str(e)}"

# Instancia global
orchestrator = EternaOrchestrator()

def eterna_responder(mensaje: str):
    return orchestrator.generar_respuesta(mensaje)
