# api_eter_na.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# === IMPORTANTE: Nueva importación del orquestador ===
from eterna_orchestrator import eterna_responder

app = FastAPI(title="ETERNA API - Orquestador Multi-Agente")

class ChatRequest(BaseModel):
    mensaje: str
    contexto: str = ""
    historial: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Aquí se usa el nuevo sistema multi-agente + física cuántica
        respuesta = eterna_responder(request.mensaje)
        
        return {
            "status": "success",
            "respuesta": respuesta
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando mensaje: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "ETERNA API funcionando con Orquestador v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
