# api_eterna.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from eterna_core import (
    generar_respuesta,
    planificar_y_ejecutar,
    buscar_textos_similares,
    guardar_interaccion,
    cargar_modelos_embedding,
    cargar_modelos_generacion,
)

# Cargar modelos al arrancar
cargar_modelos_embedding()
cargar_modelos_generacion()

app = FastAPI(title="ETERNA API")

class ChatRequest(BaseModel):
    mensaje: str
    contexto: str = ""
    historial: list = []

class PlanRequest(BaseModel):
    objetivo: str

class SimilarRequest(BaseModel):
    consulta: str
    k: int = 3

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        respuesta = generar_respuesta(req.mensaje, req.contexto, req.historial)
        guardar_interaccion("user", req.mensaje)
        guardar_interaccion("assistant", respuesta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan")
def plan(req: PlanRequest):
    try:
        resultado = planificar_y_ejecutar(req.objetivo)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buscar_similares")
def buscar_similares(req: SimilarRequest):
    try:
        resultado = buscar_textos_similares(req.consulta, req.k)
        return {"textos": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)
