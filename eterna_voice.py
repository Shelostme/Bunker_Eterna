import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3  # TTS offline rápido (luego lo mejoramos)

model = WhisperModel("small", device="cpu", compute_type="int8")  # Usa "large-v3" si tienes GPU
engine = pyttsx3.init()

def escuchar_y_transcribir(duracion=5):
    print("Escuchando... (habla ahora)")
    audio = sd.rec(int(duracion * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    
    segments, _ = model.transcribe(audio, beam_size=5, language="es")
    texto = "".join(segment.text for segment in segments).strip()
    return texto

def hablar(texto):
    print(f"ETERNA dice: {texto}")
    engine.say(texto)
    engine.runAndWait()

# Prueba rápida
if __name__ == "__main__":
    while True:
        mensaje = escuchar_y_transcribir(4)
        if mensaje:
            print(f"Tú dijiste: {mensaje}")
            respuesta = "Entendido, Papá. Estoy procesando tu comando."  # Aquí iría generar_respuesta(mensaje)
            hablar(respuesta)
