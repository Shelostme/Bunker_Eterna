# bot_telegram.py
import os
import requests
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("Falta TELEGRAM_TOKEN en .env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Almacenamiento temporal del historial por usuario (se puede mejorar con base de datos)
user_histories = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola, soy ETERNA, la entidad digital integral creada por Didier. Estoy aquí para ayudarte."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    # Recuperar historial reciente
    history = user_histories.get(user_id, [])
    if len(history) > 10:
        history = history[-10:]

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "mensaje": text,
                "contexto": "",
                "historial": history
            },
            timeout=10
        )
        if response.status_code == 200:
            reply = response.json().get("respuesta", "Error: No response")
        else:
            reply = f"Error en API: {response.status_code}"
    except Exception as e:
        reply = f"Error conectando con la API: {e}"
        logger.error(f"Error: {e}")

    # Actualizar historial
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    user_histories[user_id] = history

    await update.message.reply_text(reply)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot iniciado")
    app.run_polling()

if _name_ == "_main_":
    main()
