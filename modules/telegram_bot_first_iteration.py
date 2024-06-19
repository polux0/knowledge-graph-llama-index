import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters
from dotenv import load_dotenv
import requests


load_dotenv()


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hello! I am your Auravana chatbot.')


async def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    print(f"Received message: {user_message}")
    ask_response = requests.post(
        os.getenv("ASK_URL"),
        json={'question': user_message}
    )
    if ask_response.status_code == 200:
        answer = ask_response.json().get('answer')
        print(f"Response: {answer}")
    else:
        answer = "Sorry, we could not process your question. Please try again later."
        print(f"Error: {ask_response.text}")
    print(f"Response: {answer}")
    await update.message.reply_text(answer)


def main():
    token = os.getenv("TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN")
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == '__main__':
    main()
