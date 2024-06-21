import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackContext,
    CallbackQueryHandler,
    filters
)

from dotenv import load_dotenv
import requests


load_dotenv()


# TODO: Consider moving this to some other class:
def feedback_keyboard():
    keyboard = [
        [InlineKeyboardButton("⭐ 1", callback_data='star_1'),
         InlineKeyboardButton("⭐ 2", callback_data='star_2'),
         InlineKeyboardButton("⭐ 3", callback_data='star_3'),
         InlineKeyboardButton("⭐ 4", callback_data='star_4'),
         InlineKeyboardButton("⭐ 5", callback_data='star_5')],
    ]
    return InlineKeyboardMarkup(keyboard)


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
    await update.message.reply_text(answer, reply_markup=feedback_keyboard())


async def feedback_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    feedback = query.data
    # Data about the feedback:
    asked_by_telegram_user_id = query.from_user.id
    telegram_chat_id = query.message.chat.id
    telegram_message_id = query.message.message_id
    # Print the required information
    print(f"Asked by User ID: {asked_by_telegram_user_id}, Chat ID: {telegram_chat_id}, Message ID: {telegram_message_id}")
    await query.edit_message_reply_markup(reply_markup=None)  # Optionally remove the feedback buttons

    # Store the feedback and message info temporarily
    context.user_data['feedback'] = feedback
    context.user_data['original_message_id'] = telegram_message_id
    context.user_data['original_chat_id'] = telegram_chat_id

    return ConversationHandler.END  # End the conversation here


async def feedback_command(update: Update, context: CallbackContext) -> None:
    if update.message.reply_to_message and update.message.reply_to_message.message_id == context.user_data.get('original_message_id'):
        additional_feedback = update.message.text[len('/feedback '):]  # Remove the command part from the original message
        user_feedback = context.user_data.get('feedback')
        original_message_id = context.user_data.get('original_message_id')
        original_chat_id = context.user_data.get('original_chat_id')
        asked_by_telegram_user_id = update.message.from_user.id
        print(f"Asked by User ID: {asked_by_telegram_user_id}, Chat ID: {original_chat_id}, Original Message ID: {original_message_id}")
        print(f"Feedback: {user_feedback}, Additional Comments: {additional_feedback}")

        await update.message.reply_text("Thank you for your additional feedback!")
    else:
        await update.message.reply_text("Please reply to the message that asked for feedback with /feedback followed by your additional comments.")


def main():
    token = os.getenv("TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN")
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(feedback_handler))
    application.add_handler(CommandHandler("feedback", feedback_command))

    application.run_polling()


if __name__ == '__main__':
    main()
