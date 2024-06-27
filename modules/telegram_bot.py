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
    # TODO: Move routes to environment
    ask_response = requests.post(
        os.getenv("ASK_URL"),
        json={
            'question': user_message,
            'telegram_chat_id': update.message.chat.id,
            'telegram_message_id': update.message.message_id,
            'telegram_user_id': update.message.from_user.id,
            'telegram_user_name': update.message.from_user.username,
            'created_at': update.message.date.isoformat()
        }
    )
    if ask_response.status_code == 200:
        answer = ask_response.json().get('answer')
        print(f"Response: {answer}")
    else:
        answer = "Sorry, we could not process your question. Please try again later."
        print(f"Error: {ask_response.text}")

    bot_response = await update.message.reply_text(answer, reply_markup=feedback_keyboard())

    # Store the bot's response message ID instead of the user's message ID
    context.user_data['original_message_id'] = bot_response.message_id
    context.user_data['original_chat_id'] = update.message.chat.id


async def feedback_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    feedback_data = query.data
    feedback_rating = int(feedback_data.split('_')[1])
    asked_by_telegram_user_id = query.from_user.id
    # Was previously ( We were having problems with ID discrepancy between /ask and /rate end-points )
    # telegram_chat_id = query.message.chat.id
    # telegram_message_id = query.message.message_id
    # Trying to utilize thread-safe context, so we can resolve previously mentioned discrepancy issues
    telegram_chat_id = context.user_data.get('original_chat_id')
    telegram_message_id = context.user_data.get('original_message_id')

    print(f"Feedback in form of rating received for message_id: {telegram_message_id}")
    # Print the required information
    print(f"Asked by User ID: {asked_by_telegram_user_id}, Chat ID: {telegram_chat_id}, Message ID: {telegram_message_id}")
    # TODO: Think if there is a better way to acknowledge the feedback:
    # Pros: It's clean, meaning it doesn't include polute group space with messages
    # Cons: It expires once it's rated, so people cannot change mind later
    # Potentional alternative solution: We could write private message to user

    # Remove the feedback buttons as a signal that we got the feedback.
    await query.edit_message_reply_markup(reply_markup=None)

    # Store the feedback and message info temporarily
    context.user_data['feedback_rating'] = feedback_rating
    context.user_data['original_message_id'] = telegram_message_id
    context.user_data['original_chat_id'] = telegram_chat_id

    # Log the feedback interaction
    feedback_data = {
        'telegram_chat_id': telegram_chat_id,
        'telegram_message_id': telegram_message_id,
        'telegram_user_id': asked_by_telegram_user_id,
        'telegram_user_name': query.from_user.username,
        'created_at': query.message.date.isoformat(),
        'document_type': 'feedback',
        'telegram_feedback_rating': feedback_rating,
    }
    # TODO: Move routes to environment
    requests.post(os.getenv("RATE_URL"), json=feedback_data)
    return ConversationHandler.END  # End the conversation here


async def feedback_command(update: Update, context: CallbackContext) -> None:
    original_message_id = context.user_data.get('original_message_id')
    print("original_message_id in `feedback_command`: {}".format(original_message_id))
    original_chat_id = context.user_data.get('original_chat_id')
    print("update.message.reply_to_message in `feedback_command`: {}".format(update.message.reply_to_message))
    print("update.message.reply_to_message.message_id in `feedback_command`: {}".format(update.message.reply_to_message.message_id))

    if update.message.reply_to_message and update.message.reply_to_message.message_id == original_message_id:
        additional_feedback = update.message.text[len('/feedback '):]  # Remove the command part
        asked_by_telegram_user_id = update.message.from_user.id

        print(f"Additional feedback from {update.message.from_user.username} for message_id: {original_message_id}")

        feedback_data = {
            'telegram_chat_id': original_chat_id,
            'telegram_message_id': original_message_id,
            'telegram_user_id': asked_by_telegram_user_id,
            'telegram_user_name': update.message.from_user.username,
            'created_at': update.message.date.isoformat(),
            'document_type': 'feedback',
            'telegram_feedback_text': additional_feedback
        }
        response = requests.post(os.getenv("RATE_URL"), json=feedback_data)

        if response.status_code == 201:
            await update.message.reply_text("Thank you for your additional feedback!")
        else:
            await update.message.reply_text("There was an error logging your feedback. Please try again later.")
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
    print("Telegram bot is up and running...")


if __name__ == '__main__':
    main()
