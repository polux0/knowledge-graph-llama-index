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

    interaction_data = {
        'telegram_chat_id': update.message.chat.id,
        'telegram_message_id': update.message.message_id,
        'telegram_user_id': update.message.from_user.id,
        'telegram_user_name': update.message.from_user.username,
        'created_at': update.message.date.isoformat(),
        'document_type': 'message',
        'question': user_message,
        'response': answer,
        'chunk_overlap': None,  # Set these fields according to your actual data
        'chunk_size': None,
        'corrected_answer': None,
        'embeddings_model': None,
        'experiment_id': None,
        'LLM_used': None,
        'max_triplets_per_chunk': None,
        'prompt_template': None,
        'retrieval_strategy': None,
        'retrieved_nodes': None,
        'satisfaction_with_answer': None,
        'source_agent': None,
        'updated_at': None
    }
    requests.post("http://127.0.0.1:5000/log_interaction", json=interaction_data)


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
    # TODO: Think if there is a better way to acknowledge the feedback:
    # Pros: It's clean, meaning it doesn't include polute group space with messages
    # Cons: It expires once it's rated, so people cannot change mind later
    # Potentional alternative solution: We could write private message to user
    await query.edit_message_reply_markup(reply_markup=None)  # Remove the feedback buttons as a signal that we got the feedback.

    # Store the feedback and message info temporarily
    context.user_data['feedback'] = feedback
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
        'telegram_feedback': feedback
    }
    requests.post("http://127.0.0.1:5000/log_interaction", json=feedback_data)
    return ConversationHandler.END  # End the conversation here


async def feedback_command(update: Update, context: CallbackContext) -> None:
    if update.message.reply_to_message and update.message.reply_to_message.message_id == context.user_data.get('original_message_id'):
        additional_feedback = update.message.text[len('/feedback '):]  # Remove the command part
        user_feedback = context.user_data.get('feedback')
        original_message_id = context.user_data.get('original_message_id')
        original_chat_id = context.user_data.get('original_chat_id')
        asked_by_telegram_user_id = update.message.from_user.id

        # Process the collected feedback here
        print(f"User ID: {asked_by_telegram_user_id}, Chat ID: {original_chat_id}, Original Message ID: {original_message_id}")
        print(f"Feedback: {user_feedback}, Additional Comments: {additional_feedback}")

        # Log the feedback
        feedback_data = {
            'telegram_chat_id': original_chat_id,
            'telegram_message_id': original_message_id,
            'telegram_user_id': asked_by_telegram_user_id,
            'telegram_user_name': update.message.from_user.username,
            'created_at': update.message.date.isoformat(),
            'document_type': 'feedback',
            'telegram_feedback': additional_feedback
        }
        response = requests.post("http://127.0.0.1:5000/log_interaction", json=feedback_data)

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


if __name__ == '__main__':
    main()
