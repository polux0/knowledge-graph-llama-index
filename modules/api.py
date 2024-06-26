from datetime import datetime, timezone
from create_raptor_indexing_langchain import generate_response_based_on_raptor_indexing_with_debt
from flask import Flask, request, jsonify
from multi_representation_indexing import generate_response_based_on_multirepresentation_indexing_with_debt
# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
import logging


app = Flask(__name__)

elasticsearch_client = ElasticsearchClient()
logging.basicConfig(level=logging.DEBUG)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    print(f"Received data in /ask route: {data}")
    question = data.get('question')
    # Telegram related data
    telegram_chat_id = data.get('telegram_chat_id')
    telegram_message_id = data.get('telegram_message_id')
    telegram_user_id = data.get('telegram_user_id')
    telegram_user_name = data.get('telegram_user_name')
    # created_at = data.get('created_at')
    # Telegram related data

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer_raptor, experiment_raptor, source_nodes_raptor = generate_response_based_on_raptor_indexing_with_debt(question)
    answer_mri, experiment_mri, source_nodes_mri = generate_response_based_on_multirepresentation_indexing_with_debt(question)

    formatted_response = (
        f"Raptor agent answer:\n{answer_raptor}\n\n"
        f"MRI agent answer:\n{answer_mri}"
    )
    # TODO: Modularize
    additional_fields = {
        'telegram_chat_id': telegram_chat_id,
        'telegram_message_id': telegram_message_id,
        'telegram_user_id': telegram_user_id,
        'telegram_user_name': telegram_user_name,
        'document_type': 'message'
    }
    # Log interactions
    elasticsearch_client.save_interaction(experiment_raptor, additional_fields)
    elasticsearch_client.save_interaction(experiment_mri, additional_fields)

    return jsonify({
        'answer': formatted_response,
        # 'interaction_data_raptor': interaction_data_raptor,
        # 'interaction_data_mri': interaction_data_mri
    }), 200


@app.route('/rate', methods=['POST'])
def store_feedback():
    try:
        data = request.get_json()
        chat_id = data.get('telegram_chat_id')
        user_id = data.get('telegram_user_id')
        user_name = data.get('telegram_user_name')
        message_id = data.get('telegram_message_id')
        feedback_rating = data.get('telegram_feedback_rating')
        feedback_text = data.get('telegram_feedback_text')

        # Log received data for debugging
        logging.debug(f"Received data in /rate route: {data}")
        print(f"Received data in /rate route: {data}")

        if not chat_id:
            logging.error("Missing telegram_chat_id")
            print("Missing telegram_chat_id")
            return jsonify({'error': 'Missing telegram_chat_id'}), 400
        if not user_id:
            logging.error("Missing telegram_user_id")
            print("Missing telegram_user_id")
            return jsonify({'error': 'Missing telegram_user_id'}), 400
        if not user_name:
            logging.error("Missing telegram_user_name")
            print("Missing telegram_user_name")
            return jsonify({'error': 'Missing telegram_user_name'}), 400
        if not message_id:
            logging.error("Missing telegram_message_id")
            print("Missing telegram_message_id")
            return jsonify({'error': 'Missing telegram_message_id'}), 400

        # Check if at least one feedback field is provided
        if feedback_rating is None and feedback_text is None:
            logging.error("No feedback provided")
            print("No feedback provided")
            return jsonify({'error': 'No feedback provided'}), 400

        # Search for existing feedback document ( for specific user )
        logging.debug("Searching for the existing feedback")
        print("Searching for the existing feedback")
        res = elasticsearch_client.search_feedback(
            chat_id, message_id, user_id
        )

        if res['hits']['total']['value'] > 0:
            # Was used previously, we don't want things to be overrided, so we'll try another approach
            # Document exists, update it
            # doc_id = res['hits']['hits'][0]['_id']
            # existing_feedback = res['hits']['hits'][0]['_source']

            # # Update the feedback fields
            # if feedback_rating:
            #     existing_feedback['telegram_feedback_rating'] = feedback_rating
            # if feedback_text:
            #     existing_feedback['telegram_feedback_text'] = feedback_text

            # existing_feedback['updated_at'] = datetime.now(timezone.utc)
            # Testing this approach: 
            doc_id = res['hits']['hits'][0]['_id']
            new_feedback_data = {}

            # Update the feedback fields
            if feedback_rating is not None:
                new_feedback_data['telegram_feedback_rating'] = feedback_rating
            if feedback_text is not None:
                new_feedback_data['telegram_feedback_text'] = feedback_text

            new_feedback_data['updated_at'] = datetime.now(timezone.utc)
            # Update the document
            logging.debug("Updating the feedback...")  # Log received data for debugging
            logging.info("Updating the feedback...")  # Log received data for debugging
            logging.error("Updating the feedback...")
            print("Updating the feedback...")  # Log received data for debugging
            elasticsearch_client.update_feedback(doc_id, new_feedback_data)
            print(f"Updated feedback from user {user_name} ({user_id}) on message {message_id} in chat {chat_id}: {feedback_rating}, {feedback_text}")
        else:
            logging.debug("We need to create new feedback document")
            print("We need to create new feedback document")
            # Create a new feedback document
            feedback_data = {
                'telegram_chat_id': chat_id,
                'telegram_message_id': message_id,
                'telegram_user_id': user_id,
                'telegram_user_name': user_name,
                'created_at': datetime.now(timezone.utc),
                'document_type': 'feedback'
            }
            if feedback_rating is not None:
                feedback_data['telegram_feedback_rating'] = feedback_rating
            if feedback_text is not None:
                feedback_data['telegram_feedback_text'] = feedback_text

            # Log the feedback data before indexing
            logging.debug(f"Feedback data to be indexed: {feedback_data}")
            print(f"Feedback data to be indexed: {feedback_data}")
            elasticsearch_client.index_document(
                index='interaction', document=feedback_data
            )
            print(f"Created new feedback from user {user_name} ({user_id}) on message {message_id} in chat {chat_id}: {feedback_rating}, {feedback_text}")

        return jsonify({'message': 'Feedback processed successfully!'}), 201
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        logging.debug(f"Exception occurred: {e}")
        logging.info(f"Exception occurred: {e}")  # Log received data for debugging
        logging.error(f"Exception occurred: {e}")
        print(f"Exception occurred: {e}")  # Fallback print statement
        return jsonify({'error': 'Internal Server Error'}), 500


app.run(debug=True, port=5000)
