import datetime
from create_raptor_indexing_langchain import generate_response_based_on_raptor_indexing_with_debt
from flask import Flask, request, jsonify
from multi_representation_indexing import generate_response_based_on_multirepresentation_indexing_with_debt

app = Flask(__name__)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer_raptor, experiment_raptor, source_nodes_raptor = generate_response_based_on_raptor_indexing_with_debt(question)
    answer_mri, experiment_mri, source_nodes_mri = generate_response_based_on_multirepresentation_indexing_with_debt(question)

    formatted_response = (
        f"Raptor agent answer:\n{answer_raptor}\n\n"
        f"MRI agent answer:\n{answer_mri}"
    )

    return jsonify({'answer': formatted_response}), 200


@app.route('/rate', methods=['POST'])
def store_feedback():
    data = request.get_json()
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    user_name = data.get('user_name')
    message_id = data.get('message_id')
    feedback_rating = data.get('feedback_rating')
    feedback_text = data.get('feedback_text')

    if not all([chat_id, user_id, user_name, message_id]) or (not feedback_rating and not feedback_text):
        return jsonify({'error': 'Missing required fields'}), 400

    # Search for existing feedback document
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"telegram_chat_id": chat_id}},
                    {"term": {"telegram_message_id": message_id}},
                    {"term": {"telegram_user_id": user_id}},
                    {"term": {"document_type": "feedback"}}
                ]
            }
        }
    }
    res = es.search(index='interaction', body=query)

    if res['hits']['total']['value'] > 0:
        # Document exists, update it
        doc_id = res['hits']['hits'][0]['_id']
        existing_feedback = res['hits']['hits'][0]['_source']

        # Update the feedback fields
        if feedback_rating:
            existing_feedback['telegram_feedback_rating'] = feedback_rating
        if feedback_text:
            existing_feedback['telegram_feedback_text'] = feedback_text

        existing_feedback['updated_at'] = datetime.datetime.utcnow().isoformat()

        # Update the document
        es.update(index='interaction', id=doc_id, body={"doc": existing_feedback})
        print(f"Updated feedback from user {user_name} ({user_id}) on message {message_id} in chat {chat_id}: {feedback_rating}, {feedback_text}")
    else:
        # Create a new feedback document
        feedback_data = {
            'telegram_chat_id': chat_id,
            'telegram_message_id': message_id,
            'telegram_user_id': user_id,
            'telegram_user_name': user_name,
            'telegram_feedback_rating': feedback_rating,
            'telegram_feedback_text': feedback_text,
            'created_at': datetime.datetime.utcnow().isoformat(),
            'document_type': 'feedback'
        }
        es.index(index='interaction', body=feedback_data)
        print(f"Created new feedback from user {user_name} ({user_id}) on message {message_id} in chat {chat_id}: {feedback_rating}, {feedback_text}")

    return jsonify({'message': 'Feedback processed successfully!'}), 201


app.run(debug=True, port=5000)
