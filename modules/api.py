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


app.run(debug=True, port=5000)
