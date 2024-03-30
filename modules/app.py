import streamlit as st
from create_knowledge_graph import generate_response_based_on_knowledge_graph_with_debt
from create_vector_embeddings_llama import generate_response_based_on_vector_embeddings_with_debt

from elasticsearch_service import ElasticsearchClient

es_client = ElasticsearchClient(scheme='http', host='elasticsearch', port=9200)

st.title("Explore Auravana")

# Initialize session state for the first run
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.knowledgeGraphResponse = ""
    st.session_state.vectorEmbeddingsResponse = ""
    st.session_state.experimentKG = None
    st.session_state.experimentVDB = None
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0

# Function to reset response values
def reset_response_values():
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Before processing new input, assign satisfaction values and save experiments
    if st.session_state.experimentKG:
        st.session_state.experimentKG.satisfaction_with_answer = st.session_state.kg_response_value
        es_client.save_experiment(experiment_document=st.session_state.experimentKG)
    
    if st.session_state.experimentVDB:
        st.session_state.experimentVDB.satisfaction_with_answer = st.session_state.ve_response_value
        es_client.save_experiment(experiment_document=st.session_state.experimentVDB)

    # Reset response values for the new question
    reset_response_values()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate responses based on input
    responseKG, experimentKG = generate_response_based_on_knowledge_graph_with_debt(prompt)
    responseVE, experimentVDB = generate_response_based_on_vector_embeddings_with_debt(prompt)

    st.session_state.knowledgeGraphResponse = responseKG
    st.session_state.vectorEmbeddingsResponse = responseVE
    st.session_state.experimentKG = experimentKG
    st.session_state.experimentVDB = experimentVDB

# Display stored responses
st.markdown("Knowledge Graph: ")
st.markdown(st.session_state.knowledgeGraphResponse)
if st.button('👍', key="kg_like"):
    st.session_state.kg_response_value = 2
if st.button('👎', key="kg_dislike"):
    st.session_state.kg_response_value = 1

st.markdown("Vector embeddings: ")
st.markdown(st.session_state.vectorEmbeddingsResponse)
if st.button('👍', key="ve_like"):
    st.session_state.ve_response_value = 2
if st.button('👎', key="ve_dislike"):
    st.session_state.ve_response_value = 1
