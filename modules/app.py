from environment_setup import setup_logging
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from create_knowledge_graph import generate_response_based_on_knowledge_graph_with_debt
from create_vector_embeddings_llama import generate_response_based_on_vector_embeddings_with_debt
from response_synthesizer import get_synthesized_response_based_on_nodes_with_score, merge_nodes
import sys, logging

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
setup_logging()

from elasticsearch_service import ElasticsearchClient

es_client = ElasticsearchClient()

st.title("Explore Auravana")

# Initialize session state for the first run
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.knowledgeGraphResponse = ""
    st.session_state.vectorEmbeddingsResponse = ""
    st.session_state.responseSynthesized = ""  # Add a new state for the synthesized response
    st.session_state.experimentKG = None
    st.session_state.experimentVDB = None
    st.session_state.experimentSynthesized = None  # Add a new state for the synthesized experiment
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0
    st.session_state.synthesized_response_value = 0  # Add a new state for the synthesized response value

# Function to reset response values
def reset_response_values():
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0
    st.session_state.synthesized_response_value = 0  # Reset the synthesized response value as well

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

    if st.session_state.experimentSynthesized:  # Handle the synthesized experiment as well
        st.session_state.experimentSynthesized.satisfaction_with_answer = st.session_state.synthesized_response_value
        es_client.save_experiment(experiment_document=st.session_state.experimentSynthesized)

    # Reset response values for the new question
    reset_response_values()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Schedule the functions to be executed in parallel
        future_kg = executor.submit(generate_response_based_on_knowledge_graph_with_debt, prompt)
        future_ve = executor.submit(generate_response_based_on_vector_embeddings_with_debt, prompt)

        # Wait for the functions to complete and retrieve results
        responseKG, experimentKG, sourceNodesKG = future_kg.result()
        responseVE, experimentVDB, sourceNodesVDB = future_ve.result()
    
    nodesCombined = merge_nodes(sourceNodesKG, sourceNodesVDB)
    print("** Nodes combined: %s" % nodesCombined)

    responseSynthesized, experimentSynthesized = get_synthesized_response_based_on_nodes_with_score(prompt, nodesCombined)  # Generate the synthesized response

    st.session_state.knowledgeGraphResponse = responseKG
    st.session_state.vectorEmbeddingsResponse = responseVE
    st.session_state.responseSynthesized = responseSynthesized  # Store the synthesized response
    st.session_state.experimentKG = experimentKG
    st.session_state.experimentVDB = experimentVDB
    st.session_state.experimentSynthesized = experimentSynthesized  # Store the synthesized experiment

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

# Display synthesized response and handle feedback
st.markdown("Synthesized response: ")
st.markdown(st.session_state.responseSynthesized)
if st.button('👍', key="synthesized_like"):
    st.session_state.synthesized_response_value = 2
if st.button('👎', key="synthesized_dislike"):
    st.session_state.synthesized_response_value = 1