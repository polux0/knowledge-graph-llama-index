from concurrent.futures import ThreadPoolExecutor

from multi_representation_indexing import generate_response_based_on_multirepresentation_indexing_with_debt
import streamlit as st
# from create_knowledge_graph import generate_response_based_on_knowledge_graph_with_debt
# from create_vector_embeddings_llama import generate_response_based_on_vector_embeddings_with_debt
from create_raptor_indexing_langchain import generate_response_based_on_raptor_indexing_with_debt
from utils.environment_setup import setup_logging
# from response_synthesizer import (
#     get_synthesized_response_based_on_nodes_with_score, merge_nodes)
from streamlit_star_rating import st_star_rating

setup_logging()

from elasticsearch_service import ElasticsearchClient

es_client = ElasticsearchClient()

st.title("Explore Auravana")

# Initialize session state for the first run
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.knowledgeGraphResponse = ""
    # st.session_state.vectorEmbeddingsResponse = ""
    st.session_state.raptorIndexingResponse = ""  # Initialize new response
    st.session_state.responseSynthesized = ""
    st.session_state.multirepresentationIndexingResponse = ""  # Initialize new response1
    st.session_state.experimentKG = None
    # st.session_state.experimentVDB = None
    st.session_state.experimentRI = None  # Initialize new experiment
    st.session_state.experimentSynthesized = None
    st.session_state.experimentMRI = None  # Initialize new experiment1
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0
    st.session_state.ri_response_value = 0  # Initialize new response value
    st.session_state.mri_response_value = 0  # Initialize new response value1
    st.session_state.synthesized_response_value = 0


# Function to reset response values
def reset_response_values():
    st.session_state.kg_response_value = 0
    st.session_state.ve_response_value = 0
    st.session_state.ri_response_value = 0  # Reset the new response value
    st.session_state.synthesized_response_value = 0
    st.session_state.mri_response_value = 0  # Reset the new response value1


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Before processing new input, assign satisfaction values and save experiments
    # if st.session_state.experimentKG:
    #     st.session_state.experimentKG.satisfaction_with_answer = st.session_state.kg_response_value
    #     es_client.save_experiment(experiment_document=st.session_state.experimentKG)

    # if st.session_state.experimentVDB:
    #     st.session_state.experimentVDB.satisfaction_with_answer = st.session_state.ve_response_value
    #     es_client.save_experiment(
    #         experiment_document=st.session_state.experimentVDB
    #     )

    if st.session_state.experimentRI:
        st.session_state.experimentRI.satisfaction_with_answer = st.session_state.ri_response_value
        es_client.save_experiment(
            experiment_document=st.session_state.experimentRI
        )
    # if st.session_state.experimentSynthesized:  # Handle the synthesized experiment as well
    #     st.session_state.experimentSynthesized.satisfaction_with_answer = st.session_state.synthesized_response_value
    #     es_client.save_experiment(experiment_document=st.session_state.experimentSynthesized)

    # Reset response values for the new question
    if st.session_state.experimentMRI:
        st.session_state.experimentMRI.satisfaction_with_answer = st.session_state.mri_response_value
        es_client.save_experiment(
            experiment_document=st.session_state.experimentMRI
        )
    reset_response_values()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with ThreadPoolExecutor(max_workers=4) as executor:  # Increase max_workers to 4
        # Schedule the functions to be executed in parallel
        # future_kg = executor.submit(
        #     generate_response_based_on_knowledge_graph_with_debt,
        #     prompt
        # )
        # future_ve = executor.submit(
        #     generate_response_based_on_vector_embeddings_with_debt,
        #     prompt
        # )
        future_ri = executor.submit(
            generate_response_based_on_raptor_indexing_with_debt,
            prompt
            )  # Add new response function

        future_mri = executor.submit(
            generate_response_based_on_multirepresentation_indexing_with_debt,
            prompt
            )
        # Wait for the functions to complete and retrieve results
        # responseKG, experimentKG, sourceNodesKG = future_kg.result()
        # responseVE, experimentVDB, sourceNodesVDB = future_ve.result()
        responseRI, experimentRI, sourceNodesRI = future_ri.result()  # Retrieve new response
        responseMRI, experimentMRI, sourceNodesMRI, retrievedNodesMRI, retrievedDocsMRI = future_mri.result() # Retrieve new response1

    # nodesCombined = merge_nodes(sourceNodesKG, sourceNodesVDB, sourceNodesRI)  # Combine new nodes
    # print("** Nodes combined: %s" % nodesCombined)


    # responseSynthesized, experimentSynthesized = get_synthesized_response_based_on_nodes_with_score(prompt, nodesCombined)  # Generate the synthesized response

    # st.session_state.knowledgeGraphResponse = responseKG
    # st.session_state.vectorEmbeddingsResponse = responseVE
    st.session_state.raptorIndexingResponse = responseRI  # Store the new response
    # st.session_state.responseSynthesized = responseSynthesized  # Store the synthesized response
    # st.session_state.experimentKG = experimentKG
    # st.session_state.experimentVDB = experimentVDB
    st.session_state.experimentRI = experimentRI  # Store the new experiment
    # st.session_state.experimentSynthesized = experimentSynthesized  # Store the synthesized experiment
    st.session_state.multirepresentationIndexingResponse = responseMRI
    st.session_state.experimentMRI = experimentMRI

# Display stored responses
# st.markdown("Knowledge Graph: ")
# st.markdown(st.session_state.knowledgeGraphResponse)
# if st.button('👍', key="kg_like"):
#     st.session_state.kg_response_value = 2
# if st.button('👎', key="kg_dislike"):
#     st.session_state.kg_response_value = 1

# st.markdown("Parent-child Indexing: ")
# st.markdown(st.session_state.vectorEmbeddingsResponse)
# if st.button('👍', key="ve_like"):
#     st.session_state.ve_response_value = 2
# if st.button('👎', key="ve_dislike"):
#     st.session_state.ve_response_value = 1

st.markdown("Raptor Indexing: ")
st.markdown(st.session_state.raptorIndexingResponse)
ri_rating = st_star_rating(label="Please rate your experience", maxValue=5, defaultValue=3, key="ri_rating", read_only=False)
if ri_rating:
    st.session_state.ri_response_value = ri_rating

# Display synthesized response and handle feedback
# st.markdown("Synthesized response: ")
# st.markdown(st.session_state.responseSynthesized)
# if st.button('👍', key="synthesized_like"):
#     st.session_state.synthesized_response_value = 2
# if st.button('👎', key="synthesized_dislike"):
#     st.session_state.synthesized_response_value = 1

st.markdown("Multirepresentation Indexing: ")
st.markdown(st.session_state.multirepresentationIndexingResponse)
mri_rating = st_star_rating(label="Please rate your experience", maxValue=5, defaultValue=3, key="mri_rating", read_only=False)
if mri_rating:
    st.session_state.mri_response_value = mri_rating
