import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from create_knowledge_graph import generate_response_based_on_knowledge_graph
from create_vector_embeddings_llama import generate_response_based_on_vector_embeddings

st.title("Explore Auravana")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container##
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

        # Use ThreadPoolExecutor to parallelize the function calls
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_knowledge_graph = executor.submit(generate_response_based_on_knowledge_graph, prompt)
        future_vector_embeddings = executor.submit(generate_response_based_on_vector_embeddings, prompt)

        knowledgeGraphResponse = future_knowledge_graph.result()
        vectorEmbeddingsResponse = future_vector_embeddings.result()

    # knowledgeGraphResponse = generate_response_based_on_knowledge_graph(prompt)
    # vectorEmbeddingsResponse = generate_response_based_on_vector_embeddings(prompt)
    st.markdown("Knowledge Graph: ")
    st.markdown(knowledgeGraphResponse)
    st.markdown("Vector embeddings: ")
    st.markdown(vectorEmbeddingsResponse)