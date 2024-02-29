from llama_index import KnowledgeGraphIndex, load_index_from_storage

def manage_knowledge_graph_index(documents, max_triplets_per_chunk, service_context, storage_context):
    """Manage the Knowledge Graph Index by either loading or constructing it.

    Args:
        documents (list): The documents to construct the index from, if needed.
        service_context (ServiceContext): The service context.
        storage_context (StorageContext): The storage context.
        index_loaded (bool): Flag indicating whether the index is already loaded.

    Returns:
        KnowledgeGraphIndex: The loaded or constructed Knowledge Graph Index.
    """
    try:
        return load_index_from_storage(storage_context=storage_context, service_context=service_context, include_embeddings=True)
    except:
        return KnowledgeGraphIndex.from_documents(documents=documents, max_triplets_per_chunk=max_triplets_per_chunk, service_context=service_context, storage_context=storage_context, include_embeddings=True)
