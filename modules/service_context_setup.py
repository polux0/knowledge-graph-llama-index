from llama_index.core import ServiceContext, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore

def create_service_context(llm, chunk_size, embed_model, chunk_overlap):
    """Create a service context with default settings, LLM, and embedding model.

    Args:
        llm: The language model.
        embed_model: The embedding model.

    Returns:
        ServiceContext: The initialized service context.
    """
    return ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm, embed_model=embed_model, chunk_overlap=chunk_overlap)
