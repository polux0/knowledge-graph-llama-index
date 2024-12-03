from utils.environment_setup import load_environment_variables

from typing import List, Tuple

from data_loading import load_documents_langchain
from embedding_model_modular_setup import get_embedding_model_based_on_model_name_id
# Graph queries
# from gqlalchemy import Memgraph
# Graph queries, alternative that is less problematic in terms of dependencies
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Elasticsearch
from elasticsearch_service import ElasticsearchClient, ExperimentDocument
from datetime import datetime, timezone
# HuggingFace
from langchain_community.llms import HuggingFaceEndpoint
# OpenAI
from langchain_openai import OpenAIEmbeddings
# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LLM Graph transformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
# JSON output parser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import LLMChain
from langchain.chains.base import RunnableConfig

from langchain_core.prompts import ChatPromptTemplate

# Enviornment variables

env_vars = load_environment_variables()



# TODO: move this function
def parse_triplet_response(
    response: str, max_length: int = 128
) -> List[Tuple[str, str, str]]:
    knowledge_strs = response.strip().split("\n")
    results = []
    for text in knowledge_strs:
        if "(" not in text or ")" not in text or text.index(")") < text.index("("):
            # skip empty lines and non-triplets
            continue
        triplet_part = text[text.index("(") + 1 : text.index(")")]
        tokens = triplet_part.split(",")
        if len(tokens) != 3:
            continue

        if any(len(s.encode("utf-8")) > max_length for s in tokens):
            # We count byte-length instead of len() for UTF-8 chars,
            # will skip if any of the tokens are too long.
            # This is normally due to a poorly formatted triplet
            # extraction, in more serious KG building cases
            # we'll need NLP models to better extract triplets.
            continue

        subj, pred, obj = map(str.strip, tokens)
        if not subj or not pred or not obj:
            # skip partial triplets
            continue

        # Strip double quotes and Capitalize triplets for disambiguation
        subj, pred, obj = (
            entity.strip('"').capitalize() for entity in [subj, pred, obj]
        )

        results.append((subj, pred, obj))
    return results


def _parse_triplet_response(response: str, max_length: int = 128) -> List[Tuple[str, str, str]]:
    knowledge_strs = response.strip().split("\n")
    results = []
    for text in knowledge_strs:
        if "(" not in text or ")" not in text or text.index(")") < text.index("("):
            # skip empty lines and non-triplets
            continue
        triplet_part = text[text.index("(") + 1 : text.index(")")]
        tokens = triplet_part.split(",")
        if len(tokens) != 3:
            continue

        if any(len(s.encode("utf-8")) > max_length for s in tokens):
            # We count byte-length instead of len() for UTF-8 chars,
            # will skip if any of the tokens are too long.
            # This is normally due to a poorly formatted triplet
            # extraction, in more serious KG building cases
            # we'll need NLP models to better extract triplets.
            continue

        subj, pred, obj = map(str.strip, tokens)
        if not subj or not pred or not obj:
            # skip partial triplets
            continue

        # Strip double quotes and Capitalize triplets for disambiguation
        subj, pred, obj = (
            entity.strip('"').capitalize() for entity in [subj, pred, obj]
        )

        results.append((subj, pred, obj))
    return results


def generate_neo4j_query(triplets: List[Tuple[str, str, str]]) -> str:
    queries = []
    for subj, pred, obj in triplets:
        query = f"MERGE (a:Entity {{name: '{subj}'}}) " \
                f"MERGE (b:Entity {{name: '{obj}'}}) " \
                f"MERGE (a)-[:{pred.replace(' ', '_').upper()}]->(b);"
        queries.append(query)
    return "\n".join(queries)

# Constants

# Local
chunk_size = 256
chunk_overlap = 0
model_name_id = "default"
embedding_model_id = "default"
folders = ['test1']

# Production
chunk_size = 256
chunk_overlap = 0
model_name_id = "gpt-3.5-turbo"
embedding_model_id = "openai-text-embedding-3-large"
folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan',
           'social-system', 'system-overview']
# Test
folders = ['test1']


# Elasticsearch related
current_time = datetime.now(timezone.utc)
elasticsearch_client = ElasticsearchClient()
experiment = ExperimentDocument()
experiment.created_at = current_time.isoformat(timespec="milliseconds")

# Initialize LLM

# Local
repository_id = "mistralai/Mistral-7B-Instruct-v0.2"
# repository_id = "meta-llama/Meta-Llama-3-8B"
llm = HuggingFaceEndpoint(
    repo_id=repository_id,
    temperature=0.1,
    huggingfacehub_api_token=env_vars['HUGGING_FACE_INFERENCE_ENDPOINT'],
)

# Production
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Logging configuration variables, so they can be stored in elasticsearch
experiment.embeddings_model = get_embedding_model_based_on_model_name_id(
    model_name_id="openai-text-embedding-3-large"
)

experiment.chunk_size = chunk_size
experiment.chunk_overlap = chunk_overlap

# Configure GQLAlchemy to write queries

# memgraph = Memgraph(
#     env_vars['MEMGRAPH_HOST', 'localhost'],
#     env_vars['MEMGRAPH_PORT', 7687]
# )

# Configure Neo4j to write queries
uri = "bolt://127.0.0.1:7687"  # The URI for your Neo4j instance
username = "your_username"      # Your Neo4j username
password = "your_password"      # Your Neo4j password

# With username and password
# driver = GraphDatabase.driver(uri, auth=(username, password))
# Without username and password
driver = GraphDatabase.driver(uri)

# Configure LLM Graph Transformer
llm_transformer = LLMGraphTransformer(llm=llm)

prompt = PromptTemplate.from_template(
    'In JSON format, give me a list of {topic} and their '
    'corresponding names in French, Spanish and in a '
    'Cat Language.'
)


query = """
    MERGE (g:Game {name: "Baldur's Gate 3"})
    WITH g, ["PlayStation 5", "Mac OS", "Windows", "Xbox Series X/S"] AS platforms,
            ["Adventure", "Role-Playing Game", "Strategy"] AS genres
    FOREACH (platform IN platforms |
        MERGE (p:Platform {name: platform})
        MERGE (g)-[:AVAILABLE_ON]->(p)
    )
    FOREACH (genre IN genres |
        MERGE (gn:Genre {name: genre})
        MERGE (g)-[:HAS_GENRE]->(gn)
    )
    MERGE (p:Publisher {name: "Larian Studios"})
    MERGE (g)-[:PUBLISHED_BY]->(p);
"""

driver.execute_query(query)

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

TRIPLET_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["max_knowledge_triplets", "text"],
    template=DEFAULT_KG_TRIPLET_EXTRACT_TMPL
)

sequence = TRIPLET_EXTRACTION_PROMPT | llm | SimpleJsonOutputParser()

runnable_config = RunnableConfig(input_variables={"max_knowledge_triplets": 15})

# Define the chain
chain = (
    {"text": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
            "Some text is provided below. Given the text, extract up to "
            "15 "
            "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
            "---------------------\n"
            "Example:"
            "Text: Alice is Bob's mother."
            "Triplets:\n(Alice, is mother of, Bob)\n"
            "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
            "Triplets:\n"
            "(Philz, is, coffee shop)\n"
            "(Philz, founded in, Berkeley)\n"
            "(Philz, founded in, 1982)\n"
            "---------------------\n"
            "Text: {text}\n"
            "Triplets:\n"
    )
    | llm
    | StrOutputParser()
)
# Understanding if graph is empty, so we create the embeddings
# If graph!=empty
for folder in folders:
    documents_directory = f"../data/documentation_optimal/{folder}"

    documents_pdf = load_documents_langchain(documents_directory)
    # documents_text = [d.page_content for d in documents_pdf]

    # Concatenate the content, sorted and reversed
    # d_sorted = sorted(documents_pdf, key=lambda x: x.metadata["source"])
    # d_reversed = list(reversed(d_sorted))
    # concatenated_content = "\n\n\n --- \n\n\n".join(
    #     [doc.page_content for doc in d_reversed]
    # )

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents_split = text_splitter.split_documents(documents_pdf)
    documents_another_format = []
    for document_split in documents_split:
        # print("Document:\n", document_split.page_content)
        # texts_split = text_splitter.split_text(concatenated_content)
        # Invoke the sequence with the text input
        result = chain.batch([document_split], {"max_concurrency": 2})
        print("type of the result: ", type(result))
        # Run the chain with max_knowledge_triplets_parameter
        # result = chain.batch([document_split], {"max_concurrency": 2})
        print("result: ", result)
        unpacked_string = result[0]
        result_parsed = _parse_triplet_response(unpacked_string)
        print("result_parsed: ", result_parsed)
        neo4j_query = generate_neo4j_query(result_parsed)
        print(neo4j_query)
        driver.execute_query(query)
        print("executed neo4j query...")
        # graph_documents = llm_transformer.convert_to_graph_documents(
        #     [Document(page_content=text)]
        # )

# Prompt refinement


CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}
"""

# We need to insert triplets into knowledge graphs

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Configure MemgraphLangchain
graph = MemgraphGraph(
    url="bolt://localhost:7687",
    username=env_vars['MEMGRAPH_USERNAME'],
    password=env_vars['MEMGRAPH_PASSWORD'],
)
# Return the result of querying the graph directly
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True
)

response = chain.run("What are domains of the real world community model?")
print(response)
