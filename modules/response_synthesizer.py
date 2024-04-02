from typing import List
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from environment_setup import load_environment_variables
from large_language_model_setup import initialize_llm

# def get_synthesized_response(query: string, responses: string[]):
#     response_str = response_synthesizer.get_response(
#     "query string", text_chunks=["text1", "text2", ...])
#     return response_str


# technical debt, adopt so it can work with an array
# def merge_nodes(nodes_with_score1: List[NodeWithScore], nodes_with_score2: List[NodeWithScore]):
#     nodes = []
#     for item in nodes_with_score1:
#         # text = item['node']['text']
#         # score = item['score']
#         text = item.node.text  # Assuming 'node' is an attribute and it has a 'text' attribute
#         score = item.score  # Assuming 'score' is a direct attribute of NodeWithScore
#         node = Node(text=text)
#         node_with_score = NodeWithScore(node=node, score=score)
#         nodes.append(node_with_score)

#     for item in nodes_with_score2:
#         # text = item['node']['text']
#         # score = item['score']
#         text = item.node.text  # Assuming 'node' is an attribute and it has a 'text' attribute
#         score = item.score  # Assuming 'score' is a direct attribute of NodeWithScore
#         node = Node(text=text)
#         node_with_score = NodeWithScore(node=node, score=score)
#         nodes.append(node_with_score)

# def merge_nodes(nodes_with_score1: List[NodeWithScore], nodes_with_score2: List[NodeWithScore]): 
#     nodes = []
#     for item in nodes_with_score1:
#         # text = item['node']['text']
#         # score = item['score']
#         text = item.node.text  # Assuming 'node' is an attribute and it has a 'text' attribute
#         score = item.score  # Assuming 'score' is a direct attribute of NodeWithScore
#         node = Node(text=text)
#         node_with_score = NodeWithScore(node=node, score=score)
#         nodes.append(node_with_score)

#     for item in nodes_with_score2:
#         # text = item['node']['text']
#         # score = item['score']
#         text = item.node.text  # Assuming 'node' is an attribute and it has a 'text' attribute
#         score = item.score  # Assuming 'score' is a direct attribute of NodeWithScore
#         node = Node(text=text)
#         node_with_score = NodeWithScore(node=node, score=score)
#         nodes.append(node_with_score)
#     return nodes
def merge_nodes(nodes_with_score1: List[NodeWithScore], nodes_with_score2: List[NodeWithScore]): 
    return [
        NodeWithScore(node=Node(text=item.node.text), score=item.score)
        for item in nodes_with_score1 + nodes_with_score2
    ]
                
def get_synthesized_response_based_on_nodes_with_score(query: str, nodes_with_score: any):
    # technical debt - response_mode should be settings variable
    env_vars = load_environment_variables()
    # nodes = []
    # for item in nodes_with_score:
    #     # text = item['node']['text']
    #     # score = item['score']
    #     text = item.node.text  # Assuming 'node' is an attribute and it has a 'text' attribute
    #     score = item.score  # Assuming 'score' is a direct attribute of NodeWithScore
    #     node = Node(text=text)
    #     node_with_score = NodeWithScore(node=node, score=score)
    #     nodes.append(node_with_score)

    llm = initialize_llm(hf_token=env_vars['HF_TOKEN'], model_name_id = "")
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    response = response_synthesizer.synthesize(
        query, nodes=nodes_with_score
    )
    return response
