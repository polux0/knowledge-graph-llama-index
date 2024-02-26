from pyvis.network import Network

def visualize_knowledge_graph(index, output_file):
    """Visualize the Knowledge Graph and save it to an HTML file.

    Args:
        index (KnowledgeGraphIndex): The Knowledge Graph Index to visualize.
        output_file (str): The path to the output HTML file.

    """
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show(output_file)