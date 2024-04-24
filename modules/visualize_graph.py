from pyvis.network import Network


def generate_network_graph(index, database_name):
    """
    Generates a directed network graph from a given index and displays it in an HTML file.

    Parameters:
    - index: The source index from which the networkx graph is obtained.

    This function will create a visual representation of the network graph and save it as 'example.html'.
    It assumes that the index object has a method `get_networkx_graph` that returns a networkx graph object.
    """
    # Retrieve the networkx graph from the provided index
    g = index.get_networkx_graph()

    # Initialize the PyVis network, specifying that the visualization is for a Jupyter notebook
    net = Network(notebook=True, cdn_resources="in_line", directed=True)

    # Convert the networkx graph into a PyVis network graph
    net.from_nx(g)

    # Display the network graph in an HTML file named 'example.html'
    net.show(f"{database_name}.html")
