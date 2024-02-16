import math

import networkx as nx
from graph_creation import load_graph_for, geocode, grap_path, get_plotly_edge_traces, get_plotly_node_traces, \
    get_plotly_map
import plotly.graph_objects as go


def compute_centrality(graph: nx.Graph, centrality: str = "degree") -> dict[str, float] | None:
    """
    Compute the centrality of a graph
    """
    centrality_measures = {
        'degree': nx.degree_centrality,
        'eigenvector': nx.eigenvector_centrality,
        'closeness': nx.closeness_centrality,
        'katz': nx.katz_centrality,
        'betweenness': nx.betweenness_centrality
    }
    centralities = centrality_measures[centrality](graph)
    def compute_kwargs(node):
        return dict(
            hovertext = f'{node} centrality: \n' + str(round(centralities[node], 3)),
            hoverinfo = 'text',
            text = node,
            marker = dict(
            size=centralities[node] * 100,
            cmin=0,
            reversescale=True,
            autocolorscale=False,
            color=[centralities[node]],
            colorscale='aggrnyl',
            cmax=1,
            colorbar_title="Centrality"
            ))

    return centralities, compute_kwargs

def compute_clique(graph: nx.Graph) -> set:
    """
    Compute the cliques of a graph
    """
    max_clique = nx.algorithms.approximation.max_clique(graph)
    def node_kwargs(node):
        return dict(
            marker = dict(
            size= 14 if node in max_clique else 8,
            cmin=0,
            reversescale=True,
            autocolorscale=False,
            color= 'red' if node in max_clique else 'gray'
            ))
    def edge_kwargs(edge):
        source, target = edge
        return dict(
            line = dict(
            color='yellow' if source in max_clique and target in max_clique else 'lightgray',
            width= 4 if source in max_clique and target in max_clique else 1,)
            )

    return max_clique, node_kwargs, edge_kwargs

def compute_k_components(graph: nx.Graph) -> dict[int, set] | None:
    """
    Compute the cliques of a graph
    """
    return nx.algorithms.approximation.k_components(graph)

def compute_communities(graph: nx.Graph) -> dict[str, float] | None:
    """
    Compute the cliques of a graph
    """
    return nx.greedgreedy_modularity_communities(graph, weight='weight')



graph = load_graph_for(1990)
#cliques, node_kwargs, edge_kwargs = compute_clique(graph)
#centrality, centrality_kwargs = compute_centrality(graph, "betweenness")
k_components = compute_k_components(graph)
#node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
#edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
#fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
#fig.show()
print()
