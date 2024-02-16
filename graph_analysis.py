import math

import networkx as nx
from graph_creation import load_graph_for, geocode, grap_path
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
    centrality = centrality_measures[centrality](graph),
    def compute_kwargs(node):
        return dict(
            hovertext = f'{node} centrality: \n' + str(round(centrality[node], 3)),
            hoverinfo = 'text',
            text = node,
            marker = dict(
            size=centrality[node] * 100,
            cmin=0,
            reversescale=True,
            autocolorscale=False,
            color=[centrality[node]],
            colorscale='aggrnyl',
            cmax=1,
            colorbar_title="Centrality"
            ))

    return centrality, compute_kwargs

def compute_clique(graph: nx.Graph) -> set:
    """
    Compute the cliques of a graph
    """
    return nx.algorithms.approximation.max_clique(graph)

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
centrality = compute_centrality(graph, "degree")
plot_centrality(graph, centrality)

