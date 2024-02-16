import collections

import networkx as nx
import numpy as np

from graph_creation import load_graph_for, get_plotly_node_traces, get_plotly_map, get_plotly_edge_traces


def clustering(graph: nx.Graph) -> None:
    """
    Compute the clustering of a graph
    """
    average_clustering = nx.average_clustering(graph, weight='weight')
    return average_clustering

def clustering_coefficient(graph: nx.Graph, node: 'str') -> None:
    """
    Compute the clustering of a graph
    """
    node_clustering = nx.clustering(graph, weight='weight')
    return node_clustering


def convert_weight(graph: nx.Graph, func) -> None:
    """
    Convert the weight of a graph
    """
    for source, target, data in graph.edges(data=True):
        data['weight'] = func(data['weight'])


def compute_centrality(graph: nx.Graph, centrality: str = "degree") -> dict[str, float] | None:
    """
    Compute the centrality of a graph
    """

    if centrality == "degree":
        centralities = nx.degree_centrality(graph)
    elif centrality == "eigenvector":
        convert_weight(graph, abs)
        centralities = nx.eigenvector_centrality(graph, weight="weight")
    elif centrality == "closeness":
        min_value = min([data['weight'] for _, _, data in graph.edges(data=True)])
        convert_weight(graph, lambda x: 1 / (x - min_value + 0.01))
        centralities = nx.closeness_centrality(graph, distance="weight")
        max_centrality = max(centralities.values())
        centralities = {k: v / max_centrality for k, v in centralities.items()}
    elif centrality == "katz":
        centralities = nx.katz_centrality(graph, weight="weight")

    def compute_kwargs(node):
        return dict(
            hovertext=f'{node} centrality: \n' + str(round(centralities[node], 3)),
            hoverinfo='text',
            text=node,
            marker=dict(
                size=(centralities[node] * 4) ** 3,
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
            marker=dict(
                size=14 if node in max_clique else 8,
                cmin=0,
                reversescale=True,
                autocolorscale=False,
                color='red' if node in max_clique else 'gray'
            ))

    def edge_kwargs(edge):
        source, target = edge
        return dict(
            line=dict(
                color='yellow' if source in max_clique and target in max_clique else 'lightgray',
                width=4 if source in max_clique and target in max_clique else 1, )
        )

    return max_clique, node_kwargs, edge_kwargs


def compute_k_components(graph: nx.Graph, k=3) -> collections.defaultdict:
    """
    Compute the cliques of a graph
    """
    graph.remove_edges_from(nx.selfloop_edges(graph))
    components = nx.approximation.k_components(graph)
    k_components = components.get(k, [])
    node_colors = [f'rgb{tuple(np.random.randint(0, 256, size=3))}' for _ in range(len(components))]
    edge_colors = [f'rgb{tuple(np.random.randint(0, 256, size=3))}' for _ in range(len(components))]
    nodes_seen = {}

    def node_kwargs(node):
        color = 'gray'
        size = 6
        for i, component in enumerate(k_components):
            if node in component:
                color = node_colors[i]
                size = 16
                if node in nodes_seen.keys():
                    nodes_seen[node] = nodes_seen[node] + [i]
                else:
                    nodes_seen[node] = [i]
        return dict(
            hovertext=f'{node} components {nodes_seen.get(node, "none")}',
            hoverinfo='text',
            marker=dict(
                color=color,
                size=size,
            ))

    def edge_kwargs(edge):
        source, target = edge
        width = 0
        color = 'white'
        for i, component in enumerate(k_components):
            if source in component and target in component:
                color = edge_colors[i]
                width = 0
        return dict(
            line=dict(
                color=color,
                width=width)
        )

    return components, node_kwargs, edge_kwargs


def get_map_centrality(graph: nx.Graph, centrality: str = "degree") -> None:
    """
    Compute the cliques of a graph
    """
    centrality, centrality_kwargs = compute_centrality(graph, centrality)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=centrality_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=[])
    return fig

def get_map_clique(graph: nx.Graph) -> None:
    """
    Compute the cliques of a graph
    """
    max_clique, node_kwargs, edge_kwargs = compute_clique(graph)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_k_components(graph: nx.Graph) -> None:
    """
    Compute the cliques of a graph
    """
    k_components, node_kwargs, edge_kwargs = compute_k_components(graph)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_community(graph: nx.Graph, algorithm="louvain") -> None:
    """
    Compute the cliques of a graph
    """
    import seaborn as sns

    if algorithm == "louvain":
        x = nx.algorithms.community.louvain_communities(graph, weight="weight", resolution=0.7)
    elif algorithm == "greedy":
        x = nx.algorithms.community.greedy_modularity_communities(graph)
    elif algorithm == "label":
        x = list(nx.algorithms.community.asyn_lpa_communities(graph, weight="weight"))
    # elif algorithm == "girvan_newman":
    # x = list(nx.algorithms.community.girvan_newman(graph))

    color_palette = sns.color_palette("hls", len(x))

    def node_kwargs(node):
        for i, community in enumerate(x):
            if node in community:
                size = 16
                return dict(
                    marker=dict(color=f"rgb{color_palette[i]}", size=size),
                    hovertext=f'{node} community: \n' + str(i),
                )
        return dict(marker=dict(color="gray"))

    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=[])
    return fig


def get_map_for_measure(graph: nx.graph, measure: str) -> None:
    match measure:
        # if startswith("centrality"):
        #     return display_centrality
        case "centrality-degree" | "centrality-eigenvector" | "centrality-closeness" | "centrality-katz" | "centrality-betweenness":
            centrality_measure = measure.split("-")[1]
            return get_map_centrality(graph, centrality_measure)
        case "community-louvain" | "community-greedy" | "community-label":
            community_measure = measure.split("-")[1]
            return get_map_community(graph, community_measure)
        case "k-components":
            return get_map_k_components(graph)
        case "clique":
            return get_map_clique(graph)
        case _:
            raise ValueError(f"Measure {measure} not recognized")
