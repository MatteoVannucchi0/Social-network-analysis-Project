import typing

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from graph_creation import get_plotly_node_traces, get_plotly_map, get_plotly_edge_traces


def convert_weight(graph: nx.Graph, func):
    """
    Convert the weight of a graph
    """
    for source, target, data in graph.edges(data=True):
        data['weight'] = func(data['weight'])


def compute_centrality(graph: nx.Graph, centrality: str = "degree", **kwargs) -> typing.Tuple[
    dict[str, float], typing.Callable]:
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
    elif centrality == "betweenness":
        centralities = nx.betweenness_centrality(graph, weight="weight")

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


def compute_clique(graph: nx.Graph, **kwargs) -> typing.Tuple[dict[str, float], typing.Callable, typing.Callable]:
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


def compute_k_components(graph: nx.Graph, k=3, **kwargs) -> typing.Tuple[
    dict[str, float], typing.Callable, typing.Callable]:
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


def compute_core_periphery(graph: nx.Graph, periphery_k: int = 3, **kwargs) \
        -> typing.Tuple[dict[str, float], typing.Callable, typing.Callable]:
    """
    Compute the core-periphery decomposition of a graph
    """
    graph.remove_edges_from(nx.selfloop_edges(graph))
    core_periphery = nx.algorithms.k_core(graph, periphery_k)
    core_nodes = core_periphery.nodes

    def node_kwargs(node):
        return dict(
            marker=dict(
                size=14 if node in core_nodes else 8,
                cmin=0,
                reversescale=True,
                autocolorscale=False,
                color='red' if node in core_nodes else 'gray'
            ))

    def edge_kwargs(edge):
        source, target = edge
        return dict(
            line=dict(
                color='lightgray' if source in core_nodes and target in core_nodes else 'white',
                width=1 if source in core_nodes and target in core_nodes else 0)
        )

    return core_periphery, node_kwargs, edge_kwargs


def compute_dominating_set(graph: nx.Graph, **kwargs) -> typing.Tuple[
    dict[str, float], typing.Callable, typing.Callable]:
    """
    Compute the dominating set of a graph
    """
    dominating_set_nodes = nx.algorithms.approximation.min_weighted_dominating_set(graph)

    def node_kwargs(node):
        return dict(
            marker=dict(
                size=14 if node in dominating_set_nodes else 8,
                cmin=0,
                reversescale=True,
                autocolorscale=False,
                color='red' if node in dominating_set_nodes else 'gray'
            ))

    def edge_kwargs(edge):
        source, target = edge
        return dict(
            line=dict(
                color='yellow' if source in dominating_set_nodes and target in dominating_set_nodes else 'lightgray',
                width=0 if source in dominating_set_nodes and target in dominating_set_nodes else 0)
        )

    return dominating_set_nodes, node_kwargs, edge_kwargs


def compute_community(graph: nx.Graph, algorithm="louvain", louvain_resolution=0.7, **kwargs) \
        -> typing.Tuple[dict[str, float], typing.Callable]:
    """
    Compute the cliques of a graph
    """
    import seaborn as sns

    if algorithm == "louvain":
        x = nx.algorithms.community.louvain_communities(graph, weight="weight", resolution=louvain_resolution)
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

    return x, node_kwargs


def compute_page_rank(graph: nx.Graph, **kwargs) -> typing.Tuple[dict[str, float], typing.Callable]:
    """
    Compute the cliques of a graph
    """
    convert_weight(graph, abs)

    page_rank = nx.pagerank(graph, weight="weight")

    def node_kwargs(node):
        return dict(
            hovertext=f'{node} page rank: \n' + str(round(page_rank[node], 3)),
            hoverinfo='text',
            marker=dict(
                size=(page_rank[node] * 1000),
                cmin=0,
                reversescale=True,
                autocolorscale=False,
                color=[page_rank[node]],
                colorscale='aggrnyl',
                cmax=max(page_rank.values()),
                colorbar_title="Page Rank"
            ))

    return page_rank, node_kwargs


def get_map_centrality(graph: nx.Graph, centrality: str = "degree", **kwargs):
    """
    Compute the cliques of a graph
    """
    centrality, centrality_kwargs = compute_centrality(graph, centrality)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=centrality_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=[])
    return fig


def get_map_clique(graph: nx.Graph, **kwargs) -> go.Figure:
    """
    Compute the cliques of a graph
    """
    max_clique, node_kwargs, edge_kwargs = compute_clique(graph)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_k_components(graph: nx.Graph, k_components=3, **kwargs) -> go.Figure:
    """
    Compute the cliques of a graph
    """
    k_components, node_kwargs, edge_kwargs = compute_k_components(graph, k_components)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_dominating_set(graph: nx.Graph, **kwargs) -> go.Figure:
    dominating_set_nodes, node_kwargs, edge_kwargs = compute_dominating_set(graph)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_core_periphery(graph: nx.Graph, periphery_k: int = 3, **kwargs) -> go.Figure:
    k_components, node_kwargs, edge_kwargs = compute_core_periphery(graph, periphery_k)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    edge_traces = get_plotly_edge_traces(graph, get_scatter_geo_kwargs=edge_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=edge_traces)
    return fig


def get_map_community(graph: nx.Graph, algorithm="louvain", louvain_resolution=0.7, **kwargs) -> go.Figure:
    community, node_kwargs = compute_community(graph, algorithm, louvain_resolution)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=[])
    return fig


def get_map_page_rank(graph: nx.Graph, **kwargs) -> go.Figure:
    """
    Compute the cliques of a graph
    """

    page_rank, node_kwargs = compute_page_rank(graph)
    node_traces = get_plotly_node_traces(graph, get_scatter_geo_kwargs=node_kwargs)
    fig = get_plotly_map(graph, node_traces=node_traces, edge_traces=[])
    return fig


def compute_measure(graph: nx.Graph, measure: str, **kwargs) -> typing.Any:
    match measure:
        case "centrality-degree" | "centrality-eigenvector" | "centrality-closeness" | "centrality-katz" | "centrality-betweenness":
            centrality_measure = measure.split("-")[1]
            return compute_centrality(graph, centrality_measure)
        case "community-louvain" | "community-greedy" | "community-label":
            community_measure = measure.split("-")[1]
            return compute_community(graph, community_measure, **kwargs)
        case "k-components":
            return compute_k_components(graph, **kwargs)
        case "clique":
            return compute_clique(graph, **kwargs)
        case "core-periphery":
            return compute_core_periphery(graph, **kwargs)
        case "dominating-set":
            return compute_dominating_set(graph, **kwargs)
        case "page-rank":
            return compute_page_rank(graph, **kwargs)
        case _:
            raise ValueError(f"Measure {measure} not recognized")


def get_map_for_measure(graph: nx.graph, measure: str, **kwargs) -> go.Figure:
    match measure:
        # if startswith("centrality"):
        #     return display_centrality
        case "centrality-degree" | "centrality-eigenvector" | "centrality-closeness" | "centrality-katz" | "centrality-betweenness":
            centrality_measure = measure.split("-")[1]
            return get_map_centrality(graph, centrality_measure, **kwargs)
        case "community-louvain" | "community-greedy" | "community-label":
            community_measure = measure.split("-")[1]
            return get_map_community(graph, community_measure, **kwargs)
        case "k-components":
            return get_map_k_components(graph, **kwargs)
        case "clique":
            return get_map_clique(graph, **kwargs)
        case "core-periphery":
            return get_map_core_periphery(graph, **kwargs)
        case "dominating-set":
            return get_map_dominating_set(graph, **kwargs)
        case "page-rank":
            return get_map_page_rank(graph, **kwargs)
        case _:
            raise ValueError(f"Measure {measure} not recognized")
