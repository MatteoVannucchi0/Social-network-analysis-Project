import typing
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from aggregation import load_dataset

grap_path = Path('./graphs')

GEOCODE_DICT: dict[str, (str, int, float, float)] = pd.read_csv("data/country_position2.csv").set_index(
    'alpha_3').T.to_dict('list')


def geocode(node):
    if node in GEOCODE_DICT:
        _, _, longitude, latitude = GEOCODE_DICT[node]
        return latitude, longitude
    else:
        return (0, 0)


def load_graph_for(year: int, quantile=0.9, map_type: typing.Literal["all", "only_positive", "only_negative"] = "all",
                   include_international_orgs: bool = False, method: str = "sum", **kwargs) -> nx.Graph:
    df = load_dataset(year, quantile, map_type)

    if not include_international_orgs:
        df = df[df['Source code'].isin(GEOCODE_DICT.keys()) & df['Target code'].isin(GEOCODE_DICT.keys())]

    # for every row in the dataframe, add an edge to the graph
    graph = nx.Graph()

    edges = {}
    total_sum = 0
    total_count = 0
    nation_total_sum = {}
    nation_total_count = {}
    for _, row in df.iterrows():
        source, target = row['Source code'], row['Target code']
        edges[(source, target)] = {
            "mean": row['Goldstein_mean'],
            "sum": row['Goldstein_sum'],
            "count": row['Goldstein_count']
        }
        total_count += row['Goldstein_count']
        nation_total_count[source] = nation_total_count.get(source, 0) + row['Goldstein_count']
        nation_total_count[target] = nation_total_count.get(target, 0) + row['Goldstein_count']

        total_sum += row['Goldstein_sum']
        nation_total_sum[source] = nation_total_sum.get(source, 0) + row['Goldstein_sum']
        nation_total_sum[target] = nation_total_sum.get(target, 0) + row['Goldstein_sum']

    for (source, target) in set(edges.keys()):
        first = edges.get((source, target), {"mean": 0, "sum": 0, "count": 0})
        second = edges.get((target, source), {"mean": 0, "sum": 0, "count": 0})

        if first["count"] == 0 and second["count"] == 0:
            continue

        log_threshold = 15
        alpha = 0.35


        if method == "mean":
            mean_ = (first["mean"] * first["count"] + second["mean"] * second["count"]) / (
                    first["count"] + second["count"])

            weight = mean_
            line_width = weight
        elif method == "sum":
            sum_ = (first["sum"] * first["count"] + second["sum"] * second["count"]) / (
                    first["count"] + second["count"])
            nation_total = nation_total_count[source] + nation_total_count[target]
            divisor = 0.005 * nation_total + 0.0005 * total_count
            weight = (sum_ / divisor)
            line_width = weight
            if abs(line_width) > log_threshold:
                line_width = np.sign(line_width) * (log_threshold + np.log10(abs(line_width) - log_threshold + 1))
        elif method == "mixed":
            mean_ = (first["mean"] * first["count"] + second["mean"] * second["count"]) / (
                    first["count"] + second["count"])

            weight = mean_

            sum_ = (first["sum"] * first["count"] + second["sum"] * second["count"]) / (
                    first["count"] + second["count"])
            nation_total = nation_total_count[source] + nation_total_count[target]
            divisor = 0.005 * nation_total + 0.0005 * total_count
            line_width = (sum_ / divisor)
            if abs(line_width) > log_threshold:
                line_width = np.sign(line_width) * (log_threshold + np.log10(abs(line_width) - log_threshold + 1))

        elif method == "relevance_weighted_average":
            sum_ = (first["sum"] * first["count"] + second["sum"] * second["count"]) / (
                    first["count"] + second["count"])
            nation_total_sum_mean = (nation_total_sum[source] * nation_total_count[source] + nation_total_sum[
                target] * nation_total_count[target]) / (nation_total_count[source] + nation_total_count[target])

            beta = kwargs.get("method_beta", 0.1)
            theta = 100  # 300 + beta * (20 - 300)
            nation_relevance = sum_ / abs(nation_total_sum_mean)
            global_relevance = sum_ / abs(total_sum)
            weight = beta * nation_relevance + (1 - beta) * global_relevance

            line_width = weight * theta
            if abs(line_width) > log_threshold:
                line_width = np.sign(line_width) * (log_threshold + np.log10(abs(line_width) - log_threshold + 1))
        else:
            raise ValueError(f"Method {method} not recognized")

        if map_type == "only_negative":
            weight = -1 * weight

        graph.add_edge(source, target, weight=weight, line_width=line_width, alpha=alpha)

    clustering_coefficients = nx.clustering(graph, weight='weight')

    # We cannot compute assortativity since our graph is not directed
    # degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph, weight='weight')
    # small_worldness_sigma = nx.algorithms.smallworld.sigma(graph, niter=1, nrand=10)
    # small_worldness_omega = nx.algorithms.smallworld.omega(graph, niter=1, nrand=10)
    degree = dict(graph.degree())

    for node in graph.nodes:
        graph.nodes[node]['clustering_coefficient'] = clustering_coefficients[node]
        graph.nodes[node]['degree'] = degree[node]

    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph



def get_plotly_node_traces(graph: nx.Graph, get_scatter_geo_kwargs: Callable[[typing.Any], dict] = None):
    import plotly.graph_objects as go

    node_x = []
    node_y = []
    for node in graph.nodes:
        latitude, longitude = geocode(node)
        node_x.append(longitude)
        node_y.append(latitude)

    node_traces = []
    for node in graph.nodes:
        latitude, longitude = geocode(node)

        # get nodes attributes
        hovertext = f"Node: {node}<br>"
        hovertext += "<br>".join([f"{k}: {v:.3f}" for k, v in graph.nodes[node].items()])

        default_kwargs = {
            "lon": [latitude],
            "lat": [longitude],
            "mode": 'markers+text',
            "textposition": "top left",
            "textfont": dict(size=12, color='black'),
            "text": node,
            "marker": dict(size=8, color='blue'),
            "hoverinfo": 'text',
            "hovertext": hovertext,
            # f'Node: {node}<br>Clustering Coefficient: {clustering_coefficients[node]:.2f}<br>Degree: {degree[node]}'
        }

        if get_scatter_geo_kwargs:
            default_kwargs.update(get_scatter_geo_kwargs(node))

        node_traces.append(go.Scattergeo(**default_kwargs))

    return node_traces


def get_plotly_edge_traces(graph: nx.Graph, self_loop: bool = False,
                           get_scatter_geo_kwargs: Callable[[typing.Any], dict] = None):
    import plotly.graph_objects as go

    if not self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))

    edge_traces = []
    for i, edge in enumerate(graph.edges):
        source, target = edge
        x0, y0 = geocode(source)
        x1, y1 = geocode(target)

        line_width = graph.edges[edge]['line_width']
        alpha = graph.edges[edge]['alpha']

        color = f'rgba(0,0,255,{alpha})' if line_width > 0 else f"rgba(255,0,0,{alpha})"

        default_kwargs = {
            "lon": [x0, x1],
            "lat": [y0, y1],
            "mode": 'lines',
            "line": dict(width=abs(line_width), color=color),
        }

        if get_scatter_geo_kwargs:
            default_kwargs.update(get_scatter_geo_kwargs(edge))

        edge_traces.append(go.Scattergeo(**default_kwargs))

    return edge_traces


def get_plotly_world_map_trace():
    import plotly.graph_objects as go

    # Adding a simple world map outline
    world_map_trace = go.Scattergeo(
        lon=[-180, 180, 180, -180, -180],  # Simple square to mimic a map outline
        lat=[-90, -90, 90, 90, -90],
        mode='lines',
        line=dict(width=0, color='white'),
        showlegend=False
    )

    return world_map_trace


def get_plotly_map(graph: nx.Graph, self_loop: bool = False, node_traces: list = None, edge_traces: list = None):
    import plotly.graph_objects as go

    if not self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))
    if node_traces is None:
        node_traces = get_plotly_node_traces(graph)
    if edge_traces is None:
        edge_traces = get_plotly_edge_traces(graph, self_loop)
    world_map_trace = get_plotly_world_map_trace()
    traces = edge_traces + node_traces + [world_map_trace]

    # Create the figure
    fig = go.Figure(data=traces,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        geo=dict(
                            showframe=False,
                            showcoastlines=False,
                            projection_type='equirectangular'
                        ),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def display_map_plotly(graph: nx.Graph, self_loop: bool = False):
    fig = get_plotly_map(graph, self_loop)
    fig.show()

    fig.write_image(grap_path / "plotly_map.png")


def display_earth_plotly(graph: nx.Graph, self_loop: bool = False):
    fig = get_plotly_map(graph, self_loop)
    fig.update_geos(projection_type="orthographic")
    fig.show()

    fig.write_image(grap_path / "plotly_earth.png")


if __name__ == "__main__":
    graph = load_graph_for(1990)
    display_map_plotly(graph, True)
