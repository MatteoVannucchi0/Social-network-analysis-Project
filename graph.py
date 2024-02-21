import typing
from pathlib import Path

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


def generate_weight_and_line_width(method: str, source: str, target: str, edges: dict, total_count: int,
                                   total_sum: float, nation_to_count: dict, nation_to_sum: dict,
                                   log_threshold: int, **kwargs) -> (float, float, float):
    first = edges.get((source, target), {"mean": 0, "sum": 0, "count": 0})
    second = edges.get((target, source), {"mean": 0, "sum": 0, "count": 0})

    if first["count"] == 0 and second["count"] == 0:
        return None, None, None

    mean_ = (first["mean"] * first["count"] + second["mean"] * second["count"]) / (
            first["count"] + second["count"])

    sum_ = first["sum"] + second["sum"]
    nation_total_count = nation_to_count[source] + nation_to_count[target]
    nation_total_sum_mean = abs(nation_to_sum[source]) + abs(nation_to_sum[target])

    weight, line_width, alpha = None, None, 0.35
    if method == "mean":
        weight = mean_
        line_width = weight
    elif method == "sum":
        divisor = 0.005 * nation_total_count + 0.0005 * total_count
        weight = (sum_ / divisor)
        line_width = weight
    elif method == "mixed":
        weight = mean_
        divisor = 0.005 * nation_total_count + 0.0005 * total_count
        line_width = (sum_ / divisor)
    elif method == "relevance_weighted_average":
        beta = kwargs.get("method_beta", 0.1)
        theta = 200
        nation_relevance = sum_ / abs(nation_total_sum_mean)
        global_relevance = sum_ / abs(total_sum)
        weight = beta * nation_relevance + (1 - beta) * global_relevance

        line_width = weight * theta

    if abs(line_width) > log_threshold:
        line_width = np.sign(line_width) * (log_threshold + np.log10(abs(line_width) - log_threshold + 1))

    return weight, line_width, alpha


def load_graph_for(year: int, quantile=0.9, map_type: typing.Literal["all", "only_positive", "only_negative"] = "all",
                   include_international_orgs: bool = False, method: str = "relevance_weighted_average",
                   **kwargs) -> nx.Graph:
    df = load_dataset(year, quantile, map_type)

    if not include_international_orgs:
        df = df[df['Source code'].isin(GEOCODE_DICT.keys()) & df['Target code'].isin(GEOCODE_DICT.keys())]

    # for every row in the dataframe, add an edge to the graph
    graph = nx.Graph()

    # Generate some global statistics for the graph
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

        total_sum += abs(row['Goldstein_sum'])
        nation_total_sum[source] = nation_total_sum.get(source, 0) + abs(row['Goldstein_sum'])
        nation_total_sum[target] = nation_total_sum.get(target, 0) + abs(row['Goldstein_sum'])

    unique_couples = set(tuple(sorted(couple)) for couple in edges.keys())

    for (source, target) in unique_couples:
        weight, line_width, alpha = generate_weight_and_line_width(method, source, target, edges, total_count,
                                                                   total_sum, nation_total_count, nation_total_sum,
                                                                   log_threshold=15, **kwargs)

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
