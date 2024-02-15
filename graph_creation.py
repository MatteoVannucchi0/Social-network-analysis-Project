from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aggregation import aggregated_path

grap_path = Path('./graphs')
operations = ["mean", "sum"]
operation = 'other'


def load_graph_for(year: int, operation: str = 'mean') -> nx.Graph:
    df = pd.read_csv(aggregated_path / f"aggregated_{year}.csv")

    # for every row in the dataframe, add an edge to the graph
    graph = nx.Graph()

    edges = {}
    total_count = 0
    for _, row in df.iterrows():
        source, target = row['Source code'], row['Target code']
        edges[(source, target)] = {
            "mean": row['Goldstein_mean'],
            "sum": row['Goldstein_sum'],
            "count": row['Goldstein_count']
        }
        total_count += row['Goldstein_count']

    aggregated = {}

    for (source, target) in set(edges.keys()):
        first = edges.get((source, target), {"mean": 0, "sum": 0, "count": 0})
        second = edges.get((target, source), {"mean": 0, "sum": 0, "count": 0})

        if first["count"] == 0 and second["count"] == 0:
            continue

        mean_ = (first["mean"] * first["count"] + second["mean"] * second["count"]) / (first["count"] + second["count"])
        sum_ = (first["sum"] * first["count"] + second["sum"] * second["count"]) / (first["count"] + second["count"])
        count_ = first["count"] + second["count"]

        weight = np.log(abs(sum_)) * np.sign(sum_)
        line_width = weight / 3
        alpha = 0.35

        graph.add_edge(source, target, weight=weight, line_width=line_width, alpha=alpha)

    return graph


def display_graph(graph: nx.Graph, self_loop: bool = False):
    if not self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))

    # Community detection
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i

    nx.set_node_attributes(graph, community_map, 'community')
    colors = [graph.nodes[node]['community'] for node in graph.nodes]

    pos = nx.spring_layout(graph, k=5, iterations=100)
    node_sizes = [v * 5 for v in dict(graph.degree).values()]
    node_degrees = dict(graph.degree())

    # Draw the graph with additional styling
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, alpha=0.7, node_color=colors)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=weights, alpha=0.2, edge_color='black')

    # Draw only node labels for significant nodes (e.g., major hubs or nodes with high degree)
    labels = {}
    for node in graph.nodes():
        if graph.degree[node] > 70:  # arbitrary threshold for significant nodes
            labels[node] = node

    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights)
    # set the image size to be 15x15
    plt.figure(figsize=(25, 20))
    plt.axis('off')  # No axis for a cleaner look
    plt.show()


def add_cyclic_point(lon, lat, data=None):
    """
    Adds a cyclic point to a line to properly handle the dateline discontinuity.
    """
    if lon[-1] < lon[0]:
        lon = np.ma.concatenate((lon, [lon[0] + 360]))
        lat = np.ma.concatenate((lat, [lat[0]]))
        if data is not None:
            data = np.ma.concatenate((data, [data[0]]))
    else:
        lon = np.ma.concatenate(([lon[-1] - 360], lon))
        lat = np.ma.concatenate(([lat[-1]], lat))
        if data is not None:
            data = np.ma.concatenate(([data[-1]], data))
    return lon, lat, data

def display_map(graph: nx.Graph, self_loop: bool = False, operation: str = 'mean'):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pandas as pd

    # df with alpha-3 code and Latitude and Longitude columns
    df = pd.read_csv("data/country_position2.csv")

    # Create a dictionary with alpha-3 code as key and a tuple with latitude and longitude as value
    geocode_dict: dict[str, (str, int, float, float)] = df.set_index('alpha_3').T.to_dict('list')

    south_pole_longitude = -80
    north_pole_longitude = 80
    count = 0

    def geocode(node):
        if node in geocode_dict:
            _, _, longitude, latitude = geocode_dict[node]
            return latitude, longitude
        else:
            nonlocal south_pole_longitude, north_pole_longitude, count
            count += 1
            if count % 2 == 0:
                return np.sin(count / 5) * 160, north_pole_longitude + np.random.rand() * 10
            else:
                return np.sin(count / 5) * 160, south_pole_longitude + np.random.rand() * 10

    # Define the map projectio
    proj = ccrs.PlateCarree()

    # Create a matplotlib figure with a GeoAxes set with the projection
    fig = plt.figure(figsize=(50, 30))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN)  # , fill_color='lightblue')
    ax.add_feature(cfeature.LAND)  # , fill_color='lightgreen')

    # Set the extent of the map in the desired coordinates (longitude/latitude)
    ax.set_extent([-180, 180, -90, 90], crs=proj)

    # Transform graph nodes to the map's projection
    node_positions = {node: geocode(node) for node in graph.nodes()}

    # Draw nodes
    for node, (lon, lat) in node_positions.items():
        ax.plot(lon, lat, marker='o', color='red', markersize=4, transform=proj)

    # Add labels
    for node, (lon, lat) in node_positions.items():
        ax.text(lon, lat, node, transform=ccrs.Geodetic(), fontsize=8)

    # Draw edges
    for edge in graph.edges():
        start_pos, end_pos = node_positions[edge[0]], node_positions[edge[1]]

        start_lon, start_lat = start_pos
        end_lon, end_lat = end_pos

        if abs(start_lon - end_lon) > 180:
            start_lon, start_lat, _ = add_cyclic_point(np.array([start_lon]), np.array([start_lat]))
            end_lon, end_lat, _ = add_cyclic_point(np.array([end_lon]), np.array([end_lat]))

        weight = graph[edge[0]][edge[1]]['weight']
        line_width = graph[edge[0]][edge[1]]['line_width']
        alpha = graph[edge[0]][edge[1]]['alpha']

        color = 'red' if weight < 0 else 'blue'

        ax.plot([start_lon, end_lon], [start_lat, end_lat], color=color,
                linewidth=line_width, transform=proj, alpha=alpha)

    # Show plot
    # plt.show()
    # save the plot
    plt.savefig(grap_path / f'{operation}_map')  # , dpi=300)


graph = load_graph_for(2013, operation)
display_map(graph, False, operation)
# for operation in operations:
#     graph = load_graph_for(1994, operation)
#     display_map(graph, operation)
