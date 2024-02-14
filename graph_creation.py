import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aggregation import aggregated_path

COORDINATE_MAPPING_PATH = Path("data") / "coordinate_mapping.json"


def load_graph_for(year: int) -> nx.Graph:
    df = pd.read_csv(aggregated_path / f"aggregated_{year}.csv")

    # for every row in the dataframe, add an edge to the graph
    graph = nx.Graph()

    for _, row in df.iterrows():
        graph.add_edge(row['Source code'], row['Target code'], weight=row['Goldstein_mean'])

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


# def generate_coordinate_mapping(graph: nx.Graph) -> dict:
#     # Load from the file
#     try:
#         with open(COORDINATE_MAPPING_PATH, 'r', encoding="utf-8") as f:
#             coordinate_mapping = json.load(f)
#     except FileNotFoundError:
#         coordinate_mapping = {}
#     except json.JSONDecodeError:
#         coordinate_mapping = {}
#
#     # Nominatim accepts only alpha-2 country codes
#     import pycountry
#     alpha_3_to_alpha_2 = {country.alpha_3: country.alpha_2 for country in pycountry.countries}
#
#     from geopy.geocoders import Nominatim
#     from geopy.location import Location
#     geolocator = Nominatim(user_agent="test")
#
#     # For each node generate a coordinate if it doesn't exist
#     for node in graph.nodes():
#         if node not in coordinate_mapping:
#             alpha_2_node = alpha_3_to_alpha_2.get(node, None)
#             print(alpha_2_node)
#             if alpha_2_node:
#                 location: Location = geolocator.geocode({"country_codes": alpha_2_node})
#                 if location:
#                     coordinate_mapping[node] = {
#                         "longitude": location.longitude,
#                         "latitude": location.latitude,
#                         "address": location.address
#                     }
#                 else:
#                     coordinate_mapping[node] = {
#                         "longitude": None,
#                         "latitude": None,
#                         "address": None
#                     }
#             else:
#                 coordinate_mapping[node] = {
#                     "longitude": None,
#                     "latitude": None,
#                     "address": None
#                 }
#             # time.sleep(1)  # Rate limiting
#
#     # Save to the file
#     with open(COORDINATE_MAPPING_PATH, 'w', encoding="utf-8") as f:
#         json.dump(coordinate_mapping, f, indent=4, sort_keys=True, ensure_ascii=False)
#

def display_map(graph: nx.Graph, self_loop: bool = False):
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
    fig = plt.figure(figsize=(25, 15))
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
        weight = graph[edge[0]][edge[1]]['weight']

        print(edge[0], edge[1], weight)

        color = 'red' if weight < 0 else 'blue'


        # np.exp(abs(weight)) / (np.e ** 9.5),
        # weight/5000
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, linewidth=np.exp(abs(weight)) / (np.e ** 5.5), transform=proj, alpha=0.5)

    # Show plot
    # plt.show()
    # save the plot
    plt.savefig('map.png')


graph = load_graph_for(1994)
display_map(graph)
