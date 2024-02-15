from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aggregation import aggregated_path

grap_path = Path('./graphs')
operations = ["mean", "sum"]
operation = 'other'

GEOCODE_DICT: dict[str, (str, int, float, float)] = pd.read_csv("data/country_position2.csv").set_index(
    'alpha_3').T.to_dict('list')


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
        line_width = weight / 10
        alpha = 0.35

        graph.add_edge(source, target, weight=weight, line_width=line_width, alpha=alpha)

    return graph


def geocode(node):
    if node in GEOCODE_DICT:
        _, _, longitude, latitude = GEOCODE_DICT[node]
        return latitude, longitude
    else:
        return (0, 0)
        # nonlocal south_pole_longitude, north_pole_longitude, count
        # count += 1
        # if count % 2 == 0:
        #     return np.sin(count / 5) * 160, north_pole_longitude + np.random.rand() * 10
        # else:
        #     return np.sin(count / 5) * 160, south_pole_longitude + np.random.rand() * 10


def get_map(graph: nx.Graph):
    node_x = []
    node_y = []
    for node in graph.nodes:
        latitude, longitude = geocode(node)
        node_x.append(longitude)
        node_y.append(latitude)

    edge_traces = []
    for i, edge in enumerate(graph.edges):
        source, target = edge
        x0, y0 = geocode(source)
        x1, y1 = geocode(target)

        line_width = graph.edges[edge]['line_width']
        weight = graph.edges[edge]['weight']

        color = "blue" if line_width > 0 else "red"

        print(color)

        edge_traces.append(go.Scattergeo(
            lon=[x0, x1],
            lat=[y0, y1],
            mode='lines',
            line=dict(width=abs(line_width), color=color),
        ))

    # positive_edges = []
    # positive_props = []
    #
    # negative_edges = []
    # negative_props = []
    #
    # edge_traces = []
    # for edge in graph.edges:
    #     source, target = edge
    #     x0, y0 = geocode(source)
    #     x1, y1 = geocode(target)
    #
    #     line_width = graph.edges[edge]['line_width']
    #     weight = graph.edges[edge]['weight']
    #
    #     if weight > 0:
    #         positive_edges.append((x0, y0, x1, y1))
    #         positive_props.append({
    #             "line_width": line_width,
    #             "weight": weight
    #         })
    #     else:
    #         negative_edges.append((x0, y0, x1, y1))
    #         negative_props.append({
    #             "line_width": line_width,
    #             "weight": weight
    #         })
    #
    # # Create the positive edges trace
    # edge_traces.append(go.Scattergeo(
    #     lon=[x[0] for x in positive_edges],
    #     lat=[x[1] for x in positive_edges],
    #     mode='lines',
    #     line=dict(width=[x["line_width"] for x in positive_props], color='blue'),
    # ))
    #
    # # Create the negative edges trace
    # edge_traces.append(go.Scattergeo(
    #     lon=[x[0] for x in negative_edges],
    #     lat=[x[1] for x in negative_edges],
    #     mode='lines',
    #     line=dict(width=[x["line_width"] for x in negative_props], color='red'),
    # ))

    # Node trace
    node_traces = []
    for node in graph.nodes:
        latitude, longitude = geocode(node)

        node_traces.append(go.Scattergeo(
            lon=[latitude],
            lat=[longitude],
            mode='markers+text',
            textposition="top left",
            textfont=dict(size=12, color='black'),
            text=node,
            marker=dict(size=8, color='blue'),
        ))

    # edge_x = []
    # edge_y = []
    # for edge in graph.edges:
    #     source, target = edge
    #     x0, y0 = geocode(source)
    #     x1, y1 = geocode(target)
    #     edge_x.extend([x0, x1, None])
    #     edge_y.extend([y0, y1, None])
    #
    # # Create a trace for the edges
    # edge_trace = go.Scattergeo(
    #     lon=edge_x,
    #     lat=edge_y,
    #     mode='lines',
    #     line=dict(width=0.5, color='red'),
    # )

    # Create a trace for the nodes
    # Plot Nodes
    # node_trace = go.Scattergeo(
    #     lon=node_y,
    #     lat=node_x,
    #     mode='markers',
    #     marker=dict(size=8, color='blue'),
    # )

    # Adding a simple world map outline
    world_map_trace = go.Scattergeo(
        lon=[-180, 180, 180, -180, -180],  # Simple square to mimic a map outline
        lat=[-90, -90, 90, 90, -90],
        mode='lines',
        line=dict(width=0, color='white'),
        showlegend=False
    )

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


def display_map(graph: nx.Graph):
    fig = get_map(graph)
    fig.show()


def display_earth(graph: nx.Graph):
    fig = get_map(graph)
    fig.update_geos(projection_type="orthographic")
    fig.show()


# df with alpha-3 code and Latitude and Longitude columns
graph = load_graph_for(2013, operation)
display_map(graph)
