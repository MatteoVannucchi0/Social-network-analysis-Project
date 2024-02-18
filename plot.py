import typing
from typing import Callable

import networkx as nx

from graph import geocode, grap_path, load_graph_for


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