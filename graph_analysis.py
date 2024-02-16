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
        'katz': nx.katz_centrality
    }
    return centrality_measures[centrality](graph)

def plot_centrality(graph: nx.Graph, centrality: dict[str, float], max_value_centrality: float = 1, self_loop: bool = False, scale: int =70) -> go.Figure:
    """
    Plot the centrality of a graph
    """
    if not self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))

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
        alpha = graph.edges[edge]['alpha']

        color = f'rgba(0,0,255,{alpha})' if line_width > 0 else f"rgba(255,0,0,{alpha})"

        edge_traces.append(go.Scattergeo(
            lon=[x0, x1],
            lat=[y0, y1],
            mode='lines',
            line=dict(width=abs(line_width), color=color),
        ))

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
            hovertext=f'{node} centrality: \n' + str(round(centrality[node], 3)),
            hoverinfo='text',
            text=node,
            marker=dict(
            size=centrality[node] * scale,
            cmin= 0,
            reversescale = True,
            autocolorscale = False,
            color=[centrality[node]],
            colorscale='aggrnyl',
            cmax=max_value_centrality,
            colorbar_title="Centrality",
            ),
        ))

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
    fig.show()

    fig.write_image(grap_path / "plotly_map.png")
    return fig


graph = load_graph_for(1990)
centrality = compute_centrality(graph, "degree")
plot_centrality(graph, centrality)

