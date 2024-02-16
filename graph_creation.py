from pathlib import Path

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from aggregation import load_dataset

grap_path = Path('./graphs')

GEOCODE_DICT: dict[str, (str, int, float, float)] = pd.read_csv("data/country_position2.csv").set_index(
    'alpha_3').T.to_dict('list')


def load_graph_for(year: int, operation: str = 'mean') -> nx.Graph:
    df = load_dataset(1990, 0.80)

    # for every row in the dataframe, add an edge to the graph
    graph = nx.Graph()

    edges = {}
    total_count = 0
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

    aggregated = {}

    for (source, target) in set(edges.keys()):
        first = edges.get((source, target), {"mean": 0, "sum": 0, "count": 0})
        second = edges.get((target, source), {"mean": 0, "sum": 0, "count": 0})

        if first["count"] == 0 and second["count"] == 0:
            continue

        mean_ = (first["mean"] * first["count"] + second["mean"] * second["count"]) / (first["count"] + second["count"])
        sum_ = (first["sum"] * first["count"] + second["sum"] * second["count"]) / (first["count"] + second["count"])
        count_ = first["count"] + second["count"]

        nation_total = nation_total_count[source] + nation_total_count[target]
        divisor = 0.005 * nation_total + 0.0005 * total_count

        weight = sum_ / divisor
        line_width = weight

        # weight = np.log(abs(sum_)) * np.sign(sum_)
        # line_width = (sum_ / (0.005 * (nation_total_count[source] + nation_total_count[target]) + 0.0005 * total_count))
        alpha = 0.35

        graph.add_edge(source, target, weight=weight, line_width=line_width, alpha=alpha)

    return graph


def geocode(node):
    if node in GEOCODE_DICT:
        _, _, longitude, latitude = GEOCODE_DICT[node]
        return latitude, longitude
    else:
        return (0, 0)


def display_graph_no_plotly(graph: nx.Graph, self_loop: bool = False):
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


def display_map_no_plotly(graph: nx.Graph, self_loop: bool = False):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if not self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))


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

        weight = graph[edge[0]][edge[1]]['weight']
        line_width = graph[edge[0]][edge[1]]['line_width']
        alpha = graph[edge[0]][edge[1]]['alpha']

        color = 'red' if weight < 0 else 'blue'

        ax.plot([start_lon, end_lon], [start_lat, end_lat], color=color,
                linewidth=line_width, transform=proj, alpha=alpha)

    # save the plot
    plt.savefig(grap_path / f'no_plotly_map')  # , dpi=300)


def get_plotly_map(graph: nx.Graph, self_loop: bool = False):
    import plotly.graph_objects as go

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
            text=node,
            marker=dict(size=8, color='blue'),
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


from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State


app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='interactive-graph'),
    dcc.Slider(
        id='year-slider',
        min=1979,  # Extract minimum year from data
        max=2014,  # Extract maximum year from data
        value=1979,  # Set initial value to minimum year
        marks={str(year): str(year) for year in range(1979, 2015)},
        step=None
    ),
    dcc.Interval(id="animate", disabled=True),
    html.Button("Play", id='Play'),
])

@app.callback(
    Output('interactive-graph', 'figure'),
    Input('year-slider', 'value')
)
def display_map_interactive_plotly(year: int):
    graph = load_graph_for(year)
    fig = get_plotly_map(graph, self_loop=False)
    return fig


@app.callback(
    Output('year-slider', 'value'),
    Input('animate', 'n_intervals'),
    State('year-slider', 'value'),
    prevent_initial_call=True,
)
def animate(n_intervals, value):
    return (value + 1) % 2014

@app.callback(
    Output('animate', 'disabled'),
    Input('Play', 'n_clicks'),
    State('animate', 'disabled')
)
def play(n, playing):
    if n:
        return not playing
    return playing


if __name__ == '__main__':
    app.run_server(debug=True)

# graph = load_graph_for(2004)
# display_map_no_plotly(graph, True)
# display_map_plotly(graph, True)
# # for operation in operations:
# #     graph = load_graph_for(1994, operation)
# #     display_map(graph, operation)
