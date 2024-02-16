from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State

from graph_analysis import get_map_for_measure
from graph_creation import load_graph_for, get_plotly_map

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive map", style={'textAlign': 'center'}, id="title"),
    # Make the graph larger
    dcc.Graph(id='interactive-graph', style={'height': '60vh'}),
    html.Div([
        # Add a label
        html.Label('Select the year'),
        dcc.Slider(
            id='year-slider',
            min=1979,  # Extract minimum year from data
            max=2014,  # Extract maximum year from data
            value=1979,  # Set initial value to minimum year
            marks={str(year): str(year) for year in range(1979, 2015)},
            step=1
        ),
    ], style={'textAlign': 'center', 'width': '50%', 'margin': 'auto'}),
    html.Div([
        # Add a label
        html.Label('Select the quantile'),
        dcc.Slider(
            id='quantile-slider',
            min=0,
            max=1,
            value=0.8,
            marks={str(quantile): f"{quantile:0.2f}" for quantile in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            step=0.01,
            # Display the current value
            tooltip={'placement': 'bottom', 'always_visible': True}
        )],
        # Align to center and width = 20%
        style={'textAlign': 'center', 'width': '20%', 'margin': 'auto'}),
    dcc.Interval(id="animate", disabled=True),
    html.Button("Play", id='Play', style={
        'margin': 'auto',
        'display': 'block',
        'width': '50%',
        'textAlign': 'center',
        'padding': '10px',
        'background-color': '#75B2F8',  # Customizable color
        'color': '#fff',  # Customizable text color
        'border': 'none',  # Remove default border
        'font-size': '16px',  # Font size
        'font-weight': 'bold',  # Bold text
        'cursor': 'pointer',  # Indicate interactiveness
        ':hover': {
            'background-color': '#5096E3',  # Hover effect color
        }
    }),
    html.Div([
        html.Label('Select the measure'),
        dcc.RadioItems(
            id="measure-selection",
            options=[
                {'label': 'none', 'value': 'none'},
                {'label': 'centrality-degree', 'value': 'centrality-degree'},
                {'label': 'centrality-eigenvector', 'value': 'centrality-eigenvector'},
                {'label': 'centrality-closeness', 'value': 'centrality-closeness'},
                {'label': 'centrality-katz', 'value': 'centrality-katz'},
                {'label': 'centrality-betweenness', 'value': 'centrality-betweenness'},
                {'label': 'k-components', 'value': 'k-components'},
                {'label': 'clique', 'value': 'clique'},
                {'label': 'community-louvain', 'value': 'community-louvain'},
                {'label': 'community-greedy', 'value': 'community-greedy'},
                {'label': 'community-label', 'value': 'community-label'},
            ],
            labelStyle={'display': 'block'},
            value='none',
        ),
    ]),
])


@app.callback(
    Output('interactive-graph', 'figure'),
    Output('title', 'children'),
    Input('year-slider', 'value'),
    Input('quantile-slider', 'value'),
    Input("measure-selection", "value"),
)
def display_map_interactive_plotly(year, quantile, measure):
    graph = load_graph_for(year, quantile)
    print()
    if measure == "none":
        fig = get_plotly_map(graph, self_loop=False)
        title = f"Interactive map for year {year} with quantile {quantile}"
    else:
        fig = get_map_for_measure(graph, measure)
        title = f"Interactive map for year {year} with quantile {quantile} and measure {measure}"

    return fig, title


@app.callback(
    Output('year-slider', 'value'),
    Input('animate', 'n_intervals'),
    State('year-slider', 'value'),
    prevent_initial_call=True,
)
def animate(n_intervals, value):
    if value == 2014:
        return 1979

    return value + 1


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