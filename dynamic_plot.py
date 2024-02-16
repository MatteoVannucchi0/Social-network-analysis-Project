from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State

from graph_analysis import get_map_for_measure
from graph_creation import load_graph_for, get_plotly_map

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive map", style={'textAlign': 'center'}, id="title"),

    html.Div([
        html.Div([
            # Make the graph larger
            dcc.Graph(id='interactive-graph', style={'width': '80%', 'height': '75vh'},
                      config={'displayModeBar': False}),
            html.Div([
                # Sliders
                html.Div([
                    html.Div([
                        html.Label('Select the year'),
                        dcc.Slider(
                            id='year-slider',
                            min=1979,  # Extract minimum year from data
                            max=2014,  # Extract maximum year from data
                            value=1979,  # Set initial value to minimum year
                            marks={str(year): str(year) for year in range(1979, 2015, 2)},
                            step=1,
                        ),
                    ], style={'textAlign': 'center', 'width': '60%', 'margin': 'auto'}),
                    html.Div([
                        html.Label('Select the quantile'),
                        dcc.Slider(
                            id='quantile-slider',
                            min=0,
                            max=1,
                            value=0.8,
                            marks={str(quantile): f"{quantile:0.2f}" for quantile in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                            step=0.01,
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                    ], style={'textAlign': 'center', 'width': '60%', 'margin': 'auto'}),
                ], style={'margin-bottom': '20px'}),

                # Select
                html.Div([
                    html.Div([
                        html.H3('Select the measure', style={'textAlign': 'center'}),
                        dcc.Dropdown(
                            id="measure-selection",
                            options=[
                                {'label': 'No measure', 'value': 'none'},
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
                            value='none',
                        ),
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    html.Div([
                        html.H3('Select the map type', style={'textAlign': 'center'}),
                        dcc.Dropdown(
                            id="map-selection",
                            options=[
                                {'label': 'Aggregated', 'value': 'all'},
                                {'label': 'Only positive', 'value': 'only_positive'},
                                {'label': 'Only negative', 'value': 'only_negative'},
                            ],
                            value='all',
                        ),
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'display': 'flex', 'justifyContent': 'center', 'width': '60%', 'margin': '0 auto'}),
            ], style={'textAlign': 'center', 'width': '60%', 'margin': 'auto'})
        ], style={'display': 'flex', 'justifyContent': 'center'}),
    ], style={'textAlign': 'center', 'width': '100%', 'margin': 'auto'}),

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

])


@app.callback(
    Output('interactive-graph', 'figure'),
    Output('title', 'children'),
    Input('year-slider', 'value'),
    Input('quantile-slider', 'value'),
    Input("measure-selection", "value"),
    Input("map-selection", "value"),
)
def display_map_interactive_plotly(year, quantile, measure, map_type):
    graph = load_graph_for(year, quantile, map_type)
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
