import networkx as nx
import numpy as np
import pandas as pd
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State

from analysis import get_map_for_measure
from graph import load_graph_for
from plot import get_plotly_map

app = Dash(__name__)
server = app.server

graph_component = dcc.Graph(id='interactive-graph',
                            style={'width': '80%', 'height': '75vh', "border": "1px solid black"},
                            config={'displayModeBar': True})

method_beta_value_slider = html.Div(id='method-beta-slider-container', children=[
    html.Label('Select the beta value'),
    dcc.Slider(id='method-beta-slider',
               min=0,
               max=1,
               step=0.01,
               value=0.1,
               marks={str(i): f"{i:.2f}" for i in np.arange(0, 1, 0.1)},
               )

], style={'display': 'block', 'textAlign': 'center', 'width': '100%', 'margin': 'auto', "padding-top": "10px"},
                                    hidden=True)

k_components_slider = html.Div(id='k-slider-container', children=[
    html.Label('Select the number of components'),
    # Create element to hide/show, in this case a slider
    dcc.Slider(id='k-slider',
               min=1,
               max=20,
               step=1,
               value=5,
               marks={str(i): str(i) for i in range(1, 20)})

], style={'display': 'block', 'textAlign': 'center', 'width': '60%', 'margin': 'auto'},
                               hidden=True)  # <-- This is the line that will be changed by the dropdown callback

louvain_slider = html.Div(id='louvain-slider-container', children=[
    # Create element to hide/show, in this case a slider
    html.Label('Select the resolution'),
    dcc.Slider(id='louvain-slider',
               min=0,
               max=5,
               step=0.1,
               value=1,
               marks={str(i): str(i) for i in np.arange(0, 5, 0.5)},
               )

], style={'display': 'block', 'textAlign': 'center', 'width': '40%', 'margin': 'auto'},
                          hidden=True)

sliders_components = html.Div([
    html.Div([
        html.Label('Year slider'),
        dcc.Slider(
            id='year-slider',
            min=1979,  # Extract minimum year from data
            max=2014,  # Extract maximum year from data
            value=1979,  # Set initial value to minimum year
            marks={str(year): str(year) for year in range(1979, 2015, 2)},
            step=1,
        ),
    ], style={'textAlign': 'center', 'width': '100%', 'margin': 'auto'}),
    html.Div([
        html.Label('Top % of relationship taken'),
        dcc.Slider(
            id='quantile-slider',
            min=0,
            max=100,
            value=20,
            marks={str(quantile): f"{quantile}%" for quantile in [0, 20, 40, 60, 80, 100]},
            step=1,
            tooltip={'placement': 'bottom', 'always_visible': True}
        ),
    ], style={'textAlign': 'center', 'width': '100%', 'margin': 'auto'}),
], style={'margin-bottom': '20px'})

select_components = html.Div([
    html.Div([
        html.H4('Measure displayed', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id="measure-selection",
            options=[
                {'label': 'No measure', 'value': 'none'},
                {'label': 'centrality-degree', 'value': 'centrality-degree'},
                {'label': 'centrality-eigenvector', 'value': 'centrality-eigenvector'},
                {'label': 'centrality-closeness', 'value': 'centrality-closeness'},
                # {'label': 'centrality-katz', 'value': 'centrality-katz'},
                {'label': 'centrality-betweenness', 'value': 'centrality-betweenness'},
                {'label': 'k-components', 'value': 'k-components'},
                {'label': 'clique', 'value': 'clique'},
                {'label': 'community-louvain', 'value': 'community-louvain'},
                {'label': 'community-greedy', 'value': 'community-greedy'},
                {'label': 'community-label', 'value': 'community-label'},
                {'label': 'dominating-set', 'value': 'dominating-set'},
                {'label': 'core-periphery', 'value': 'core-periphery'},
                {'label': 'pagerank', 'value': 'page-rank'}

            ],
            value='none',
        ),
    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.H4('Map displayed', style={'textAlign': 'center'}),
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
], style={'display': 'flex', 'justifyContent': 'center', 'width': '60%', 'margin': '0 auto'})

table_component = dash_table.DataTable(id="table", style_table={'width': '50%', 'margin': 'auto'},
                                       style_cell={'textAlign': 'center'}, )

app.layout = html.Div([
    html.H1("Interactive map", style={'textAlign': 'center'}, id="title"),

    html.Div([
        html.Div([
            # Make the graph larger
            graph_component,

            # Vertical column for the sliders and dropdowns
            html.Div([
                html.Div([
                    html.H2('Method selection', style={'textAlign': 'center'}),

                    dcc.Dropdown(
                        id="method-selection",
                        options=[
                            {"label": "Relevance weighted average", "value": "relevance_weighted_average"},
                            {'label': 'Sum', 'value': 'sum'},
                            {'label': 'Mean', 'value': 'mean'},
                            {'label': 'Mixed', 'value': 'mixed'},
                        ],
                        value='relevance_weighted_average',
                    ),
                    method_beta_value_slider,
                ], style={'width': '100%', 'display': 'inline-block', 'verticalAlign': 'top',
                          "padding-bottom": "25px"}),

                # Sliders
                html.Div([
                    html.H2('Map value options', style={'textAlign': 'center'}),
                    sliders_components,
                    k_components_slider,
                    louvain_slider,

                    # Select
                    select_components,

                ], style={'width': '100%', 'display': 'inline-block', 'verticalAlign': 'top',
                          "padding-bottom": "25px"}),

                # Add a vertical space
                html.Div(style={'height': '50px'}),
                # Add a table
                html.H3('Selection data table', style={'textAlign': 'center'}),
                table_component
            ], style={'textAlign': 'center', 'width': '60%', 'margin': 'auto', "padding": "20px"}),
            # add a border
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
    Output('table', 'columns'),
    Output('table', 'data'),
    Input('year-slider', 'value'),
    Input('quantile-slider', 'value'),
    Input("measure-selection", "value"),
    Input("map-selection", "value"),
    Input("k-slider", "value"),
    Input("louvain-slider", "value"),
    Input('method-selection', 'value'),
    Input('method-beta-slider', 'value'),
)
def display_map_interactive_plotly(year, quantile, measure, map_type, k_components, louvain_resolution,
                                   method_selection, method_beta):
    graph = load_graph_for(year, (1 - quantile / 100), map_type, method=method_selection, method_beta=method_beta)
    if measure == "none":
        fig = get_plotly_map(graph, self_loop=False)
        title = f"Interactive map for year {year} taking top {quantile}% relationship"
    else:
        fig = get_map_for_measure(graph, measure, k_components=k_components,
                                  louvain_resolution=louvain_resolution)
        title = f"Interactive map for year {year} taking top {quantile}% relationship event and measure {measure}"

    # CLustering
    average_clustering = nx.average_clustering(graph, weight='weight')

    #calculate small world
    # try:
    #     small_world = nx.algorithms.smallworld.sigma(graph, niter=1, nrand=1)
    # except Exception as e:
    #     small_world = "Not computable"

    df = pd.DataFrame({
        "Measure": ["Average clustering"],#, "Small worldness"],
        "Value": [average_clustering]#, small_world]
    })
    table_title = [{"name": i, "id": i} for i in df.columns]

    return fig, title, table_title, df.to_dict('records')


@app.callback(
    Output(component_id='k-slider-container', component_property='style'),
    [Input(component_id='measure-selection', component_property='value')]
)
def show_hide_element_k_components(measure):
    k_measures = ['k-components', 'core-periphery']
    if measure in k_measures:
        return {'display': 'block'}
    if measure not in k_measures:
        return {'display': 'none'}


@app.callback(
    Output(component_id='louvain-slider-container', component_property='style'),
    [Input(component_id='measure-selection', component_property='value')]
)
def show_hide_element_louvain(measure):
    louvain_measures = ['community-louvain']
    if measure in louvain_measures:
        return {'display': 'block'}
    if measure not in louvain_measures:
        return {'display': 'none'}


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
