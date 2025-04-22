"""
Layout definitions for the UBEM Quebec dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..config import config
from .schedule_module import create_schedule_visualization_tab, create_schedule_config_panel

def create_navbar() -> dbc.Navbar:
    """Create the navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Residence Quebec", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Simulation", href="/simulation", active="exact")),
                dbc.NavItem(dbc.NavLink("Sensitivity", href="/sensitivity", active="exact")),
                dbc.NavItem(dbc.NavLink("Metamodel", href="/metamodel", active="exact")),
                dbc.NavItem(dbc.NavLink("Hierarchical", href="/hierarchical", active="exact")),
            ]),
        ]),
        color="dark",
        dark=True,
    )

def create_simulation_controls() -> dbc.Card:
    """Create simulation control panel."""
    return dbc.Card([
        dbc.CardHeader("Simulation Controls"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Year"),
                    dcc.Dropdown(
                        id="simulation_year",
                        options=[
                            {"label": str(year), "value": year}
                            for year in range(2019, 2024)
                        ],
                        value=2023
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Scenario"),
                    dcc.Dropdown(
                        id="simulation_scenario",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "2035", "value": "2035"},
                            {"label": "2050", "value": "2050"}
                        ],
                        value="baseline"
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Schedules"),
                    dbc.RadioItems(
                        id="simulation_schedules",
                        options=[
                            {"label": "Standard", "value": False},
                            {"label": "Stochastic", "value": True}
                        ],
                        value=False,
                        inline=True
                    ),
                ], width=3),
                dbc.Col([
                    html.Br(),  # Align button with dropdowns
                    dbc.Button(
                        "Run Simulation",
                        id="run_simulation_button",
                        color="primary",
                        className="w-100"
                    ),
                ], width=3),
            ]),
            
            # Add future scenario selector
            dbc.Row([
                dbc.Col([
                    html.Label("Future Scenario (optional)"),
                    dcc.Dropdown(
                        id="future_scenario",
                        options=[
                            {"label": config.future_scenarios.SCENARIO_NAMES[scenario], 
                             "value": scenario}
                            for scenario in config.future_scenarios.SCENARIOS
                        ],
                        value=None,
                        placeholder="Select a future scenario (optional)"
                    ),
                ], width=12),
            ], className="mt-2"),
            
            # Panneau de configuration des schedules (affiché seulement si stochastique=True)
            create_schedule_config_panel(),
            
            # Div caché pour stocker le résultat de la configuration
            html.Div(id="schedule_config_output", style={"display": "none"})
        ]),
    ])

# Supprimé: Fonction create_calibration_controls() pour PSO+MCMC

def create_simulation_content() -> html.Div:
    """Create simulation page content."""
    return html.Div([
        dbc.Row([
            dbc.Col(create_simulation_controls(), width=12),
        ], className="mb-4"),
        
        # Provincial Consumption
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Provincial Energy Consumption"),
                    dbc.CardBody(
                        dcc.Graph(id="provincial_consumption_graph")
                    ),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Zone Map and Statistics
        dbc.Row([
            # Quebec Zone Map
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Geographic Visualization"),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Graph(id="zones_map"),
                                label="Zones Map"
                            ),
                            dbc.Tab(
                                dcc.Graph(id="mrc_map"),
                                label="MRC Map"
                            ),
                        ]),
                    ]),
                ]),
            ], width=6),
            
            # Zone Statistics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Area Statistics"),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Graph(id="zone_statistics_graph"),
                                label="By Zone"
                            ),
                            dbc.Tab(
                                dcc.Graph(id="mrc_statistics_graph"),
                                label="By MRC"
                            ),
                        ]),
                    ]),
                ]),
            ], width=6),
        ], className="mb-4"),
        
        # Load Profiles
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Load Profiles"),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Graph(id="daily_profile_graph"),
                                label="Daily"
                            ),
                            dbc.Tab(
                                dcc.Graph(id="weekly_profile_graph"),
                                label="Weekly"
                            ),
                            dbc.Tab(
                                dcc.Graph(id="seasonal_profile_graph"),
                                label="Seasonal"
                            ),
                            # Nouvel onglet pour la visualisation des schedules
                            create_schedule_visualization_tab(),
                        ]),
                    ]),
                ]),
            ], width=12),
        ]),
    ], id="simulation_content")

def create_calibration_content() -> html.Div:
    """Create calibration page content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Calibration PSO+MCMC"),
                    dbc.CardBody([
                        html.H4("Cette fonctionnalité a été supprimée"),
                        html.P("La calibration PSO+MCMC a été remplacée par des approches plus efficaces:"),
                        html.Ul([
                            html.Li("Analyse de sensibilité (onglet Sensitivity)"),
                            html.Li("Calibration par métamodèle (onglet Metamodel)")
                        ])
                    ]),
                ]),
            ], width=12),
        ]),
    ], id="calibration_content")

def create_sensitivity_content() -> html.Div:
    """Create sensitivity analysis page content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analyse de sensibilité"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Année"),
                                dcc.Dropdown(
                                    id="sensitivity_year",
                                    options=[
                                        {"label": str(year), "value": year}
                                        for year in range(2019, 2024)
                                    ],
                                    value=2022
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("Trajectoires"),
                                dbc.Input(
                                    id="sensitivity_trajectories",
                                    type="number",
                                    min=5,
                                    max=20,
                                    step=1,
                                    value=10
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Br(),  
                                dbc.Button(
                                    "Lancer analyse",
                                    id="run_sensitivity_button",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], width=6),
                        ]),
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Sensitivity results
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Résultats d'analyse de sensibilité"),
                    dbc.CardBody([
                        dcc.Graph(id="sensitivity_graph")
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Parameter details
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Détails des paramètres"),
                    dbc.CardBody([
                        html.Div(id="sensitivity_details")
                    ]),
                ]),
            ], width=12),
        ]),
    ], id="sensitivity_content")

def create_metamodel_content() -> html.Div:
    """Create metamodel calibration page content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Calibration par Métamodèle"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Année"),
                                dcc.Dropdown(
                                    id="metamodel_year",
                                    options=[
                                        {"label": str(year), "value": year}
                                        for year in range(2019, 2024)
                                    ],
                                    value=2022
                                ),
                            ], width=2),
                            dbc.Col([
                                html.Label("Taille DOE"),
                                dbc.Input(
                                    id="doe_size",
                                    type="number",
                                    min=50,
                                    max=200,
                                    step=10,
                                    value=100
                                ),
                            ], width=2),
                            dbc.Col([
                                html.Label("Type de métamodèle"),
                                dcc.Dropdown(
                                    id="metamodel_type",
                                    options=[
                                        {"label": "Gaussian Process (Krigeage)", "value": "gpr"},
                                        {"label": "Random Forest", "value": "rf"}
                                    ],
                                    value="gpr"
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("Horaires"),
                                dbc.RadioItems(
                                    id="metamodel_schedules",
                                    options=[
                                        {"label": "Standards", "value": False},
                                        {"label": "Stochastiques", "value": True}
                                    ],
                                    value=False,
                                    inline=True
                                ),
                            ], width=2),
                            dbc.Col([
                                html.Br(),  # Align button with inputs
                                dbc.Button(
                                    "Lancer Calibration",
                                    id="run_metamodel_button",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], width=3),
                        ]),
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # DOE Results
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Résultats du Design d'Expériences"),
                    dbc.CardBody([
                        dcc.Graph(id="doe_results_graph")
                    ]),
                ]),
            ], width=6),
            
            # Parameter Importance
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Importance des Paramètres"),
                    dbc.CardBody([
                        dcc.Graph(id="parameter_importance_graph")
                    ]),
                ]),
            ], width=6),
        ], className="mb-4"),
        
        # Calibration Results
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Résultats de Calibration"),
                    dbc.CardBody([
                        dcc.Graph(id="metamodel_results_graph")
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Parameter Details
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Détails des Paramètres Calibrés"),
                    dbc.CardBody([
                        html.Div(id="metamodel_parameter_details")
                    ]),
                ]),
            ], width=12),
        ]),
    ], id="metamodel_content")

def create_hierarchical_content() -> html.Div:
    """Create hierarchical calibration page content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Calibration Hiérarchique Multi-niveaux"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Année"),
                                dcc.Dropdown(
                                    id="hierarchical_year",
                                    options=[
                                        {"label": str(year), "value": year}
                                        for year in range(2019, 2024)
                                    ],
                                    value=2022
                                ),
                            ], width=2),
                            dbc.Col([
                                html.Label("Liste de paramètres"),
                                dcc.Dropdown(
                                    id="hierarchical_params",
                                    options=[
                                        {"label": "Tous les paramètres", "value": "all"},
                                        {"label": "Top paramètres (sensibilité)", "value": "sensitivity"},
                                        {"label": "Personnalisé", "value": "custom"}
                                    ],
                                    value="all"
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("Horaires"),
                                dbc.RadioItems(
                                    id="hierarchical_schedules",
                                    options=[
                                        {"label": "Standards", "value": False},
                                        {"label": "Stochastiques", "value": True}
                                    ],
                                    value=False,
                                    inline=True
                                ),
                            ], width=2),
                            dbc.Col([
                                html.Br(),  # Align button with inputs
                                dbc.Button(
                                    "Lancer Calibration",
                                    id="run_hierarchical_button",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], width=5),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="custom_params_container", style={"display": "none"}, children=[
                                    html.Label("Paramètres personnalisés (séparés par des virgules)"),
                                    dbc.Input(
                                        id="custom_params_input",
                                        type="text",
                                        placeholder="infiltration_rate,wall_rvalue,ceiling_rvalue,window_ufactor"
                                    )
                                ])
                            ], width=12)
                        ], className="mt-3")
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Metrics Evolution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Évolution des métriques par niveau"),
                    dbc.CardBody([
                        dcc.Graph(id="hierarchical_metrics_graph")
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Calibrated Parameters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Paramètres calibrés par niveau"),
                    dbc.CardBody([
                        dcc.Graph(id="hierarchical_params_graph")
                    ]),
                ]),
            ], width=6),
            
            # Level Details
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Détails des niveaux"),
                    dbc.CardBody([
                        html.Div(id="hierarchical_level_details")
                    ]),
                ]),
            ], width=6),
        ], className="mb-4"),
        
        # Final Results
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Résultats finaux"),
                    dbc.CardBody([
                        dcc.Graph(id="hierarchical_results_graph")
                    ]),
                ]),
            ], width=12),
        ]),
    ], id="hierarchical_content")

def create_layout() -> html.Div:
    """Create the main dashboard layout."""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        create_navbar(),
        dbc.Container([
            html.Div(id='page_content', className="mt-4"),
            
            # Loading overlays
            dcc.Loading(
                id="simulation_loading",
                type="circle",
                children=html.Div(id="simulation_loading_output")
            ),
            dcc.Loading(
                id="calibration_loading",
                type="circle",
                children=html.Div(id="calibration_loading_output")
            ),
            dcc.Loading(
                id="sensitivity_loading",
                type="circle",
                children=html.Div(id="sensitivity_loading_output")
            ),
            dcc.Loading(
                id="metamodel_loading",
                type="circle",
                children=html.Div(id="metamodel_loading_output")
            ),
            dcc.Loading(
                id="hierarchical_loading",
                type="circle",
                children=html.Div(id="hierarchical_loading_output")
            ),
            
            # Store components for state management
            dcc.Store(id='simulation_results_store'),
            dcc.Store(id='calibration_results_store'),
            dcc.Store(id='sensitivity_results_store'),
            dcc.Store(id='metamodel_results_store'),
            dcc.Store(id='hierarchical_results_store'),
        ], fluid=True),
    ])
