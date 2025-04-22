"""
Module dédié à la gestion des schedules dans le dashboard.
Ce module séparé permet d'éviter de surcharger le fichier callbacks.py principal.
"""
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

def create_schedule_visualization_tab():
    """Crée l'onglet de visualisation des schedules."""
    return dbc.Tab([
        dbc.Row([
            dbc.Col([
                html.Label("Type de Schedule"),
                dcc.Dropdown(
                    id="schedule_type_selector",
                    options=[
                        {"label": "Occupation", "value": "occupants"},
                        {"label": "Éclairage", "value": "lighting_interior"},
                        {"label": "Cuisson", "value": "cooking_range"},
                        {"label": "Lave-vaisselle", "value": "dishwasher"},
                        {"label": "Lave-linge", "value": "clothes_washer"},
                        {"label": "Autres charges", "value": "plug_loads_other"},
                        {"label": "Eau chaude", "value": "hot_water_fixtures"},
                        {"label": "Télévision", "value": "plug_loads_tv"}
                    ],
                    value="occupants"
                ),
            ], width=4),
            dbc.Col([
                dbc.Button(
                    "Rafraîchir",
                    id="refresh_schedule_button",
                    color="secondary",
                    size="sm",
                    className="mt-4"
                ),
            ], width=2),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="schedule_visualization_graph")
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("Statistiques des Schedules", className="mt-3"),
                html.Div(id="schedule_statistics")
            ], width=12),
        ]),
    ], label="Schedules")

def create_schedule_config_panel():
    """Crée le panneau de configuration des schedules."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Configuration des Schedules", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Facteur d'occupation"),
                        dbc.Input(
                            id="occupancy_scale_input",
                            type="number",
                            min=-0.3,
                            max=0.3,
                            step=0.05,
                            value=0,
                            disabled=False
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Facteur d'éclairage"),
                        dbc.Input(
                            id="lighting_scale_input",
                            type="number",
                            min=-0.3,
                            max=0.3,
                            step=0.05,
                            value=0,
                            disabled=False
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Facteur des appareils"),
                        dbc.Input(
                            id="appliance_scale_input",
                            type="number",
                            min=-0.3,
                            max=0.3,
                            step=0.05,
                            value=0,
                            disabled=False
                        ),
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Diversité temporelle"),
                        dbc.Input(
                            id="temporal_diversity_input",
                            type="number",
                            min=0,
                            max=1.0,
                            step=0.1,
                            value=0,
                            disabled=False
                        ),
                        dbc.FormText("Diversifie les horaires entre archétypes (0 = aucune, 1 = maximale)")
                    ], width=6),
                ]),
            ], width=12),
        ]),
    ], id="schedule_config_container", style={"display": "none"})

# Callbacks à intégrer dans le fichier callbacks.py principal
def register_schedule_callbacks(app):
    """Enregistre les callbacks liés aux schedules."""
    
    @app.callback(
        Output("schedule_visualization_graph", "figure"),
        [Input("refresh_schedule_button", "n_clicks"),
         Input("schedule_type_selector", "value"),
         Input("simulation_results_store", "data")],
        prevent_initial_call=True
    )
    def update_schedule_visualization(n_clicks, schedule_type, store_data):
        """Mise à jour du graphique de visualisation des schedules."""
        if not store_data or not schedule_type:
            return generate_empty_figure("Aucune donnée de schedule disponible")
            
        try:
            # Rechercher les fichiers de schedule dans les résultats de simulation
            schedule_files = []
            if 'schedule_paths' in store_data:
                schedule_files = store_data['schedule_paths']
            else:
                # Chercher les fichiers de schedule dans le répertoire de simulation
                sim_dir = store_data.get('simulation_dir', '')
                if sim_dir:
                    sim_path = Path(sim_dir)
                    if sim_path.exists():
                        for sched_dir in sim_path.glob("**/schedules"):
                            schedule_files.extend(list(sched_dir.glob("*.csv")))
            
            if not schedule_files:
                return generate_empty_figure("Aucun schedule trouvé pour cette simulation")
                
            # Sélectionner un schedule pour visualisation (le premier par défaut)
            schedule_file = schedule_files[0] if isinstance(schedule_files[0], Path) else Path(schedule_files[0])
            if not schedule_file.exists():
                return generate_empty_figure(f"Fichier {schedule_file} introuvable")
                
            schedule_df = pd.read_csv(schedule_file)
            
            # Si le type demandé n'est pas disponible, prendre le premier type
            if schedule_type not in schedule_df.columns:
                if len(schedule_df.columns) > 0:
                    schedule_type = schedule_df.columns[0]
                else:
                    return generate_empty_figure("Schedule sans données")
            
            # Calculer le profil journalier moyen
            daily_profiles = []
            for day in range(min(365, len(schedule_df) // 24)):
                start_idx = day * 24
                end_idx = start_idx + 24
                if end_idx <= len(schedule_df):
                    daily = schedule_df[schedule_type][start_idx:end_idx].values
                    daily_profiles.append(daily)
            
            if not daily_profiles:
                return generate_empty_figure("Impossible de calculer les profils journaliers")
                
            avg_daily = np.mean(daily_profiles, axis=0)
            
            # Créer la figure
            fig = go.Figure()
            
            # Ajouter le profil journalier moyen
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=avg_daily,
                mode='lines',
                name='Profil journalier moyen',
                line=dict(color='blue', width=2)
            ))
            
            # Ajouter une plage pour montrer la variation
            if len(daily_profiles) > 1:
                lower = np.percentile(daily_profiles, 25, axis=0)
                upper = np.percentile(daily_profiles, 75, axis=0)
                
                fig.add_trace(go.Scatter(
                    x=list(range(24)) + list(range(24))[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.1)',
                    line=dict(color='rgba(0,0,255,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Mise en forme
            fig.update_layout(
                title=f"Profil journalier moyen pour '{schedule_type}'",
                xaxis_title="Heure",
                yaxis_title="Valeur",
                template="plotly_white",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f"{h}h" for h in range(24)]
                )
            )
            
            return fig
            
        except Exception as e:
            return generate_empty_figure(f"Erreur: {str(e)}")
    
    @app.callback(
        Output("schedule_statistics", "children"),
        [Input("refresh_schedule_button", "n_clicks"),
         Input("simulation_results_store", "data")],
        prevent_initial_call=True
    )
    def update_schedule_statistics(n_clicks, store_data):
        """Mise à jour des statistiques des schedules."""
        if not store_data:
            return "Aucune donnée disponible"
            
        try:
            # Vérifier si des statistiques de schedule sont disponibles
            if 'schedule_statistics' in store_data:
                stats = store_data['schedule_statistics']
                
                # Créer un tableau de statistiques
                table_header = [
                    html.Thead(html.Tr([
                        html.Th("Statistique"), 
                        html.Th("Valeur")
                    ]))
                ]
                
                rows = []
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        rows.append(html.Tr([
                            html.Td(stat_name),
                            html.Td(f"{stat_value:.3f}")
                        ]))
                
                table_body = [html.Tbody(rows)]
                
                return dbc.Table(
                    table_header + table_body,
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm"
                )
                
            # Si pas de stats dans le store, chercher dans les résultats de simulation
            return "Statistiques non disponibles pour cette simulation"
            
        except Exception as e:
            return f"Erreur lors du chargement des statistiques: {str(e)}"
    
    @app.callback(
        Output("schedule_config_container", "style"),
        Input("simulation_schedules", "value"),
        prevent_initial_call=True
    )
    def toggle_schedule_config(use_stochastic):
        """Affiche/masque le panneau de configuration des schedules."""
        if use_stochastic:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("schedule_config_output", "children"),
        [Input("occupancy_scale_input", "value"),
         Input("lighting_scale_input", "value"),
         Input("appliance_scale_input", "value"),
         Input("temporal_diversity_input", "value")],
        prevent_initial_call=True
    )
    def update_schedule_config(occupancy_scale, lighting_scale, appliance_scale, temporal_diversity):
        """Met à jour la configuration des schedules."""
        # Cette fonction ne fait que stocker les valeurs pour être utilisées lors du démarrage de la simulation
        return ""

def generate_empty_figure(message="Aucune donnée"):
    """Génère une figure vide avec un message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig
