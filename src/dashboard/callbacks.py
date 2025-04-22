"""
Callback definitions for the UBEM Quebec dashboard.
"""
from dash import Input, Output, State, callback, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Any
import json
from io import StringIO
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

from ..config import config
from ..utils import process_hydro_data, ensure_dir
from ..managers.simulation_manager import SimulationManager
from ..managers.calibration_manager import CalibrationManager
from ..managers.aggregation_manager import AggregationManager
from .layouts import create_simulation_content, create_calibration_content, create_sensitivity_content, create_metamodel_content, create_hierarchical_content
    

def save_figure(fig, path: Path, filename: str):
    """
    Save a Plotly figure as an HTML file.
    
    Args:
        fig: Plotly figure to save
        path: Directory path to save to
        filename: Name of the file (without extension)
    """
    try:
        ensure_dir(path)
        fig.write_html(path / f"{filename}.html")
        print(f"Figure saved to {path / f'{filename}.html'}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")

# Initialize managers
simulation_manager = SimulationManager()
aggregation_manager = AggregationManager(simulation_manager)
calibration_manager = CalibrationManager()

def register_callbacks(app):
    """Register all callbacks for the dashboard."""
    

    # Page navigation
    @callback(
        Output('page_content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        if pathname == '/calibration':
            return create_calibration_content()
        elif pathname == '/sensitivity':
            return create_sensitivity_content()
        elif pathname == '/metamodel':
            return create_metamodel_content()
        elif pathname == '/hierarchical':
            return create_hierarchical_content()
        return create_simulation_content()  # Default to simulation
    
    # Simulation callbacks
    @callback(
        [Output('simulation_results_store', 'data'),
         Output('simulation_loading_output', 'children')],
        Input('run_simulation_button', 'n_clicks'),
        [State('simulation_year', 'value'),
         State('simulation_scenario', 'value'),
         State('simulation_schedules', 'value'),
         State('future_scenario', 'value')],  # Ajouter cette ligne
        prevent_initial_call=True
    )
    def run_simulation(n_clicks, year, scenario, use_stochastic, future_scenario):
        if not n_clicks:
            return None, ""
            
        # Check if a future scenario is selected (priority over regular scenario)
        if future_scenario:
            # Run simulation for future scenario
            simulation_data = simulation_manager.run_parallel_simulations_for_scenario(
                scenario_key=future_scenario
            )
            
            if not simulation_data or (isinstance(simulation_data, dict) and 'results' in simulation_data and not simulation_data['results']):
                return None, f"Simulation failed for future scenario {future_scenario}"
                
            # Get year from scenario
            year = config.future_scenarios.get_scenario_year(future_scenario)
            scenario = future_scenario  # Use future scenario key as scenario name
            
            # Get simulation directory
            simulation_dir = config.paths['output'].get_simulation_dir(year, scenario)
            
            # Aggregate results - passer la structure complète avec les archétypes transformés
            provincial_results, zonal_results, mrc_results = (
                aggregation_manager.aggregate_results(
                    year, simulation_data, scenario
                )
            )
            
            # Adapter pour extraire les résultats si nécessaire
            sim_results = simulation_data['results'] if isinstance(simulation_data, dict) and 'results' in simulation_data else simulation_data
            
        else:
            # Regular simulation
            results = simulation_manager.run_parallel_simulations(
                year=year,
                scenario=scenario,
                use_stochastic_schedules=use_stochastic,
                cleanup_after=False
            )
            
            if not results:
                return None, "Simulation failed"
                
            # Get simulation directory
            simulation_dir = config.paths['output'].get_simulation_dir(year, scenario)
            
            # Aggregate results
            provincial_results, zonal_results, mrc_results = (
                aggregation_manager.aggregate_results(
                    year, results, scenario
                )
            )
        
        mrc_summaries = {}
        for mrc, results in mrc_results.items():
            # Extraire seulement les métriques clés pour chaque MRC
            mrc_summaries[str(mrc)] = {
                'total_consumption': results['Fuel Use: Electricity: Total'].sum(),
                'peak_demand': results['Fuel Use: Electricity: Total'].max(),
                'average_demand': results['Fuel Use: Electricity: Total'].mean(),
                'mrc_id': mrc,
            }
            
            # Ajouter le nom de la MRC si disponible
            try:
                mrc_dir = simulation_dir / 'mrc' / f'mrc_{mrc}'
                summary_path = mrc_dir / 'summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        mrc_summaries[str(mrc)]['mrc_name'] = summary.get('mrc_name', mrc)
            except:
                mrc_summaries[str(mrc)]['mrc_name'] = mrc


        # Limiter à 15 MRCs pour la performance
        max_mrcs_to_store = 15
        top_mrc_items = list(mrc_results.items())[:max_mrcs_to_store]
        # Prepare data for storage
        store_data = {
            'provincial': provincial_results.to_json(date_format='iso'),
            'zonal': {
                str(zone): results.to_json(date_format='iso')
                for zone, results in zonal_results.items()
            },
            'mrc': {
                str(mrc): results.to_json(date_format='iso')
                for mrc, results in top_mrc_items  # Limiter à 15 MRCs
            },
            'mrc_summaries': mrc_summaries, 
            'year': year,
            'scenario': scenario,
            'stochastic': use_stochastic,
            'simulation_dir': str(simulation_dir),
            'is_future_scenario': bool(future_scenario)  # Ajouter cette ligne
        }
        
        # Ajouter les chemins des fichiers de schedule si disponibles
        if use_stochastic or future_scenario:  # Les scénarios futurs utilisent toujours des schedules stochastiques
            schedule_files = []
            # Chercher les fichiers de schedule dans le répertoire de simulation
            if simulation_dir.exists():
                for sched_dir in simulation_dir.glob("**/schedules"):
                    schedule_files.extend(list(str(f) for f in sched_dir.glob("*.csv")))
                
            if schedule_files:
                store_data['schedule_paths'] = schedule_files
                print(f"Ajouté {len(schedule_files)} fichiers de schedules au store")

        print("Traitement terminé, données prêtes pour le dashboard")
        
        return store_data, ""
    
    # Supprimé: Callbacks de calibration PSO+MCMC
    
    # Provincial consumption plot
    @callback(
        Output('provincial_consumption_graph', 'figure'),
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_provincial_consumption(store_data):
        if not store_data:
            return go.Figure()
            
        # Load provincial results
        df = pd.read_json(StringIO(store_data['provincial']))
        
        # Create figure
        fig = go.Figure()
        
        # Add end uses as stacked area
        end_use_cols = [col for col in df.columns if 'End Use: Electricity' in col]
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            # Set default visibility for heating and cooling
            visible = True if name in ['Heating', 'Cooling'] else 'legendonly'
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=name,
                stackgroup='end_uses',  # This creates the stacked effect
                visible=visible,
                hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
            ))

        # Add simulated total consumption
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Fuel Use: Electricity: Total'],
            name='Simulated Total',
            line=dict(color='red', width=2)
        ))
        
        # Load and process Hydro-Quebec data for comparison
        try:
            input_paths = config.paths['input']
            hydro_file = input_paths.get_hydro_file(store_data['year'])
            if not hydro_file.exists():
                raise FileNotFoundError(f"Hydro-Quebec data file not found: {hydro_file}")
            hydro_data = pd.read_csv(hydro_file)
            hydro_data = process_hydro_data(hydro_data, store_data['year'])
            
            # Add Hydro-Quebec data
            fig.add_trace(go.Scatter(
                x=hydro_data.index,
                y=hydro_data['energie_sum_secteur'],
                name='Measured (Hydro-Quebec)',
                line=dict(color='blue', width=2)
            ))
        except Exception as e:
            print(f"Warning: Could not load Hydro-Quebec data: {str(e)}")
        
        
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            title=f"Provincial Energy Consumption ({store_data['year']} - {store_data['scenario']} - {schedule_type} Schedules)",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Save figure to file
        try:
            year = store_data['year']
            scenario = store_data['scenario']
            figures_dir = config.paths['output'].get_simulation_figures_dir(year, scenario)
            save_figure(fig, figures_dir, 'provincial_consumption')
        except Exception as e:
            print(f"Error saving provincial consumption figure: {str(e)}")
        
        return fig
    
    # Zones map
    @callback(
        Output('zones_map', 'figure'),
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_zones_map(store_data):
        if not store_data:
            return go.Figure()
            
        # Load zone geometry
        gdf = gpd.read_file(config.paths['input'].ZONES_SHAPEFILE)
        
        # Calculate total consumption by zone
        zone_consumption = {}
        for zone, results_json in store_data['zonal'].items():
            df = pd.read_json(StringIO(results_json))
            zone_consumption[int(zone)] = df['Fuel Use: Electricity: Total'].sum()
        
        # Add consumption to GeoDataFrame
        gdf['consumption'] = gdf['weather_zo'].map(zone_consumption)
        
        # Create choropleth map using MapLibre
        fig = px.choropleth_mapbox(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color='consumption',
            color_continuous_scale='Viridis',
            hover_data=['weather_zo'],
            labels={'consumption': 'Annual Consumption (kWh)'},
            mapbox_style="carto-positron"  # Use Carto basemap
        )
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            mapbox=dict(
                zoom=5,
                center=dict(lat=47.5, lon=-72)  # Center on Quebec
            )
        )
        
        fig.update_layout(
            title=f"Zone Consumption Map ({store_data['year']} - {store_data['scenario']} - {schedule_type} Schedules)",
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        # Save figure to file
        try:
            year = store_data['year']
            scenario = store_data['scenario']
            figures_dir = config.paths['output'].get_simulation_figures_dir(year, scenario)
            save_figure(fig, figures_dir, 'zones_map')
        except Exception as e:
            print(f"Error saving zones map figure: {str(e)}")
        
        return fig
    
    # Zone statistics
    @callback(
        Output('zone_statistics_graph', 'figure'),
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_zone_statistics(store_data):
        if not store_data:
            return go.Figure()
            
        # Calculate statistics by zone
        stats = []
        for zone, results_json in store_data['zonal'].items():
            df = pd.read_json(StringIO(results_json))
            stats.append({
                'zone': int(zone),
                'total_consumption': df['Fuel Use: Electricity: Total'].sum(),
                'peak_demand': df['Fuel Use: Electricity: Total'].max(),
                'average_demand': df['Fuel Use: Electricity: Total'].mean()
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each metric
        for col in ['total_consumption', 'peak_demand', 'average_demand']:
            fig.add_trace(go.Bar(
                x=stats_df['zone'],
                y=stats_df[col],
                name=col.replace('_', ' ').title()
            ))
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            title=f"Zone Statistics ({schedule_type} Schedules)",
            xaxis_title="Zone",
            yaxis_title="Energy (kWh)",
            barmode='group',
            showlegend=True,
            template='plotly_white'
        )
        
        # Save figure to file
        try:
            year = store_data['year']
            scenario = store_data['scenario']
            figures_dir = config.paths['output'].get_simulation_figures_dir(year, scenario)
            save_figure(fig, figures_dir, 'zone_statistics')
        except Exception as e:
            print(f"Error saving zone statistics figure: {str(e)}")
        
        return fig
    
    # MRC map
    @callback(
        Output('mrc_map', 'figure'),
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_mrc_map(store_data):
        if not store_data or 'mrc_summaries' not in store_data:
            return go.Figure()
        
        # Vérifier si nous avons des données MRC
        if not store_data['mrc_summaries']:
            # Créer un graphique vide avec message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée MRC disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        try:
            # Load MRC geometry
            gdf = gpd.read_file(config.paths['input'].ZONES_SHAPEFILE)
            
            # Calculate total consumption by MRC
            # mrc_consumption = {}
            # for mrc, results_json in store_data['mrc'].items():
            #     df = pd.read_json(StringIO(results_json))
            #     mrc_consumption[mrc] = df['Fuel Use: Electricity: Total'].sum()

            mrc_consumption = {
                mrc: data['total_consumption'] 
                for mrc, data in store_data['mrc_summaries'].items()
            }
            
            # Add consumption to GeoDataFrame
            gdf['consumption'] = gdf['CDUID'].map(lambda x: mrc_consumption.get(x, 0))
            
            # Create choropleth map using MapLibre
            fig = px.choropleth_mapbox(
                gdf,
                geojson=gdf.geometry,
                locations=gdf.index,
                color='consumption',
                color_continuous_scale='Viridis',
                hover_data=['CDNAME'],
                labels={'consumption': 'Annual Consumption (kWh)'},
                mapbox_style="carto-positron"  # Use Carto basemap
            )
            
            # Update layout
            schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
            fig.update_layout(
                mapbox=dict(
                    zoom=5,
                    center=dict(lat=47.5, lon=-72)  # Center on Quebec
                ),
                title=f"MRC Consumption Map ({store_data['year']} - {store_data['scenario']} - {schedule_type} Schedules)",
                margin={"r":0,"t":30,"l":0,"b":0}
            )
            
            # Save figure to file
            try:
                year = store_data['year']
                scenario = store_data['scenario']
                figures_dir = config.paths['output'].get_simulation_figures_dir(year, scenario)
                save_figure(fig, figures_dir, 'mrc_map')
            except Exception as e:
                print(f"Error saving mrc map figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur lors du chargement des données MRC: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig

    # MRC statistics
    @callback(
        Output('mrc_statistics_graph', 'figure'),
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_mrc_statistics(store_data):
        if not store_data or 'mrc' not in store_data:
            return go.Figure()
                
        # Vérifier si nous avons des données MRC
        if not store_data['mrc']:
            # Créer un graphique vide avec message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée MRC disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Limiter le nombre de MRCs à traiter pour éviter une surcharge
        max_mrcs = 15  # Limiter à 15 MRCs pour les performances
        mrc_items = list(store_data['mrc'].items())[:max_mrcs]
                    
        # Calculate statistics by MRC
        stats = []
        for mrc, results_json in mrc_items:
            df = pd.read_json(StringIO(results_json))
            
            # Get MRC name from simulation dir if available
            mrc_name = mrc
            try:
                sim_dir = Path(store_data['simulation_dir'])
                summary_path = sim_dir / 'mrc' / f'mrc_{mrc}' / 'summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        if 'mrc_name' in summary:
                            mrc_name = summary['mrc_name']
            except:
                pass
                
            stats.append({
                'mrc': mrc,
                'mrc_name': mrc_name,
                'total_consumption': df['Fuel Use: Electricity: Total'].sum(),
                'peak_demand': df['Fuel Use: Electricity: Total'].max(),
                'average_demand': df['Fuel Use: Electricity: Total'].mean()
            })
        
        # Vérifier si nous avons des statistiques
        if not stats:
            # Créer un graphique vide avec message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune statistique MRC disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        stats_df = pd.DataFrame(stats)
        
        # Créer la figure
        fig = go.Figure()
        
        # Add bars for each metric - verify column existence
        if 'mrc_name' not in stats_df.columns:
            # Use 'mrc' as fallback
            x_column = 'mrc'
        else:
            x_column = 'mrc_name'
            
        for col in ['total_consumption', 'peak_demand', 'average_demand']:
            if col in stats_df.columns:
                fig.add_trace(go.Bar(
                    x=stats_df[x_column],
                    y=stats_df[col],
                    name=col.replace('_', ' ').title()
                ))
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            title=f"MRC Statistics ({schedule_type} Schedules)",
            xaxis_title="MRC",
            yaxis_title="Energy (kWh)",
            barmode='group',
            showlegend=True,
            template='plotly_white',
            xaxis={'categoryorder':'total descending'}
        )
        
        # Save figure to file (only if we have data)
        try:
            if not stats_df.empty:
                year = store_data['year']
                scenario = store_data['scenario']
                figures_dir = config.paths['output'].get_simulation_figures_dir(year, scenario)
                save_figure(fig, figures_dir, 'mrc_statistics')
        except Exception as e:
            print(f"Error saving mrc statistics figure: {str(e)}")
        
        return fig
    
    # Load profiles
    @callback(
        [Output('daily_profile_graph', 'figure'),
         Output('weekly_profile_graph', 'figure'),
         Output('seasonal_profile_graph', 'figure')],
        Input('simulation_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_load_profiles(store_data):
        if not store_data:
            return go.Figure(), go.Figure(), go.Figure()
            
        # Load provincial results
        df = pd.read_json(StringIO(store_data['provincial']))
        df.index = pd.to_datetime(df.index)
        
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        
        # Daily profile with end uses
        daily_fig = go.Figure()
        
        # Get end use columns
        end_use_cols = [col for col in df.columns if 'End Use: Electricity' in col]
        
        # Add each end use as stacked area
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            daily = df.groupby(df.index.hour)[col].mean()

            # visible = True if name in ['Heating', 'Cooling'] else 'legendonly'

            daily_fig.add_trace(go.Scatter(
                x=daily.index,
                y=daily.values,
                name=name,
                # visible=visible,
                stackgroup='end_uses',
                hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
            ))
        
        daily_fig.update_layout(
            title=f"Average Daily Load Profile ({schedule_type} Schedules)",
            xaxis_title="Hour",
            yaxis_title="Energy (kWh)",
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        # Weekly profile with end uses
        weekly_fig = go.Figure()
        
        # Calculate weekly profile for each end use
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            weekly = df.groupby([df.index.dayofweek, df.index.hour])[col].mean().unstack()
            
            # Set default visibility for heating and cooling
            visible = True if name in ['Heating', 'Cooling'] else 'legendonly'
            
            # Add trace for each day
            for day in range(7):
                day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]
                weekly_fig.add_trace(go.Scatter(
                    x=list(range(24)),  # Hours in a day
                    y=weekly.iloc[day],
                    name=f"{name} ({day_name})",
                    stackgroup='end_uses',
                    visible=visible,
                    hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
                ))
        
        weekly_fig.update_layout(
            title=f"Weekly Load Profile ({schedule_type} Schedules)",
            xaxis_title="Hour",
            yaxis_title="Energy (kWh)",
            template='plotly_white',
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(
                ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                tickvals=[12, 36, 60, 84, 108, 132, 156]
            )
        )
        
        # Seasonal profile with end uses
        seasonal_fig = go.Figure()
        
        # Calculate seasonal profile for each end use
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            seasonal = df.groupby([df.index.month, df.index.hour])[col].mean().unstack()
            
            # Set default visibility for heating and cooling
            visible = True if name in ['Heating', 'Cooling'] else 'legendonly'
            
            # Add trace for each month
            for month in range(12):
                month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
                seasonal_fig.add_trace(go.Scatter(
                    x=list(range(24)),  # Hours in a day
                    y=seasonal.iloc[month],
                    name=f"{name} ({month_name})",
                    stackgroup='end_uses',
                    visible=visible,
                    hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
                ))
        
        seasonal_fig.update_layout(
            title=f"Seasonal Load Profile ({schedule_type} Schedules)",
            xaxis_title="Hour",
            yaxis_title="Energy (kWh)",
            template='plotly_white',
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                tickvals=[12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276]
            )
        )
        
        # Save figures to files
        try:
            year = store_data['year']
            scenario = store_data['scenario']
            profiles_dir = config.paths['output'].get_simulation_profiles_dir(year, scenario)
            
            # Save daily profile
            save_figure(daily_fig, profiles_dir, 'daily_profile')
            
            # Save weekly profile
            save_figure(weekly_fig, profiles_dir, 'weekly_profile')
            
            # Save seasonal profile
            save_figure(seasonal_fig, profiles_dir, 'seasonal_profile')
            
        except Exception as e:
            print(f"Error saving profile figures: {str(e)}")
        
        return daily_fig, weekly_fig, seasonal_fig
    
    # Supprimé: Graphique des résultats de calibration PSO+MCMC
    
    # Supprimé: Graphiques de performance saisonnière pour PSO+MCMC
    
    # Supprimé: Graphiques de diagnostics de convergence pour MCMC
    
    # Supprimé: Graphique de validation croisée pour PSO+MCMC
    
    # Supprimé: Graphique d'évolution des métriques pour PSO+MCMC
    
    # Supprimé: Graphique de distribution des erreurs pour PSO+MCMC
    
    # Supprimé: Graphique de performance des pics pour PSO+MCMC
        
    # Sensitivity analysis callbacks
    @callback(
        [Output('sensitivity_results_store', 'data'),
         Output('sensitivity_loading_output', 'children')],
        Input('run_sensitivity_button', 'n_clicks'),
        [State('sensitivity_year', 'value'),
         State('sensitivity_trajectories', 'value')],
        prevent_initial_call=True
    )
    def run_sensitivity_analysis(n_clicks, year, n_trajectories):
        if not n_clicks:
            return None, ""
            
        try:
            # Validate inputs
            if not n_trajectories or not isinstance(n_trajectories, (int, float)):
                return None, "Veuillez entrer un nombre de trajectoires"
            n_trajectories = int(n_trajectories)
            
            # Log sensitivity analysis start
            print(f"Démarrage de l'analyse de sensibilité:")
            print(f"  Année: {year}")
            print(f"  Trajectoires: {n_trajectories}")
            
            # Run sensitivity analysis
            results = calibration_manager.run_sensitivity_analysis(
                param_subset=None,  # Use all parameters
                n_trajectories=n_trajectories,
                year=year
            )
            
            # Check results
            if not results:
                return None, "Analyse de sensibilité échouée: Pas de résultats"
            
            print(f"Analyse de sensibilité terminée avec succès")
            return results, ""
            
        except Exception as e:
            error_msg = str(e)
            print(f"Erreur d'analyse de sensibilité: {error_msg}")
            return None, f"Analyse de sensibilité échouée: {error_msg}"
    
    # Sensitivity graph
    @callback(
        Output('sensitivity_graph', 'figure'),
        Input('sensitivity_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_sensitivity_graph(store_data):
        if not store_data or 'param_sensitivity' not in store_data:
            return go.Figure()
            
        # Extract parameter sensitivity data
        param_sensitivity = store_data['param_sensitivity']
        
        # Sort by mu_star (influence)
        param_sensitivity = sorted(param_sensitivity, key=lambda x: x['mu_star'], reverse=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add mu_star (influence) bars
        fig.add_trace(go.Bar(
            x=[p['display_name'] for p in param_sensitivity],
            y=[p['mu_star'] for p in param_sensitivity],
            name='Influence (μ*)',
            marker_color='blue'
        ))
        
        # Add sigma (non-linearity) as line
        fig.add_trace(go.Scatter(
            x=[p['display_name'] for p in param_sensitivity],
            y=[p['sigma'] for p in param_sensitivity],
            name='Non-linéarité (σ)',
            mode='lines+markers',
            marker=dict(color='red'),
            line=dict(width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Analyse de sensibilité des paramètres",
            xaxis_title="Paramètre",
            yaxis_title="Valeur",
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save figure to file
        try:
            # Si les données viennent d'une analyse récente, utiliser le répertoire de cette analyse
            if store_data and 'campaign_dir' in store_data:
                campaign_dir = Path(store_data['campaign_dir'])
                sensitivity_dir = campaign_dir / 'sensitivity'
            else:
                # Sinon, chercher le répertoire de la dernière analyse de sensibilité
                sensitivity_files = []
                for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                    if year_dir.is_dir():
                        for campaign_dir in year_dir.glob('*'):
                            if campaign_dir.is_dir():
                                sensitivity_file = campaign_dir / 'sensitivity' / 'sensitivity_results.json'
                                if sensitivity_file.exists():
                                    sensitivity_files.append((sensitivity_file.parent, campaign_dir.stat().st_mtime))
                
                if sensitivity_files:
                    # Trier par date de modification (le plus récent en premier)
                    sensitivity_files.sort(key=lambda x: x[1], reverse=True)
                    sensitivity_dir = sensitivity_files[0][0]
                else:
                    # Fallback: créer un nouveau répertoire
                    year = store_data.get('year', 2022) if store_data else 2022
                    campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sensitivity_dir = config.paths['output'].CALIBRATION / str(year) / campaign_id / 'sensitivity'
            
            ensure_dir(sensitivity_dir)
            save_figure(fig, sensitivity_dir, 'sensitivity_analysis')
            print(f"Figure de sensibilité sauvegardée dans {sensitivity_dir}")
        except Exception as e:
            print(f"Error saving sensitivity figure: {str(e)}")
        
        return fig
    
    # Sensitivity details
    @callback(
        Output('sensitivity_details', 'children'),
        [Input('sensitivity_results_store', 'data'),
         Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def update_sensitivity_details(store_data, pathname):
        # Si des données sont déjà dans le store, les utiliser
        if store_data and 'param_sensitivity' in store_data:
            param_sensitivity = store_data['param_sensitivity']
            param_sensitivity = sorted(param_sensitivity, key=lambda x: x['mu_star'], reverse=True)
        else:
            # Sinon, essayer de charger les résultats précédents
            try:
                # Chercher le fichier de résultats le plus récent
                sensitivity_files = []
                for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                    if year_dir.is_dir():
                        for campaign_dir in year_dir.glob('*'):
                            if campaign_dir.is_dir():
                                sensitivity_file = campaign_dir / 'sensitivity' / 'sensitivity_results.json'
                                if sensitivity_file.exists():
                                    sensitivity_files.append((sensitivity_file, campaign_dir.stat().st_mtime))
                
                if not sensitivity_files:
                    return html.P("Aucun résultat d'analyse de sensibilité disponible.")
                
                # Trier par date de modification (le plus récent en premier)
                sensitivity_files.sort(key=lambda x: x[1], reverse=True)
                latest_file = sensitivity_files[0][0]
                
                # Charger les résultats
                with open(latest_file, 'r') as f:
                    param_sensitivity = json.load(f)
                
                print(f"Chargé les résultats de sensibilité depuis {latest_file}")
            except Exception as e:
                print(f"Erreur lors du chargement des résultats de sensibilité: {str(e)}")
                return html.P("Aucun résultat d'analyse de sensibilité disponible.")
        
        # Create table
        table_header = [
            html.Thead(html.Tr([
                html.Th("Rang"),
                html.Th("Paramètre"),
                html.Th("Influence (μ*)"),
                html.Th("Non-linéarité (σ)"),
                html.Th("Intervalle de confiance")
            ]))
        ]
        
        rows = []
        for param in param_sensitivity:
            row = html.Tr([
                html.Td(param['rank']),
                html.Td(param['display_name']),
                html.Td(f"{param['mu_star']:.4f}"),
                html.Td(f"{param['sigma']:.4f}"),
                html.Td(f"±{param['mu_star_conf']:.4f}")
            ])
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)
        
        # Add recommendations
        recommendations = html.Div([
            html.H5("Paramètres recommandés pour calibration"),
            html.P("Basé sur l'analyse de sensibilité, les paramètres suivants sont les plus influents:"),
            html.Ol([
                html.Li([
                    html.Strong(f"{param['display_name']}"), 
                    f" (influence: {param['mu_star']:.4f})"
                ]) for param in param_sensitivity[:5]
            ])
        ])
        
        # Add interpretation
        interpretation = html.Div([
            html.H5("Interprétation"),
            html.Ul([
                html.Li([
                    html.Strong("Influence (μ*)"), 
                    ": Plus cette valeur est élevée, plus le paramètre a d'impact sur les résultats"
                ]),
                html.Li([
                    html.Strong("Non-linéarité (σ)"), 
                    ": Une valeur élevée indique des effets non-linéaires ou des interactions fortes avec d'autres paramètres"
                ])
            ])
        ])
        
        return html.Div([table, html.Hr(), recommendations, html.Hr(), interpretation])
    
    # Metamodel calibration callbacks
    @callback(
        [Output('metamodel_results_store', 'data'),
         Output('metamodel_loading_output', 'children')],
        Input('run_metamodel_button', 'n_clicks'),
        [State('metamodel_year', 'value'),
         State('doe_size', 'value'),
         State('metamodel_type', 'value'),
         State('metamodel_schedules', 'value')],
        prevent_initial_call=True
    )
    def run_metamodel_calibration(n_clicks, year, doe_size, metamodel_type, use_stochastic):
        if not n_clicks:
            return None, ""
            
        try:
            # Validate inputs
            if not doe_size or not isinstance(doe_size, (int, float)):
                return None, "Veuillez entrer une taille de DOE"
            doe_size = int(doe_size)
            
            # Log metamodel calibration start
            print(f"Démarrage de la calibration par métamodèle:")
            print(f"  Année: {year}")
            print(f"  Taille DOE: {doe_size}")
            print(f"  Type de métamodèle: {metamodel_type}")
            print(f"  Horaires stochastiques: {use_stochastic}")
            
            # Try to load sensitivity results to get parameters
            sensitivity_results = calibration_manager.load_latest_sensitivity_results()
            
            if sensitivity_results and 'top_parameters' in sensitivity_results:
                # Use top 5 parameters from sensitivity analysis
                parameters = {}
                for param in sensitivity_results['top_parameters'][:5]:
                    parameters[param] = {
                        'name': param,
                        'bounds': [-0.3, 0.3]  # Default bounds
                    }
                print(f"Utilisation des {len(parameters)} paramètres les plus influents de l'analyse de sensibilité")
            else:
                # Use default parameters
                parameters = {
                    'infiltration_rate': {'name': 'Infiltration Rate', 'bounds': [-0.3, 0.3]},
                    'wall_rvalue': {'name': 'Wall R-Value', 'bounds': [-0.3, 0.3]},
                    'ceiling_rvalue': {'name': 'Ceiling R-Value', 'bounds': [-0.3, 0.3]},
                    'window_ufactor': {'name': 'Window U-Factor', 'bounds': [-0.3, 0.3]},
                    'heating_efficiency': {'name': 'Heating Efficiency', 'bounds': [-0.3, 0.3]}
                }
                print(f"Utilisation des {len(parameters)} paramètres par défaut")
            
            # Run metamodel calibration
            results = calibration_manager.run_metamodel_calibration(
                parameters=parameters,
                year=year,
                doe_size=doe_size,
                metamodel_type=metamodel_type,
                use_stochastic_schedules=use_stochastic
            )
            
            # Check results
            if not results:
                return None, "Calibration par métamodèle échouée: Pas de résultats"
            
            print(f"Calibration par métamodèle terminée avec succès")
            return results, ""
            
        except Exception as e:
            error_msg = str(e)
            print(f"Erreur de calibration par métamodèle: {error_msg}")
            return None, f"Calibration par métamodèle échouée: {error_msg}"
    
    # DOE results graph
    @callback(
        Output('doe_results_graph', 'figure'),
        Input('metamodel_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_doe_results_graph(store_data):
        if not store_data:
            return go.Figure()
            
        try:
            # Vérifier si les résultats DOE sont disponibles
            if 'doe_results' not in store_data:
                # Essayer de charger les résultats DOE depuis le fichier
                doe_results = None
                
                # Si les données viennent d'une calibration récente, utiliser le répertoire de cette calibration
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    doe_file = campaign_dir / 'doe' / 'doe_results.json'
                    if doe_file.exists():
                        with open(doe_file, 'r') as f:
                            doe_results = json.load(f)
                else:
                    # Sinon, chercher le répertoire de la dernière calibration par métamodèle
                    metamodel_files = []
                    for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                        if year_dir.is_dir():
                            for campaign_dir in year_dir.glob('*'):
                                if campaign_dir.is_dir():
                                    doe_file = campaign_dir / 'doe' / 'doe_results.json'
                                    if doe_file.exists():
                                        metamodel_files.append((doe_file, campaign_dir.stat().st_mtime))
                    
                    if metamodel_files:
                        # Trier par date de modification (le plus récent en premier)
                        metamodel_files.sort(key=lambda x: x[1], reverse=True)
                        latest_file = metamodel_files[0][0]
                        with open(latest_file, 'r') as f:
                            doe_results = json.load(f)
                
                if not doe_results:
                    # Créer un graphique vide avec message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Aucun résultat DOE disponible",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    return fig
            else:
                doe_results = store_data['doe_results']
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot of DOE points
            x_values = list(range(len(doe_results)))
            y_values = [result['rmse'] for result in doe_results]
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name='Points DOE',
                marker=dict(
                    size=10,
                    color=y_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="RMSE")
                )
            ))
            
            # Add best DOE point
            best_idx = y_values.index(min(y_values))
            
            fig.add_trace(go.Scatter(
                x=[best_idx],
                y=[y_values[best_idx]],
                mode='markers',
                name='Meilleur point',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Résultats du Design d'Expériences",
                xaxis_title="Index du point",
                yaxis_title="RMSE",
                template='plotly_white',
                showlegend=True
            )
            
            # Save figure to file
            try:
                # Déterminer le répertoire de sauvegarde
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    metamodel_dir = campaign_dir / 'doe'
                else:
                    # Chercher le répertoire de la dernière calibration
                    year = store_data.get('year', 2022)
                    campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metamodel_dir = config.paths['output'].CALIBRATION / str(year) / campaign_id / 'doe'
                
                ensure_dir(metamodel_dir)
                save_figure(fig, metamodel_dir, 'doe_results')
                print(f"Figure DOE sauvegardée dans {metamodel_dir}")
            except Exception as e:
                print(f"Error saving DOE results figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            print(f"Erreur lors de la génération du graphique DOE: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    # Parameter importance graph
    @callback(
        Output('parameter_importance_graph', 'figure'),
        Input('metamodel_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_parameter_importance_graph(store_data):
        if not store_data:
            return go.Figure()
            
        try:
            # Vérifier si l'analyse du métamodèle est disponible
            if 'metamodel_analysis' not in store_data:
                # Essayer de charger l'analyse depuis le fichier
                analysis = None
                
                # Si les données viennent d'une calibration récente, utiliser le répertoire de cette calibration
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    analysis_file = campaign_dir / 'analysis' / 'metamodel_analysis.json'
                    if analysis_file.exists():
                        with open(analysis_file, 'r') as f:
                            analysis = json.load(f)
                else:
                    # Sinon, chercher le répertoire de la dernière calibration par métamodèle
                    analysis_files = []
                    for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                        if year_dir.is_dir():
                            for campaign_dir in year_dir.glob('*'):
                                if campaign_dir.is_dir():
                                    analysis_file = campaign_dir / 'analysis' / 'metamodel_analysis.json'
                                    if analysis_file.exists():
                                        analysis_files.append((analysis_file, campaign_dir.stat().st_mtime))
                    
                    if analysis_files:
                        # Trier par date de modification (le plus récent en premier)
                        analysis_files.sort(key=lambda x: x[1], reverse=True)
                        latest_file = analysis_files[0][0]
                        with open(latest_file, 'r') as f:
                            analysis = json.load(f)
                
                if not analysis or 'sorted_importance' not in analysis:
                    # Créer un graphique vide avec message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Aucune analyse de métamodèle disponible",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    return fig
            else:
                analysis = store_data['metamodel_analysis']
                
                if 'sorted_importance' not in analysis:
                    # Créer un graphique vide avec message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Analyse de métamodèle incomplète",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    return fig
            
            sorted_importance = analysis['sorted_importance']
            
            # Create figure
            fig = go.Figure()
            
            # Add importance bars
            fig.add_trace(go.Bar(
                x=[item[0] for item in sorted_importance],
                y=[item[1] for item in sorted_importance],
                name='Importance',
                marker_color='purple'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Importance des Paramètres ({analysis.get('metamodel_type', 'gpr').upper()})",
                xaxis_title="Paramètre",
                yaxis_title="Importance relative",
                template='plotly_white',
                showlegend=False
            )
            
            # Save figure to file
            try:
                # Déterminer le répertoire de sauvegarde
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    metamodel_dir = campaign_dir / 'analysis'
                else:
                    # Chercher le répertoire de la dernière calibration
                    year = store_data.get('year', 2022)
                    campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metamodel_dir = config.paths['output'].CALIBRATION / str(year) / campaign_id / 'analysis'
                
                ensure_dir(metamodel_dir)
                save_figure(fig, metamodel_dir, 'parameter_importance')
                print(f"Figure d'importance des paramètres sauvegardée dans {metamodel_dir}")
            except Exception as e:
                print(f"Error saving parameter importance figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            print(f"Erreur lors de la génération du graphique d'importance: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    # Metamodel results graph
    @callback(
        Output('metamodel_results_graph', 'figure'),
        Input('metamodel_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_metamodel_results_graph(store_data):
        if not store_data or 'best_params' not in store_data:
            return go.Figure()
            
        try:
            # Load results
            # Vérifier si le fichier hourly.csv existe dans final_simulation
            results_dir = calibration_manager.current_campaign / 'final_simulation'/ 'provincial'
            hourly_file = results_dir / 'hourly.csv'
            
            if not hourly_file.exists():
                # Si le fichier n'existe pas, chercher dans le dossier parent
                hourly_file = calibration_manager.current_campaign / 'hourly.csv'
                if not hourly_file.exists():
                    # Si toujours pas trouvé, créer un graphique vide avec message d'erreur
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Fichier de résultats horaires non trouvé",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    return fig
            
            # Charger les résultats
            final_results = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Erreur lors du chargement des résultats: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Add end uses as stacked area
        end_use_cols = [col for col in final_results.columns if 'End Use: Electricity' in col]
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            # Set default visibility for heating and cooling
            visible = True if name in ['Heating', 'Cooling'] else 'legendonly'
            fig.add_trace(go.Scatter(
                x=final_results.index,
                y=final_results[col],
                name=name,
                stackgroup='end_uses',  # This creates the stacked effect
                visible=visible,
                hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
            ))
            
        # Add simulated data
        fig.add_trace(go.Scatter(
            x=final_results.index,
            y=final_results['Fuel Use: Electricity: Total'],
            name='Simulated',
            line=dict(color='red', width=2)
        ))
        
        # Add measured data
        fig.add_trace(go.Scatter(
            x=calibration_manager.hydro_data.index,
            y=calibration_manager.hydro_data['energie_sum_secteur'],
            name='Measured (Hydro-Quebec)',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            title=f"Résultats de Calibration par Métamodèle ({schedule_type} Schedules)",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Save figure to file
        try:
            year = calibration_manager.current_campaign.parent.name
            campaign_id = calibration_manager.current_campaign.name
            figures_dir = config.paths['output'].get_calibration_figures_dir(year, campaign_id)
            save_figure(fig, figures_dir, 'metamodel_results')
        except Exception as e:
            print(f"Error saving metamodel results figure: {str(e)}")
        
        return fig
    
    # Metamodel parameter details
    @callback(
        Output('metamodel_parameter_details', 'children'),
        [Input('metamodel_results_store', 'data'),
         Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def update_metamodel_parameter_details(store_data, pathname):
        # Si des données sont déjà dans le store, les utiliser
        if store_data and 'best_params' in store_data:
            best_params = store_data['best_params']
            best_metrics = store_data.get('best_metrics', {})
            metamodel_type = store_data.get('metamodel_type', 'gpr')
            doe_size = store_data.get('doe_size', 'N/A')
            stochastic = store_data.get('stochastic', False)
        else:
            # Sinon, essayer de charger les résultats précédents
            try:
                # Chercher le fichier de résultats le plus récent
                calibration_files = []
                for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                    if year_dir.is_dir():
                        for campaign_dir in year_dir.glob('*'):
                            if campaign_dir.is_dir():
                                calibration_file = campaign_dir / 'calibration_results.json'
                                if calibration_file.exists():
                                    # Vérifier si c'est un résultat de métamodèle
                                    with open(calibration_file, 'r') as f:
                                        data = json.load(f)
                                        if 'metamodel_type' in data:
                                            calibration_files.append((calibration_file, campaign_dir.stat().st_mtime))
                
                if not calibration_files:
                    return html.P("Aucun résultat de calibration par métamodèle disponible.")
                
                # Trier par date de modification (le plus récent en premier)
                calibration_files.sort(key=lambda x: x[1], reverse=True)
                latest_file = calibration_files[0][0]
                
                # Charger les résultats
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Extraire les données
                best_params = results.get('best_params', {})
                best_metrics = results.get('best_metrics', {})
                metamodel_type = results.get('metamodel_type', 'gpr')
                doe_size = results.get('doe_size', 'N/A')
                stochastic = results.get('stochastic', False)
                
                print(f"Chargé les résultats de métamodèle depuis {latest_file}")
            except Exception as e:
                print(f"Erreur lors du chargement des résultats de métamodèle: {str(e)}")
                return html.P("Aucun résultat de calibration par métamodèle disponible.")
        
        # Create table
        table_header = [
            html.Thead(html.Tr([
                html.Th("Paramètre"),
                html.Th("Valeur optimale")
            ]))
        ]
        
        rows = []
        for param, value in best_params.items():
            row = html.Tr([
                html.Td(param),
                html.Td(f"{value:.4f}")
            ])
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)
        
        # Add metrics
        metrics_div = html.Div([
            html.H5("Métriques de performance"),
            html.Ul([
                html.Li([
                    html.Strong("RMSE: "), 
                    f"{best_metrics.get('rmse', 'N/A'):.4f} MWh"
                ]),
                html.Li([
                    html.Strong("MAE: "), 
                    f"{best_metrics.get('mae', 'N/A'):.4f} MWh"
                ]),
                html.Li([
                    html.Strong("MAPE: "), 
                    f"{best_metrics.get('mape', 'N/A'):.2f}%"
                ])
            ])
        ])
        
        # Add metamodel info
        metamodel_info = html.Div([
            html.H5("Informations sur le métamodèle"),
            html.Ul([
                html.Li([
                    html.Strong("Type: "), 
                    metamodel_type.upper()
                ]),
                html.Li([
                    html.Strong("Taille DOE: "), 
                    str(doe_size)
                ]),
                html.Li([
                    html.Strong("Horaires stochastiques: "), 
                    "Oui" if stochastic else "Non"
                ])
            ])
        ])
        
        return html.Div([table, html.Hr(), metrics_div, html.Hr(), metamodel_info])
    
    # Hierarchical calibration callbacks
    @callback(
        [Output('hierarchical_results_store', 'data'),
         Output('hierarchical_loading_output', 'children')],
        Input('run_hierarchical_button', 'n_clicks'),
        [State('hierarchical_year', 'value'),
         State('hierarchical_params', 'value'),
         State('hierarchical_schedules', 'value'),
         State('custom_params_input', 'value')],
        prevent_initial_call=True
    )
    def run_hierarchical_calibration(n_clicks, year, params_type, use_stochastic, custom_params):
        if not n_clicks:
            return None, ""
            
        try:
            # Log hierarchical calibration start
            print(f"Démarrage de la calibration hiérarchique:")
            print(f"  Année: {year}")
            print(f"  Type de paramètres: {params_type}")
            print(f"  Horaires stochastiques: {use_stochastic}")
            
            # Determine parameters to calibrate
            if params_type == 'all':
                # Liste complète des paramètres
                parameters = {
                    'infiltration_rate': {'name': 'Infiltration Rate', 'bounds': [-0.3, 0.3]},
                    'wall_rvalue': {'name': 'Wall R-Value', 'bounds': [-0.3, 0.3]},
                    'ceiling_rvalue': {'name': 'Ceiling R-Value', 'bounds': [-0.3, 0.3]},
                    'window_ufactor': {'name': 'Window U-Factor', 'bounds': [-0.3, 0.3]},
                    'heating_efficiency': {'name': 'Heating Efficiency', 'bounds': [-0.3, 0.3]},
                    'heating_setpoint': {'name': 'Heating Setpoint', 'bounds': [-0.2, 0.2]},
                    'cooling_setpoint': {'name': 'Cooling Setpoint', 'bounds': [-0.2, 0.2]},
                    # Paramètres pour les schedules
                    'occupancy_scale': {'name': 'Facteur d\'occupation', 'bounds': [-0.3, 0.3]},
                    'lighting_scale': {'name': 'Facteur d\'éclairage', 'bounds': [-0.3, 0.3]},
                    'appliance_scale': {'name': 'Facteur des appareils', 'bounds': [-0.3, 0.3]},
                    'temporal_diversity': {'name': 'Diversité temporelle', 'bounds': [0.0, 1.0]}
                }
                print(f"Utilisation de tous les paramètres ({len(parameters)})")
            elif params_type == 'sensitivity':
                # Load results from most recent sensitivity analysis
                sensitivity_results = calibration_manager.load_latest_sensitivity_results()
                if not sensitivity_results:
                    return None, "Aucun résultat d'analyse de sensibilité trouvé. Exécutez d'abord l'analyse de sensibilité."
                    
                # Use top 7 parameters
                parameters = {}
                for param in sensitivity_results['top_parameters'][:7]:
                    parameters[param] = {
                        'name': param,
                        'bounds': [-0.3, 0.3]  # Default bounds
                    }
                print(f"Utilisation des {len(parameters)} paramètres les plus influents de l'analyse de sensibilité")
            elif params_type == 'custom':
                # Parse comma-separated list
                if not custom_params:
                    return None, "Veuillez entrer des paramètres personnalisés"
                    
                param_names = custom_params.split(',')
                parameters = {}
                for param in param_names:
                    param = param.strip()
                    if param:  # Skip empty strings
                        parameters[param] = {
                            'name': param,
                            'bounds': [-0.3, 0.3]  # Default bounds
                        }
                print(f"Utilisation de {len(parameters)} paramètres personnalisés")
            else:
                return None, "Type de paramètres non valide"
            
            # Run hierarchical calibration
            results = calibration_manager.run_hierarchical_calibration(
                parameters=parameters,
                year=year,
                use_stochastic_schedules=use_stochastic
            )
            
            # Check results
            if not results:
                return None, "Calibration hiérarchique échouée: Pas de résultats"
            
            print(f"Calibration hiérarchique terminée avec succès")
            return results, ""
            
        except Exception as e:
            error_msg = str(e)
            print(f"Erreur de calibration hiérarchique: {error_msg}")
            return None, f"Calibration hiérarchique échouée: {error_msg}"
    
    # Show/hide custom parameters input
    @callback(
        Output('custom_params_container', 'style'),
        Input('hierarchical_params', 'value'),
        prevent_initial_call=True
    )
    def toggle_custom_params(params_type):
        if params_type == 'custom':
            return {'display': 'block'}
        return {'display': 'none'}
    
    # Hierarchical metrics evolution graph
    @callback(
        Output('hierarchical_metrics_graph', 'figure'),
        Input('hierarchical_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_hierarchical_metrics_graph(store_data):
        if not store_data or 'metrics_evolution' not in store_data:
            return go.Figure()
            
        try:
            # Extract metrics evolution data
            metrics_evolution = store_data['metrics_evolution']
            
            # Create figure
            fig = go.Figure()
            
            # Add RMSE line
            fig.add_trace(go.Scatter(
                x=[m['level'] for m in metrics_evolution],
                y=[m['rmse'] for m in metrics_evolution],
                mode='lines+markers',
                name='RMSE',
                line=dict(color='red', width=2)
            ))
            
            # Add MAPE line
            fig.add_trace(go.Scatter(
                x=[m['level'] for m in metrics_evolution],
                y=[m['mape'] for m in metrics_evolution],
                mode='lines+markers',
                name='MAPE (%)',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            # Add level names as annotations
            for m in metrics_evolution:
                fig.add_annotation(
                    x=m['level'],
                    y=m['rmse'],
                    text=m['name'],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title="Évolution des métriques par niveau",
                xaxis_title="Niveau",
                yaxis_title="RMSE (MWh)",
                yaxis2=dict(
                    title="MAPE (%)",
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Save figure to file
            try:
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    hierarchical_dir = campaign_dir / 'hierarchical'
                    ensure_dir(hierarchical_dir)
                    save_figure(fig, hierarchical_dir, 'metrics_evolution')
            except Exception as e:
                print(f"Error saving hierarchical metrics figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            print(f"Erreur lors de la génération du graphique d'évolution des métriques: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    # Hierarchical parameters graph
    @callback(
        Output('hierarchical_params_graph', 'figure'),
        Input('hierarchical_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_hierarchical_params_graph(store_data):
        if not store_data or 'levels' not in store_data:
            return go.Figure()
            
        try:
            # Extract levels data
            levels = store_data['levels']
            
            # Create figure
            fig = go.Figure()
            
            # Collect all parameters across all levels
            all_params = set()
            for level in levels:
                if 'results' in level and 'best_params' in level['results']:
                    all_params.update(level['results']['best_params'].keys())
            
            # For each parameter, plot its evolution across levels
            for param in all_params:
                param_values = []
                level_nums = []
                level_names = []
                
                for level in levels:
                    if 'results' in level and 'best_params' in level['results']:
                        if param in level['results']['best_params']:
                            param_values.append(level['results']['best_params'][param])
                            level_nums.append(level['level'])
                            level_names.append(level['name'])
                
                if param_values:
                    fig.add_trace(go.Scatter(
                        x=level_nums,
                        y=param_values,
                        mode='lines+markers',
                        name=param,
                        hovertemplate="Niveau %{x} (%{text})<br>" +
                                     f"{param}: %{{y:.4f}}<extra></extra>",
                        text=level_names
                    ))
            
            # Update layout
            fig.update_layout(
                title="Évolution des paramètres par niveau",
                xaxis_title="Niveau",
                yaxis_title="Valeur du paramètre",
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Save figure to file
            try:
                if 'campaign_dir' in store_data:
                    campaign_dir = Path(store_data['campaign_dir'])
                    hierarchical_dir = campaign_dir / 'hierarchical'
                    ensure_dir(hierarchical_dir)
                    save_figure(fig, hierarchical_dir, 'parameters_evolution')
            except Exception as e:
                print(f"Error saving hierarchical parameters figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            print(f"Erreur lors de la génération du graphique des paramètres: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    # Hierarchical level details
    @callback(
        Output('hierarchical_level_details', 'children'),
        [Input('hierarchical_results_store', 'data'),
        Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def update_hierarchical_level_details(store_data, pathname):
        # Si nous avons des données dans le store, les utiliser
        if store_data and 'levels' in store_data:
            levels = store_data['levels']
            final_params = store_data.get('best_params', {})
            final_metrics = store_data.get('best_metrics', {})
        else:
            # Sinon, chercher les résultats précédents
            try:
                # Chercher les dossiers de calibration hiérarchique
                hierarchical_campaigns = []
                for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                    if year_dir.is_dir():
                        for campaign_dir in year_dir.glob('*'):
                            if campaign_dir.is_dir():
                                # Vérifier si c'est une calibration hiérarchique
                                results_file = campaign_dir / 'calibration_results.json'
                                if results_file.exists():
                                    try:
                                        with open(results_file, 'r') as f:
                                            data = json.load(f)
                                            if 'levels' in data and 'n_levels' in data:
                                                hierarchical_campaigns.append((campaign_dir, campaign_dir.stat().st_mtime))
                                    except:
                                        pass
                
                if not hierarchical_campaigns:
                    return html.P("Aucun résultat de calibration hiérarchique disponible.")
                
                # Prendre le plus récent
                hierarchical_campaigns.sort(key=lambda x: x[1], reverse=True)
                latest_dir = hierarchical_campaigns[0][0]
                
                # Charger les résultats
                with open(latest_dir / 'calibration_results.json', 'r') as f:
                    results = json.load(f)
                    
                levels = results.get('levels', [])
                final_params = results.get('best_params', {})
                final_metrics = results.get('best_metrics', {})
                
                if not levels:
                    return html.P("Structure de résultats hiérarchiques incomplète.")
                    
                print(f"Chargé {len(levels)} niveaux de calibration hiérarchique depuis {latest_dir}")
                
            except Exception as e:
                print(f"Erreur lors du chargement des résultats hiérarchiques: {str(e)}")
                return html.P("Aucun résultat de calibration hiérarchique disponible.")
        
        try:
            # Create level details
            level_details = []
            
            for level in levels:
                # Create card for each level
                level_card = dbc.Card([
                    dbc.CardHeader(f"Niveau {level['level']}: {level['name']}"),
                    dbc.CardBody([
                        html.P(level['description']),
                        html.H6("Paramètres calibrés:"),
                        html.Ul([html.Li(param) for param in level['parameters']]),
                        html.H6("Métriques:"),
                        html.Ul([
                            html.Li(f"RMSE: {level['metrics'].get('rmse', 'N/A'):.4f}"),
                            html.Li(f"MAPE: {level['metrics'].get('mape', 'N/A'):.2f}%"),
                            html.Li(f"MAE: {level['metrics'].get('mae', 'N/A'):.4f}")
                        ])
                    ])
                ], className="mb-3")
                
                level_details.append(level_card)
            
            # Add final results card
            if final_params and final_metrics:
                final_card = dbc.Card([
                    dbc.CardHeader("Résultats Finaux"),
                    dbc.CardBody([
                        html.H6("Paramètres optimaux finaux:"),
                        html.Ul([
                            html.Li(f"{param}: {value:.4f}") 
                            for param, value in final_params.items()
                        ]),
                        html.H6("Métriques finales:"),
                        html.Ul([
                            html.Li(f"RMSE: {final_metrics.get('rmse', 'N/A'):.4f}"),
                            html.Li(f"MAPE: {final_metrics.get('mape', 'N/A'):.2f}%"),
                            html.Li(f"MAE: {final_metrics.get('mae', 'N/A'):.4f}")
                        ]),
                        html.P("Ces résultats représentent la performance finale après tous les niveaux de calibration.",
                            className="text-muted mt-2")
                    ])
                ], className="mb-3 border-success")
                
                level_details.append(final_card)
            
            return html.Div(level_details)
            
        except Exception as e:
            print(f"Erreur lors de la génération des détails des niveaux: {str(e)}")
            return html.P(f"Erreur: {str(e)}")
    
    # Hierarchical results graph
    @callback(
        Output('hierarchical_results_graph', 'figure'),
        Input('hierarchical_results_store', 'data'),
        prevent_initial_call=True
    )
    def update_hierarchical_results_graph(store_data):
        if not store_data or 'best_params' not in store_data:
            return go.Figure()
            
        try:
            # Load results
            # Vérifier si le fichier hourly.csv existe dans final_simulation
            if 'campaign_dir' in store_data:
                results_dir = Path(store_data['campaign_dir']) / 'final_simulation'
                hourly_file = results_dir / 'hourly.csv'
                
                if not hourly_file.exists():
                    # Si le fichier n'existe pas, chercher dans le dossier parent
                    hourly_file = Path(store_data['campaign_dir']) / 'hourly.csv'
                    if not hourly_file.exists():
                        # Si toujours pas trouvé, créer un graphique vide avec message d'erreur
                        fig = go.Figure()
                        fig.add_annotation(
                            text="Fichier de résultats horaires non trouvé",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False
                        )
                        return fig
                
                # Charger les résultats
                final_results = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
            else:
                # Si pas de campaign_dir, créer un graphique vide avec message d'erreur
                fig = go.Figure()
                fig.add_annotation(
                    text="Répertoire de campagne non trouvé",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
        except Exception as e:
            print(f"Erreur lors du chargement des résultats: {str(e)}")
            # Créer un graphique vide avec message d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Add end uses as stacked area
        end_use_cols = [col for col in final_results.columns if 'End Use: Electricity' in col]
        for col in end_use_cols:
            name = col.replace('End Use: Electricity: ', '')
            # Set default visibility for heating and cooling
            visible = True if name in ['Heating', 'Cooling'] else 'legendonly'
            fig.add_trace(go.Scatter(
                x=final_results.index,
                y=final_results[col],
                name=name,
                stackgroup='end_uses',  # This creates the stacked effect
                visible=visible,
                hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
            ))
        
        # Add simulated data
        fig.add_trace(go.Scatter(
            x=final_results.index,
            y=final_results['Fuel Use: Electricity: Total'],
            name='Simulated',
            line=dict(color='red', width=2)
        ))
        
        # Add measured data
        fig.add_trace(go.Scatter(
            x=calibration_manager.hydro_data.index,
            y=calibration_manager.hydro_data['energie_sum_secteur'],
            name='Measured (Hydro-Quebec)',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        schedule_type = "Stochastic" if store_data.get('stochastic', False) else "Standard"
        fig.update_layout(
            title=f"Résultats de Calibration Hiérarchique ({schedule_type} Schedules)",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Save figure to file
        try:
            if 'campaign_dir' in store_data:
                campaign_dir = Path(store_data['campaign_dir'])
                hierarchical_dir = campaign_dir / 'hierarchical'
                ensure_dir(hierarchical_dir)
                save_figure(fig, hierarchical_dir, 'final_results')
        except Exception as e:
            print(f"Error saving hierarchical results figure: {str(e)}")
        
        return fig

    # Disable standard controls when a future scenario is selected
    @callback(
        [Output('simulation_year', 'disabled'),
         Output('simulation_scenario', 'disabled'),
         Output('simulation_schedules', 'disabled')],
        Input('future_scenario', 'value')
    )
    def toggle_standard_controls(future_scenario):
        """Disable standard controls when a future scenario is selected."""
        if future_scenario:
            return True, True, True
        return False, False, False
