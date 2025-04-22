"""
Module d'analyse de sensibilité pour identifier les paramètres les plus influents.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from pathlib import Path

from ..config import config
from ..utils import ensure_dir, save_results

class SensitivityAnalyzer:
    """Analyze parameter sensitivity using Morris method."""
    
    def __init__(self, simulation_manager, aggregation_manager):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            simulation_manager: Instance of SimulationManager
            aggregation_manager: Instance of AggregationManager
        """
        self.logger = logging.getLogger("src.calibration.sensitivity_analyzer")
        self.simulation_manager = simulation_manager
        self.aggregation_manager = aggregation_manager
        self.hydro_data = None
        
        # Liste des paramètres potentiels à calibrer (limités à 10)
        self.potential_parameters = {
            # Paramètres déjà calibrés dans l'approche actuelle
            'infiltration_rate': {'name': 'Taux d\'infiltration', 'bounds': [-0.3, 0.3], 'hpxml_name': 'air_leakage_value'},
            'wall_rvalue': {'name': 'R-value des murs', 'bounds': [-0.3, 0.3], 'hpxml_name': 'wall_assembly_r'},
            'ceiling_rvalue': {'name': 'R-value du plafond', 'bounds': [-0.3, 0.3], 'hpxml_name': 'ceiling_assembly_r'},
            'window_ufactor': {'name': 'U-factor des fenêtres', 'bounds': [-0.3, 0.3], 'hpxml_name': 'window_ufactor'},
            
            # Nouveaux paramètres comportementaux (basés sur le workflow existant)
            'occupancy_scale': {'name': 'Facteur d\'occupation', 'bounds': [-0.3, 0.3], 'hpxml_name': 'occupancy_scale'},
            'lighting_scale': {'name': 'Facteur d\'éclairage', 'bounds': [-0.3, 0.3], 'hpxml_name': 'lighting_scale'},
            'appliance_scale': {'name': 'Facteur des appareils', 'bounds': [-0.3, 0.3], 'hpxml_name': 'appliance_scale'},
            'temporal_diversity': {'name': 'Diversité temporelle', 'bounds': [0.0, 1.0], 'hpxml_name': 'temporal_diversity'},
            'heating_setpoint': {'name': 'Consigne de chauffage', 'bounds': [-0.15, 0.15], 'hpxml_name': 'hvac_control_heating_weekday_setpoint'},
            'cooling_setpoint': {'name': 'Consigne de climatisation', 'bounds': [-0.15, 0.15], 'hpxml_name': 'hvac_control_cooling_weekday_setpoint'},
            
            # Paramètres des systèmes (basés sur le workflow existant)
            'heating_efficiency': {'name': 'Efficacité du chauffage', 'bounds': [-0.3, 0.3], 'hpxml_name': 'heating_system_heating_efficiency'},
            'dhw_volume': {'name': 'Volume du chauffe-eau', 'bounds': [-0.3, 0.3], 'hpxml_name': 'water_heater_tank_volume'},
            'foundation_r': {'name': 'R-value des fondations', 'bounds': [-0.3, 0.3], 'hpxml_name': 'foundation_wall_assembly_r'}
        }
        
    def _load_hydro_data(self, year: int) -> None:
        """Load and process Hydro-Quebec consumption data."""
        from ..utils import process_hydro_data
        try:
            hydro_file = config.paths['input'].get_hydro_file(year)
            raw_data = pd.read_csv(hydro_file)
            self.hydro_data = process_hydro_data(raw_data, year)
        except Exception as e:
            self.logger.error(f"Error loading Hydro-Quebec data: {str(e)}")
            raise
    
    def _calculate_error_metrics(self, simulation_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate error metrics between simulation and measured data."""
        from ..utils import calculate_metrics
        return calculate_metrics(simulation_results, self.hydro_data)
    
    def _run_simulation_with_params(self, params: Dict, output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Run simulation with given parameters and calculate metrics.
        
        Args:
            params: Dictionary of parameter values
            output_dir: Optional output directory
            
        Returns:
            Dictionary of error metrics
        """
        # Convert flat params dict to nested format expected by simulation_manager
        nested_params = self._convert_to_nested_params(params)
        
        # Run simulations
        simulation_results = self.simulation_manager.run_parallel_simulations(
            year=2022,  # Année fixe pour l'analyse de sensibilité
            scenario="sensitivity",
            parameters=nested_params,
            output_dir=output_dir,  # Utiliser le répertoire fourni
            use_stochastic_schedules=False,  # Pour cohérence et rapidité
            cleanup_after=True  # Activer le nettoyage
        )
        
        # Aggregate results
        provincial_results, _, _  = self.aggregation_manager.aggregate_results(
            2022, simulation_results, "sensitivity", output_dir=output_dir
        )
        
        # Calculate metrics
        metrics = self._calculate_error_metrics(provincial_results)
        return metrics
    
    def _convert_to_nested_params(self, flat_params: Dict) -> Dict:
        """
        Convert flat parameter dict to nested format for simulation manager,
        with stratification by construction period.
        
        Args:
            flat_params: Dictionary {param_name: value}
            
        Returns:
            Nested dictionary {param_name: {archetype_id: value}}
        """
        nested_params = {}
        archetypes_df = self.simulation_manager.archetype_manager.archetypes_df
        
        for param_name, value in flat_params.items():
            if param_name not in nested_params:
                nested_params[param_name] = {}
            
            # Récupérer le nom HPXML du paramètre
            hpxml_name = self.potential_parameters[param_name]['hpxml_name']
                
            # Stratifier par période de construction (comme dans l'approche actuelle)
            for idx in range(len(archetypes_df)):
                # Déterminer la période de construction
                year = archetypes_df.iloc[idx]['vintageExact']
                
                # Appliquer le paramètre avec une stratification par période
                if year < 1980:  # pre1980
                    # Pour maintenir la cohérence avec l'approche actuelle
                    # Les paramètres peuvent avoir des plages d'ajustement différentes selon la période
                    # Ici on utilise la même valeur pour simplifier, mais on pourrait introduire
                    # des facteurs spécifiques à chaque période si nécessaire
                    nested_params[param_name][idx] = value
                else:  # post1980
                    nested_params[param_name][idx] = value
                
        return nested_params
    
    def run_morris_analysis(self, 
                          param_subset: Optional[List[str]] = None,
                          n_trajectories: int = 10,
                          year: int = 2022,
                          output_dir: Optional[Path] = None) -> Dict:
        """
        Run Morris sensitivity analysis.
        
        Args:
            param_subset: Optional subset of parameters to analyze
            n_trajectories: Number of Morris trajectories
            year: Year to analyze
            output_dir: Optional output directory
            
        Returns:
            Dictionary with sensitivity results
        """
        # Ensure Hydro data is loaded
        if self.hydro_data is None:
            self._load_hydro_data(year)
        
        # Select parameters to analyze
        if param_subset:
            parameters = {k: v for k, v in self.potential_parameters.items() if k in param_subset}
        else:
            parameters = self.potential_parameters
        
        self.logger.info(f"Running sensitivity analysis on {len(parameters)} parameters")
        
        # Setup problem definition for SALib
        problem = {
            'num_vars': len(parameters),
            'names': list(parameters.keys()),
            'bounds': [parameters[name]['bounds'] for name in parameters.keys()]
        }
        
        # Generate Morris trajectories
        self.logger.info(f"Generating {n_trajectories} Morris trajectories")
        X = morris.sample(problem, N=n_trajectories, num_levels=4)
        
        # Exécuter simulations
        results = []
        for i, param_values in enumerate(X):
            param_dict = {name: value for name, value in zip(problem['names'], param_values)}
            self.logger.info(f"Running simulation {i+1}/{len(X)}")
            
            # Log parameter values
            for name, value in param_dict.items():
                self.logger.debug(f"  {name}: {value}")
            
            try:
                metrics = self._run_simulation_with_params(param_dict, output_dir=output_dir)
                
                # On utilise RMSE comme métrique principale pour l'analyse
                results.append(metrics['rmse'])
                
                self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.error(f"Error in simulation {i+1}: {str(e)}")
                # Use a high RMSE value to represent failure
                results.append(1000.0)
        
        # Analyze results
        self.logger.info("Analyzing sensitivity results")
        morris_results = morris_analyze.analyze(
            problem, X, np.array(results), 
            conf_level=0.95, print_to_console=False, num_levels=4
        )
        
        # Organize results
        param_sensitivity = []
        for i, name in enumerate(problem['names']):
            param_sensitivity.append({
                'name': name,
                'display_name': parameters[name]['name'],
                'mu_star': float(morris_results['mu_star'][i]),
                'sigma': float(morris_results['sigma'][i]),
                'mu_star_conf': float(morris_results['mu_star_conf'][i])
            })
        
        # Sort by importance (mu_star)
        param_sensitivity.sort(key=lambda x: x['mu_star'], reverse=True)
        
        # Add rank after sorting
        for i, param in enumerate(param_sensitivity):
            param['rank'] = i + 1
        
        # Save results if output directory is provided
        if output_dir:
            save_dir = ensure_dir(output_dir)
            save_results(param_sensitivity, save_dir / 'sensitivity_results.json', 'json')
            
            # Generate simple report
            report = self._generate_sensitivity_report(param_sensitivity)
            with open(save_dir / 'sensitivity_report.md', 'w') as f:
                f.write(report)
        
        return {
            'param_sensitivity': param_sensitivity,
            'top_parameters': [p['name'] for p in param_sensitivity[:5]],
            'problem': problem,
            'X': X.tolist(),
            'Y': results
        }
    
    def _generate_sensitivity_report(self, param_sensitivity: List[Dict]) -> str:
        """Generate a markdown report of sensitivity results."""
        report = "# Analyse de sensibilité des paramètres\n\n"
        report += "## Classement des paramètres par influence\n\n"
        report += "| Rang | Paramètre | Influence (mu*) | Non-linéarité (sigma) |\n"
        report += "|------|-----------|----------------|-------------------|\n"
        
        for param in param_sensitivity:
            report += f"| {param['rank']} | {param['display_name']} | {param['mu_star']:.4f} | {param['sigma']:.4f} |\n"
        
        report += "\n\n## Interprétation\n\n"
        report += "- **Influence (mu*)**: Plus cette valeur est élevée, plus le paramètre a d'impact sur les résultats\n"
        report += "- **Non-linéarité (sigma)**: Une valeur élevée indique des effets non-linéaires ou des interactions fortes avec d'autres paramètres\n\n"
        
        # Add top parameters section
        report += "## Paramètres recommandés pour calibration\n\n"
        for i, param in enumerate(param_sensitivity[:5]):
            report += f"{i+1}. **{param['display_name']}** (influence: {param['mu_star']:.4f})\n"
        
        return report