"""
Module de calibration hiérarchique multi-niveaux pour UBEM Québec.
Décompose le processus de calibration en plusieurs niveaux successifs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import copy

from ..config import config
from ..utils import ensure_dir, save_results, calculate_metrics

class HierarchicalCalibrator:
    """Calibration hiérarchique multi-niveaux."""
    
    def __init__(self, simulation_manager, aggregation_manager, metamodel_calibrator):
        """
        Initialise le calibrateur hiérarchique.
        
        Args:
            simulation_manager: Instance du SimulationManager
            aggregation_manager: Instance de l'AggregationManager
            metamodel_calibrator: Instance du MetamodelCalibrator
        """
        self.logger = logging.getLogger("src.calibration.hierarchical_calibrator")
        self.simulation_manager = simulation_manager
        self.aggregation_manager = aggregation_manager
        self.metamodel_calibrator = metamodel_calibrator
        self.hydro_data = None
        
        
        self.levels = [
            
            {
                'name': 'global_envelope',
                'description': 'Paramètres globaux d\'enveloppe',
                'parameters': ['infiltration_rate', 'wall_rvalue', 'ceiling_rvalue', 'window_ufactor'],
                'doe_size': 60,
                'metamodel_type': 'gpr'
            },
            {
                'name': 'systems',
                'description': 'Paramètres des systèmes de chauffage et climatisation',
                'parameters': ['heating_efficiency', 'heating_setpoint', 'cooling_setpoint', 'temporal_diversity'],
                'doe_size': 60,
                'metamodel_type': 'gpr'
            },

            {
                'name': 'schedules',
                'description': 'Paramètres des schedules et diversité temporelle',
                'parameters': ['occupancy_scale', 'lighting_scale', 'appliance_scale'],
                'doe_size': 60, 
                'metamodel_type': 'gpr'  
            }
        ]
        
        # Paramètres actuels (mis à jour à chaque niveau)
        self.current_params = {}
        
    def _load_hydro_data(self, year: int) -> None:
        """Charge les données Hydro-Québec."""
        from ..utils import process_hydro_data
        try:
            hydro_file = config.paths['input'].get_hydro_file(year)
            raw_data = pd.read_csv(hydro_file)
            self.hydro_data = process_hydro_data(raw_data, year)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données Hydro-Québec : {str(e)}")
            raise
    
    def _prepare_level_parameters(self, level: Dict, 
                                all_parameters: Dict[str, Dict],
                                sensitivity_results: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Prépare les paramètres pour un niveau spécifique.
        
        Args:
            level: Définition du niveau
            all_parameters: Dictionnaire complet des paramètres
            sensitivity_results: Résultats optionnels d'analyse de sensibilité
            
        Returns:
            Sous-ensemble de paramètres pour ce niveau
        """
        # Si des résultats de sensibilité sont disponibles, utiliser pour ordonner les paramètres
        if sensitivity_results and 'param_sensitivity' in sensitivity_results:
            # Créer une liste triée par importance
            sorted_params = []
            for param_info in sensitivity_results['param_sensitivity']:
                param_name = param_info['name']
                if param_name in level['parameters'] and param_name in all_parameters:
                    sorted_params.append(param_name)
            
            # Ajouter les paramètres restants du niveau
            for param_name in level['parameters']:
                if param_name not in sorted_params and param_name in all_parameters:
                    sorted_params.append(param_name)
                    
            # Remplacer la liste de paramètres du niveau
            level['parameters'] = sorted_params
        
        # Extraire le sous-ensemble de paramètres pour ce niveau
        level_params = {}
        for param_name in level['parameters']:
            if param_name in all_parameters:
                level_params[param_name] = copy.deepcopy(all_parameters[param_name])
                
        return level_params
    
    def _prepare_specific_data(self, level: Dict) -> Optional[pd.DataFrame]:
        """
        Prépare des données spécifiques pour un niveau (ex: usage particulier).
        
        Args:
            level: Définition du niveau
            
        Returns:
            DataFrame spécifique ou None si global
        """
        if 'usage_column' in level and level['usage_column'] and self.hydro_data is not None:
            # Pour les niveaux associés à un usage spécifique
            self.logger.info(f"Préparation des données spécifiques pour l'usage {level['usage_column']}")
            
            
            return self.hydro_data
        
        return None
    
    def _update_current_params(self, level_results: Dict) -> None:
        """
        Met à jour les paramètres actuels avec les résultats du niveau.
        
        Args:
            level_results: Résultats de calibration du niveau
        """
        if 'best_params' in level_results:
            for param, value in level_results['best_params'].items():
                self.current_params[param] = value
                self.logger.info(f"Paramètre mis à jour : {param} = {value:.4f}")
    
    def _apply_current_params(self, parameters: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Applique les paramètres actuels aux paramètres fournis.
        
        Args:
            parameters: Définition des paramètres
            
        Returns:
            Paramètres avec valeurs initiales des niveaux précédents
        """
        for param, value in self.current_params.items():
            if param in parameters:
                # Modifier la valeur initiale mais garder les bornes
                parameters[param]['initial_value'] = value
                
        return parameters
    
    def run_calibration(self, 
                       parameters: Dict[str, Dict],
                       year: int,
                       use_stochastic_schedules: bool = False,
                       output_dir: Optional[Path] = None,
                       sensitivity_results: Optional[Dict] = None,
                       transfer_learning_params: Optional[Dict] = None) -> Dict:
        """
        Exécute la calibration hiérarchique multi-niveaux.
        
        Args:
            parameters: Dictionnaire des paramètres à calibrer
            year: Année de calibration
            use_stochastic_schedules: Utilisation des horaires stochastiques
            output_dir: Répertoire de sortie optionnel
            sensitivity_results: Résultats optionnels d'analyse de sensibilité
            transfer_learning_params: Paramètres initiaux optionnels du transfer learning
            
        Returns:
            Dictionnaire avec les résultats de calibration
        """
        try:
            # Vérifier que les données Hydro sont chargées
            if self.hydro_data is None:
                self._load_hydro_data(year)
                self.metamodel_calibrator.hydro_data = self.hydro_data
            
            # Créer le répertoire de sortie
            if output_dir is None:
                campaign_dir = config.paths['output'].get_calibration_dir(
                    year, datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                output_dir = ensure_dir(campaign_dir)
                
            hierarchical_dir = ensure_dir(output_dir / 'hierarchical')
            
            # Initialiser les paramètres actuels avec les résultats du transfer learning si disponibles
            self.current_params = {}
            if transfer_learning_params:
                self.current_params = copy.deepcopy(transfer_learning_params)
                self.logger.info(f"Paramètres initiaux du transfer learning appliqués: {self.current_params}")
            
            # Exécuter la calibration niveau par niveau
            level_results = []
            all_metrics = []
            final_params = {}
            
            self.logger.info(f"Démarrage de la calibration hiérarchique à {len(self.levels)} niveaux")
            
            for i, level in enumerate(self.levels):
                self.logger.info(f"==== NIVEAU {i+1}/{len(self.levels)}: {level['name'].upper()} ====")
                self.logger.info(f"Description: {level['description']}")
                self.logger.info(f"Paramètres: {level['parameters']}")
                
                # Préparer les paramètres pour ce niveau
                level_params = self._prepare_level_parameters(level, parameters, sensitivity_results)
                if not level_params:
                    self.logger.warning(f"Aucun paramètre à calibrer pour le niveau {level['name']}")
                    continue
                
                self.logger.info(f"Calibration de {len(level_params)} paramètres: {list(level_params.keys())}")
                
                # Appliquer les paramètres actuels comme valeurs initiales
                level_params = self._apply_current_params(level_params)
                
                # Préparer des données spécifiques si nécessaire
                specific_data = self._prepare_specific_data(level)
                if specific_data is not None:
                    self.metamodel_calibrator.hydro_data = specific_data
                
                # Configurer le métamodelcalibrator pour ce niveau
                self.metamodel_calibrator.doe_size = level['doe_size']
                self.metamodel_calibrator.metamodel_type = level['metamodel_type']
                
                # Si des paramètres initiaux sont disponibles de niveaux précédents, les utiliser
                if self.current_params:
                    initial_params = {p: v for p, v in self.current_params.items() if p in level_params}
                    if initial_params:
                        self.metamodel_calibrator.initial_params = initial_params
                
                # Créer le répertoire pour ce niveau
                level_dir = ensure_dir(hierarchical_dir / f"level_{i+1}_{level['name']}")
                
                # Exécuter la calibration pour ce niveau
                level_result = self.metamodel_calibrator.run_calibration(
                    parameters=level_params,
                    year=year,
                    use_stochastic_schedules=use_stochastic_schedules,
                    output_dir=level_dir
                )
                
                # Mettre à jour les paramètres actuels avec les résultats de ce niveau
                self._update_current_params(level_result)

                # Appliquer des contraintes entre paramètres
                constrained_params = self._apply_parameter_constraints(i, self.current_params)
                self.current_params = constrained_params
                
                # Sauvegarder les résultats de ce niveau
                level_data = {
                    'level': i+1,
                    'name': level['name'],
                    'description': level['description'],
                    'parameters': list(level_params.keys()),
                    'results': level_result,
                    'doe_size': level['doe_size'],
                    'metrics': level_result.get('best_metrics', {})
                }
                level_results.append(level_data)
                save_results(level_data, level_dir / 'level_results.json', 'json')
                
                # Collecter les métriques
                if 'best_metrics' in level_result:
                    all_metrics.append({
                        'level': i+1,
                        'name': level['name'],
                        'rmse': level_result['best_metrics'].get('rmse', float('nan')),
                        'mape': level_result['best_metrics'].get('mape', float('nan')),
                        'mae': level_result['best_metrics'].get('mae', float('nan'))
                    })
                
                # Mettre à jour les paramètres finaux
                if 'best_params' in level_result:
                    for param, value in level_result['best_params'].items():
                        final_params[param] = value
                
                self.logger.info(f"Niveau {i+1} ({level['name']}) terminé avec RMSE: "
                               f"{level_result.get('best_metrics', {}).get('rmse', 'N/A')}")
                
            # Ajouter une phase d'optimisation finale
            self.logger.info("Exécution d'une optimisation finale combinant tous les paramètres")

            # Créer un métamodèle final avec tous les paramètres optimisés
            final_opt_dir = ensure_dir(output_dir / 'final_optimization')

            # Utiliser une taille de DOE plus petite mais autour des valeurs déjà optimisées
            refined_parameters = {}
            for param, value in final_params.items():
                # Créer des bornes restreintes autour des valeurs optimales
                lower = max(parameters[param]['bounds'][0], value - 0.1)
                upper = min(parameters[param]['bounds'][1], value + 0.1)
                refined_parameters[param] = {
                    'name': parameters[param].get('name', param),
                    'bounds': [lower, upper],
                    'initial_value': value
                }

            # Configurer le DOE avant d'appeler run_calibration
            self.metamodel_calibrator.doe_size = 30  

            # Lancer une dernière calibration avec tous les paramètres
            final_results = self.metamodel_calibrator.run_calibration(
                parameters=refined_parameters,
                year=year,
                use_stochastic_schedules=use_stochastic_schedules,
                output_dir=final_opt_dir
               
            )
            
            # Utiliser les paramètres finaux optimisés ensemble
            if 'best_params' in final_results:
                final_params = final_results['best_params']
                self.logger.info("Paramètres affinés par l'optimisation finale")

            # Exécuter une simulation finale avec tous les paramètres
            self.logger.info("Exécution de la simulation finale avec tous les paramètres calibrés")
            
            # Convertir les paramètres finaux au format attendu par le simulation_manager
            nested_params = self._convert_to_nested_params(final_params)
            
            final_simulation = self.simulation_manager.run_parallel_simulations(
                year=year,
                scenario="final_hierarchical",
                parameters=nested_params,
                use_stochastic_schedules=use_stochastic_schedules,
                output_dir=ensure_dir(output_dir / 'final_simulation')
            )
            
            # Agréger les résultats finaux
            provincial_results, _, _ = self.aggregation_manager.aggregate_results(
                year, final_simulation, "final_hierarchical",
                output_dir=ensure_dir(output_dir / 'final_simulation'),
                skip_mrc_aggregation=False  
            )
            
            # Calculer les métriques finales
            final_metrics = calculate_metrics(provincial_results, self.hydro_data)
            
            # Sauvegarder les résultats horaires
            save_results(provincial_results, output_dir / 'hourly.csv', 'csv')
            
            # Compiler les résultats finaux
            final_results = {
                'best_params': final_params,
                'best_metrics': final_metrics,
                'levels': level_results,
                'metrics_evolution': all_metrics,
                'n_levels': len(self.levels),
                'timestamp': datetime.now().isoformat(),
                'stochastic': use_stochastic_schedules,
                'campaign_dir': str(output_dir)
            }
            
            # Sauvegarder les résultats finaux
            save_results(final_results, output_dir / 'calibration_results.json', 'json')
            
            # Générer un rapport explicatif
            report = self._generate_report(final_results)
            with open(output_dir / 'hierarchical_report.md', 'w') as f:
                f.write(report)
            
            self.logger.info("Calibration hiérarchique terminée avec succès")
            self.logger.info(f"RMSE final: {final_metrics['rmse']:.4f}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la calibration hiérarchique: {str(e)}")
            raise
    
    def _convert_to_nested_params(self, flat_params: Dict) -> Dict:
        """
        Convertit les paramètres plats en format imbriqué pour le simulation_manager.
        
        Args:
            flat_params: Dictionnaire {param_name: value}
            
        Returns:
            Dictionnaire imbriqué {param_name: {archetype_id: value}}
        """
        nested_params = {}
        archetypes_df = self.simulation_manager.archetype_manager.archetypes_df
        
        for param_name, value in flat_params.items():
            if param_name not in nested_params:
                nested_params[param_name] = {}
                
            # Stratifier par période de construction
            for idx in range(len(archetypes_df)):
                # Déterminer la période de construction
                year = archetypes_df.iloc[idx]['vintageExact']
                
                # Appliquer le paramètre avec une stratification par période
                if year < 1980:  # pre1980
                    nested_params[param_name][idx] = value
                else:  # post1980
                    nested_params[param_name][idx] = value
                
        return nested_params
    
    def _generate_report(self, results: Dict) -> str:
        """Génère un rapport explicatif des résultats."""
        report = "# Rapport de Calibration Hiérarchique Multi-niveaux\n\n"
        report += f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n\n"
        
        report += "## Aperçu des résultats\n\n"
        report += f"- **RMSE final**: {results['best_metrics']['rmse']:.4f}\n"
        report += f"- **MAPE final**: {results['best_metrics']['mape']:.4f}%\n"
        report += f"- **MAE final**: {results['best_metrics']['mae']:.4f}\n"
        report += f"- **Nombre de niveaux**: {results['n_levels']}\n"
        report += f"- **Horaires stochastiques**: {'Oui' if results['stochastic'] else 'Non'}\n\n"
        
        report += "## Paramètres optimaux\n\n"
        report += "| Paramètre | Valeur optimale |\n"
        report += "|-----------|----------------|\n"
        
        for param, value in results['best_params'].items():
            report += f"| {param} | {value:.4f} |\n"
        
        report += "\n## Évolution des métriques par niveau\n\n"
        report += "| Niveau | Nom | RMSE | MAPE | MAE |\n"
        report += "|--------|-----|------|------|-----|\n"
        
        for metric in results['metrics_evolution']:
            report += f"| {metric['level']} | {metric['name']} | {metric['rmse']:.4f} | "
            report += f"{metric['mape']:.4f}% | {metric['mae']:.4f} |\n"
        
        report += "\n## Détail des niveaux\n\n"
        
        for i, level in enumerate(results['levels']):
            report += f"### Niveau {i+1}: {level['name']}\n\n"
            report += f"*{level['description']}*\n\n"
            report += f"- **Paramètres calibrés**: {', '.join(level['parameters'])}\n"
            report += f"- **Taille du DOE**: {level['doe_size']} points\n"
            report += f"- **RMSE**: {level['metrics'].get('rmse', 'N/A'):.4f}\n\n"
        
        report += "## Analyse et conclusions\n\n"
        
        # Identifier les améliorations entre niveaux
        if len(results['metrics_evolution']) > 1:
            improvements = []
            for i in range(1, len(results['metrics_evolution'])):
                prev = results['metrics_evolution'][i-1]
                curr = results['metrics_evolution'][i]
                if 'rmse' in prev and 'rmse' in curr:
                    improvement = (prev['rmse'] - curr['rmse']) / prev['rmse'] * 100
                    improvements.append((i+1, curr['name'], improvement))
            
            report += "### Améliorations par niveau\n\n"
            for level, name, improvement in improvements:
                direction = "amélioration" if improvement > 0 else "dégradation"
                report += f"- **Niveau {level} ({name})**: {abs(improvement):.2f}% {direction}\n"
        
        return report
    
    def _apply_parameter_constraints(self, level_index, optimized_params):
        """Applique des contraintes entre paramètres pour maintenir la cohérence physique."""
        
        constrained_params = optimized_params.copy()
        
        # Exemples de contraintes:
        
        # 1. Si l'infiltration augmente, le chauffage devrait augmenter aussi
        if level_index == 1 and 'heating_efficiency' in constrained_params:
            if 'infiltration_rate' in self.current_params and self.current_params['infiltration_rate'] > 0.1:
                # Augmenter l'efficacité du chauffage si infiltration élevée
                constrained_params['heating_efficiency'] = max(
                    constrained_params['heating_efficiency'], 
                    0.05
                )
        
        # 2. Limiter l'effet des schedules si l'enveloppe est déjà bien calibrée
        if level_index == 2 and self.levels[level_index]['name'] == 'schedules':
            prev_rmse = min([level.get('metrics', {}).get('rmse', float('inf')) for level in self.levels[:level_index]])
            if prev_rmse < 500:  # Si déjà bien calibré
                # Réduire l'amplitude des ajustements de schedules
                for param in ['occupancy_scale', 'lighting_scale', 'appliance_scale']:
                    if param in constrained_params:
                        constrained_params[param] = constrained_params[param] * 0.7
        
        return constrained_params

