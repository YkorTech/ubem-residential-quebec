"""
SimulationManager handles parallel execution of building simulations.
"""
import multiprocessing as mp
from pathlib import Path
import subprocess
import logging
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from ..config import config
from ..utils import get_available_cpus, ensure_dir, generate_run_id
from .archetype_manager import ArchetypeManager

class SimulationManager:
    """Manages parallel execution of OpenStudio simulations."""
    
    def __init__(self):
        """Initialize the SimulationManager."""
        self.logger = logging.getLogger(__name__)
        self.archetype_manager = ArchetypeManager()
        # Use fewer CPUs to avoid overwhelming the system
        self.n_cpus = get_available_cpus(0.8)  # Use 80% of available CPUs
        
    def _run_simulation(self, workflow_dir: Path, cleanup_after: bool = False) -> bool:
        """
        Run a single OpenStudio simulation.
        
        Args:
            workflow_dir: Directory containing the workflow.osw file
            cleanup_after: Whether to clean up after simulation (unused but needed for compatibility)
            
        Returns:
            True if simulation successful, False otherwise
        """
        try:
            workflow_path = workflow_dir / 'workflow.osw'
            if not workflow_path.exists():
                self.logger.error(f"Workflow file not found: {workflow_path}")
                return False
                
            # Run OpenStudio CLI command
            cmd = ['openstudio', 'run', '-w', str(workflow_path)]
            try:
                result = subprocess.run(cmd, 
                                     capture_output=True, 
                                     text=True,
                                     cwd=workflow_dir,
                                     timeout=180)  # 3 minutes timeout (augmenté de 60s à 180s)
            except subprocess.TimeoutExpired:
                self.logger.error(f"Simulation timed out: {workflow_dir.name}")
                return False
            
            if result.returncode != 0:
                self.logger.error(f"Simulation failed: {result.stderr}")
                return False
                
            self.logger.info(f"Simulation completed: {workflow_dir.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running simulation: {str(e)}")
            return False
    
    def _process_simulation_results(self, arch_dir: Path, year) -> Optional[pd.DataFrame]:
        """
        Process simulation results for a single archetype.
        
        Args:
            arch_dir: Directory containing simulation results
            
        Returns:
            DataFrame with processed results or None if processing fails
        """
        try:
            results_file = arch_dir / 'run' / 'results_timeseries.csv'
            if not results_file.exists():
                self.logger.error(f"Results file not found: {results_file}")
                return None
                
            # Get archetype data
            archetype = self.archetype_manager.archetypes_df.iloc[
                int(arch_dir.name.replace('archetype_', ''))
            ]
            
            # Read results, skipping the units row
            df = pd.read_csv(results_file, skiprows=[1])
            
            # Convert time column to datetime and ensure it's in 2022
            df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)
            df['Time'] = df['Time'].map(lambda t: t.replace(year=year))
            df.set_index('Time', inplace=True)
            
            # Debug log timestamp range - reduced to debug level
            self.logger.debug(
                f"Processed results time range: {df.index.min()} to {df.index.max()}"
            )
            
            # Read units row separately
            units_df = pd.read_csv(results_file, nrows=2).iloc[1]
            
            # Convert units if needed (e.g., BTU to kWh)
            energy_cols = [col for col in df.columns if 'Electricity' in col]
            for col in energy_cols:
                if col in units_df and units_df[col] == 'kBtu':
                    df[col] = pd.to_numeric(df[col]) * 0.000293071  # BTU to kWh
            
            # Store archetype metadata in DataFrame attributes
            df.attrs['weather_zone'] = int(archetype['weather_zone'])
            df.attrs['totFloorArea'] = float(archetype['totFloorArea'])
            
            # Debug log - reduced to debug level
            self.logger.debug(
                f"Processed results for archetype {arch_dir.name} - "
                f"Zone: {df.attrs['weather_zone']}, "
                f"Floor Area: {df.attrs['totFloorArea']:.2f} m²"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing results: {str(e)}")
            return None
    
    def run_parallel_simulations(self, year: int, 
                           scenario: str = 'baseline',
                           parameters: Optional[Dict[str, Dict[int, float]]] = None,
                           output_dir: Optional[Path] = None,
                           use_stochastic_schedules: bool = False,
                           cleanup_after: Optional[bool] = None) -> Dict[int, pd.DataFrame]:
        """
        Run simulations for all archetypes in parallel.
        
        Args:
            year: Simulation year
            scenario: Scenario name (default: 'baseline')
            parameters: Optional calibration parameters by archetype
                    {param_name: {archetype_id: adjustment_factor}}
                    where adjustment_factor is between -0.3 and +0.3 (±30%)
                    representing relative changes to original parameter values
            output_dir: Optional custom output directory (for calibration iterations)
            use_stochastic_schedules: Whether to use stochastic schedules
            cleanup_after: Whether to clean up archetype folders after simulation (overrides config)
            
        Returns:
            Dictionary mapping archetype IDs to their simulation results
        """
        try:
            # Create output directory
            if output_dir is None:
                output_dir = ensure_dir(
                    config.paths['output'].get_simulation_dir(year, scenario)
                )
                
            # Determine cleanup setting
            if cleanup_after is None:
                cleanup_after = config.simulation.CLEANUP_SIMULATIONS
            
            # **** NOUVEAU: Préchargement des modèles pour tous les processus ****
            # Pour éviter que chaque processus charge sa propre copie du TransferLearningManager
            try:
                from ..calibration.transfer_learning_manager import TransferLearningManager
                # Force le chargement dans le processus parent
                transfer_manager = TransferLearningManager()
                if hasattr(transfer_manager, '_load_knowledge_base'):
                    transfer_manager._load_knowledge_base()
                if hasattr(transfer_manager, '_load_transfer_model'):
                    transfer_manager._load_transfer_model()
                
                # Définir une variable d'environnement pour indiquer aux processus enfants
                # de ne pas recharger la base de connaissances
                os.environ['TRANSFER_LEARNING_PRELOADED'] = 'TRUE'
            except Exception as e:
                self.logger.warning(f"Impossible de précharger le transfer learning manager: {str(e)}")
            
            # Prepare simulation tasks
            tasks = []
            for idx in range(len(self.archetype_manager.archetypes_df)):
                arch_dir = self.archetype_manager.prepare_archetype(
                    idx, output_dir, year, parameters,
                    use_stochastic_schedules=use_stochastic_schedules
                )
                if arch_dir:
                    tasks.append(arch_dir)
            
            # Run simulations in parallel
            self.logger.info(f"Starting {len(tasks)} simulations using {self.n_cpus} CPUs")
            with mp.Pool(self.n_cpus) as pool:
                results = pool.map(self._run_simulation, tasks)
            
            # Nettoyer la variable d'environnement après exécution
            if 'TRANSFER_LEARNING_PRELOADED' in os.environ:
                del os.environ['TRANSFER_LEARNING_PRELOADED']
            
            # Process results
            simulation_results = {}
            for idx, success in enumerate(results):
                if success:
                    arch_dir = tasks[idx]
                    results_df = self._process_simulation_results(arch_dir, year)
                    if results_df is not None:
                        simulation_results[idx] = results_df
                        
                        # Clean up if requested
                        if cleanup_after:
                            try:
                                import shutil
                                shutil.rmtree(arch_dir)
                                self.logger.debug(f"Cleaned up {arch_dir}")
                            except Exception as cleanup_e:
                                self.logger.warning(f"Failed to clean up {arch_dir}: {str(cleanup_e)}")
            
            self.logger.info(
                f"Completed {len(simulation_results)}/{len(tasks)} simulations"
            )
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Error in parallel simulation: {str(e)}")
            return {}
    
    def run_single_simulation(self, archetype_id: int, year: int,
                            scenario: str = 'baseline',
                            parameters: Optional[Dict[str, Dict[int, float]]] = None,
                            use_stochastic_schedules: bool = False) -> Optional[pd.DataFrame]:
        """
        Run simulation for a single archetype.
        
        Args:
            archetype_id: ID of the archetype to simulate
            year: Simulation year
            scenario: Scenario name
            parameters: Optional calibration parameters by archetype
                      {param_name: {archetype_id: adjustment_factor}}
                      where adjustment_factor is between -0.3 and +0.3 (±30%)
                      representing relative changes to original parameter values
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            DataFrame with simulation results or None if simulation fails
        """
        try:
            # Create output directory
            output_dir = ensure_dir(
                config.paths['output'].get_simulation_dir(year, scenario)
            )
            
            # Prepare archetype
            arch_dir = self.archetype_manager.prepare_archetype(
                archetype_id, output_dir, year, parameters,
                use_stochastic_schedules=use_stochastic_schedules
            )
            if not arch_dir:
                return None
            
            # Run simulation
            if not self._run_simulation(arch_dir):
                return None
            
            # Process results
            return self._process_simulation_results(arch_dir, year)
            
        except Exception as e:
            self.logger.error(
                f"Error simulating archetype {archetype_id}: {str(e)}"
            )
            return None
    
    def _load_latest_calibration_results(self, reference_year: int = 2023) -> Dict[str, Dict[int, float]]:
        """
        Charge les résultats de calibration les plus récents pour l'année de référence.
        
        Args:
            reference_year: Année de référence (2023 par défaut)
            
        Returns:
            Dictionnaire des paramètres calibrés {param_name: {archetype_id: value}}
        """
        try:
            # NOUVEAU: Log détaillé du processus de chargement
            self.logger.info(f"Tentative de chargement des paramètres de calibration pour l'année {reference_year}")
            
            # Chercher dans le répertoire de calibration de l'année de référence
            calibration_base_dir = config.paths['output'].CALIBRATION / str(reference_year)
            self.logger.info(f"Répertoire de calibration: {calibration_base_dir}")
            
            if not calibration_base_dir.exists():
                self.logger.warning(f"Aucun répertoire de calibration trouvé pour l'année {reference_year}")
                # NOUVEAU: Utiliser les paramètres par défaut comme solution de secours
                return self._get_default_calibration_params()
            
            # NOUVEAU: Recherche récursive de tous les fichiers calibration_results.json
            all_results_files = list(calibration_base_dir.glob("**/calibration_results.json"))
            
            if not all_results_files:
                self.logger.warning(f"Aucun fichier calibration_results.json trouvé dans {calibration_base_dir}")
                return self._get_default_calibration_params()
            
            # Trier par date de modification (plus récente en premier)
            all_results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Sélectionner le fichier le plus récent
            results_file = all_results_files[0]
            self.logger.info(f"Fichier de résultats de calibration le plus récent: {results_file}")
            
            # Charger le fichier de résultats
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extraire les paramètres calibrés
            if 'best_params' in results and isinstance(results['best_params'], dict):
                self.logger.info(f"Paramètres calibrés chargés depuis {results_file}")
                
                # NOUVEAU: Convertir les paramètres au format attendu si nécessaire
                params = self._convert_params_format(results['best_params'])
                param_count = len(params)
                
                # Compter les paramètres et les archétypes
                archetype_counts = {}
                for param_name, arch_values in params.items():
                    if isinstance(arch_values, dict):
                        archetype_counts[param_name] = len(arch_values)
                
                self.logger.debug(f"Paramètres chargés: {param_count} paramètres")
                for param, count in archetype_counts.items():
                    self.logger.debug(f"  → {param}: {count} archétypes")
                
                # Faire une vérification pour s'assurer que nous avons des paramètres valides
                if param_count == 0 or all(count == 0 for count in archetype_counts.values()):
                    self.logger.warning("Les paramètres chargés sont vides ou mal formatés, utilisation des paramètres par défaut")
                    return self._get_default_calibration_params()
                    
                return params
            else:
                self.logger.warning(f"Structure de résultats invalide dans {results_file}")
                return self._get_default_calibration_params()
                
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des résultats de calibration: {str(e)}")
            # Log plus détaillé
            import traceback
            self.logger.debug(f"Trace d'erreur: {traceback.format_exc()}")
            return self._get_default_calibration_params()
    
    def _convert_params_format(self, params: Dict) -> Dict[str, Dict[int, float]]:
        """
        Convertit les paramètres au format attendu.
        
        Args:
            params: Paramètres chargés depuis le fichier JSON
            
        Returns:
            Paramètres au format {param_name: {archetype_id: value}}
        """
        converted_params = {}
        
        # Si les paramètres sont déjà au format attendu, les retourner directement
        if all(isinstance(v, dict) for v in params.values()):
            return params
            
        # Si les paramètres sont au format {param_name: value}, les convertir
        if all(isinstance(v, (int, float)) for v in params.values()):
            self.logger.info("Conversion des paramètres plats en format par archétype")
            
            # Convertir chaque paramètre pour tous les archétypes 0-99
            for param_name, value in params.items():
                converted_params[param_name] = {}
                for arch_id in range(100):  # Supposer 100 archétypes max
                    converted_params[param_name][arch_id] = value
        
        # Si non vide, retourner les paramètres convertis
        if converted_params:
            return converted_params
            
        # Si on ne peut pas convertir, retourner les paramètres originaux
        return params
    
    def _get_default_calibration_params(self) -> Dict[str, Dict[int, float]]:
        """
        Crée des paramètres de calibration par défaut comme solution de secours.
        
        Returns:
            Dictionnaire des paramètres calibrés par défaut
        """
        # NOUVEAU: Créer un ensemble de paramètres par défaut pour s'assurer qu'il y a toujours des paramètres
        # Utiliser des valeurs conservatrices pour éviter des changements drastiques
        default_params = {
            'infiltration_rate': {},  # ACH
            'wall_rvalue': {},      # R-value des murs
            'ceiling_rvalue': {},   # R-value du plafond
            'heating_efficiency': {} # Efficacité du chauffage
        }
        
        # Appliquer ces valeurs à tous les archétypes (0-100)
        for archetype_id in range(100):
            default_params['infiltration_rate'][archetype_id] = 0.0  # Pas de modification
            default_params['wall_rvalue'][archetype_id] = 0.05      # +5% d'isolation
            default_params['ceiling_rvalue'][archetype_id] = 0.05   # +5% d'isolation
            default_params['heating_efficiency'][archetype_id] = 0.05 # +5% d'efficacité
        
        self.logger.info("Utilisation des paramètres de calibration par défaut")
        return default_params
    
    def run_parallel_simulations_for_scenario(self, 
                                    scenario_key: str,
                                    parameters: Optional[Dict[str, Dict[int, float]]] = None,
                                    output_dir: Optional[Path] = None,
                                    cleanup_after: Optional[bool] = None) -> Dict[Any, Any]:
        """
        Run simulations for a future scenario with the hybrid approach.
        
        Args:
            scenario_key: Future scenario key (e.g., '2035_warm_PV_E')
            parameters: Optional calibration parameters by archetype
            output_dir: Optional custom output directory
            cleanup_after: Whether to clean up archetype folders after simulation
            
        Returns:
            Dictionary containing simulation results and transformed archetypes
        """
        try:
            # Verify the scenario key is valid
            if not config.future_scenarios.is_future_scenario(scenario_key):
                self.logger.error(f"Invalid future scenario key: {scenario_key}")
                return {'results': {}, 'is_future_scenario': True}
            
            # Get scenario year
            year = config.future_scenarios.get_scenario_year(scenario_key)
            
            # Create output directory
            if output_dir is None:
                output_dir = ensure_dir(
                    config.paths['output'].get_simulation_dir(year, scenario_key)
                )
            
            # Set scenario key in archetype manager for weather file selection
            self.archetype_manager.current_scenario_key = scenario_key
            
            # Determine cleanup setting
            if cleanup_after is None:
                cleanup_after = config.simulation.CLEANUP_SIMULATIONS
            
            # Initialize ScenarioManager
            from ..scenario_manager import ScenarioManager
            scenario_manager = ScenarioManager()
            
            # Si no parameters provided, load latest calibration results for reference year (2023)
            if parameters is None:
                parameters = self._load_latest_calibration_results(reference_year=2023)
                self.logger.info(f"Chargement automatique des paramètres calibrés pour l'année de référence 2023")
                
                # NOUVEAU: Vérification supplémentaire pour s'assurer que des paramètres sont bien présents
                if not parameters or not isinstance(parameters, dict) or len(parameters) == 0:
                    self.logger.warning("Les paramètres chargés sont vides ou invalides, utilisation des paramètres par défaut")
                    parameters = self._get_default_calibration_params()
            
            # Vérifier que parameters est un dictionnaire valide
            if not parameters or not isinstance(parameters, dict):
                self.logger.warning(f"No valid calibration parameters found, using empty dictionary")
                adapted_params = {}
            else:
                # NOUVEAU: Log détaillé des paramètres de calibration originaux
                sample_params = {}
                for param_name, arch_values in parameters.items():
                    if isinstance(arch_values, dict):
                        # Prendre un échantillon des valeurs pour le log
                        sample_archetype_ids = list(arch_values.keys())[:3]
                        sample_params[param_name] = {
                            arch_id: arch_values[arch_id] for arch_id in sample_archetype_ids 
                            if arch_id in arch_values
                        }
                
                # Log des paramètres d'exemple
                self.logger.debug(f"Paramètres de calibration originaux (échantillon):")
                for param_name, values in sample_params.items():
                    self.logger.debug(f"  → {param_name}: {values}")
                
                # Adapt calibration parameters for future scenario
                adapted_params = scenario_manager.apply_scenario_to_calibration_params(
                    parameters, scenario_key
                )
                
                # NOUVEAU: Log détaillé des paramètres adaptés
                if isinstance(adapted_params, dict):
                    sample_adapted = {}
                    for param_name, arch_values in adapted_params.items():
                        if isinstance(arch_values, dict):
                            # Prendre les mêmes archétypes que précédemment
                            sample_adapted[param_name] = {
                                arch_id: arch_values[arch_id] for arch_id in sample_archetype_ids 
                                if arch_id in arch_values
                            }
                    
                    # Log des paramètres adaptés
                    self.logger.info(f"Paramètres de calibration adaptés pour {scenario_key} (échantillon):")
                    for param_name, values in sample_adapted.items():
                        if values:
                            self.logger.info(f"  → {param_name}: {values}")
                
                if not isinstance(adapted_params, dict):
                    self.logger.warning(f"Adapted parameters are not a valid dictionary, using empty dictionary")
                    adapted_params = {}
            
            # NOUVEAU: Sauvegarder les archétypes originaux pour comparaison
            original_archetypes_df = self.archetype_manager.archetypes_df.copy()
            
            # NOUVEAU: Transformation stratégique des archétypes
            transformed_df = scenario_manager.transform_selected_archetypes(
                self.archetype_manager.archetypes_df,
                scenario_key,
                transform_percentage=0.8
            )
            
            # NOUVEAU: Vérification avancée des transformations sur quelques attributs clés
            thermal_attributes = ['dominantWallRVal', 'dominantCeilingRVal', 'ach', 'spaceHeatingFuel', 'heatPumpType']
            changed_count = 0
            sample_changes = {}
            
            for idx in range(len(transformed_df)):
                orig = original_archetypes_df.iloc[idx]
                trans = transformed_df.iloc[idx]
                
                changes = []
                for attr in thermal_attributes:
                    if attr in orig and attr in trans:
                        old_val = orig.get(attr)
                        new_val = trans.get(attr)
                        if pd.notna(old_val) and pd.notna(new_val) and old_val != new_val:
                            changes.append((attr, old_val, new_val))
                
                if changes:
                    changed_count += 1
                    # Enregistrer un échantillon des changements (premiers 5 archétypes modifiés)
                    if len(sample_changes) < 5:
                        sample_changes[idx] = changes
            
            self.logger.info(f"Vérification des transformations pour {scenario_key}:")
            self.logger.info(f"  → {changed_count}/{len(transformed_df)} archétypes modifiés ({changed_count/len(transformed_df)*100:.1f}%)")
            
            # Afficher un échantillon des modifications
            for idx, changes in sample_changes.items():
                self.logger.debug(f"  → Archétype {idx}:")
                for attr, old_val, new_val in changes:
                    self.logger.debug(f"      {attr}: {old_val} → {new_val}")
            
            # Stocker le DataFrame transformé dans l'ArchetypeManager
            self.archetype_manager.transformed_archetypes_df = transformed_df
            
            # Préparer les tâches de simulation
            tasks = []
            for idx in range(len(self.archetype_manager.archetypes_df)):
                # Préparer l'archétype avec l'approche hybride
                arch_dir = self.archetype_manager.prepare_archetype_for_scenario(
                    idx, scenario_key, output_dir, adapted_params, use_stochastic_schedules=True
                )
                
                if arch_dir:
                    tasks.append(arch_dir)
            
            # Exécuter les simulations en parallèle
            simulation_results = {}
            if tasks:
                
                # Nombre de processeurs à utiliser
                max_workers = config.simulation.MAX_WORKERS
                if max_workers is None or max_workers <= 0:
                    import multiprocessing
                    max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))  # Utilise 80% des CPU
                
                # Préparer les arguments
                args_list = [(task, cleanup_after) for task in tasks]
                
                # Exécuter le pool de processus
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._run_simulation, *args): task for args, task in zip(args_list, tasks)}
                    
                    # Suivre l'avancement
                    completed = 0
                    for future in as_completed(futures):
                        task = futures[future]
                        completed += 1
                        
                        try:
                            # Extraire l'ID d'archétype du chemin
                            archetype_id = int(task.name.split('_')[-1])
                            success = future.result()
                            
                            if success:
                                # Traiter les résultats pour obtenir le DataFrame
                                results_df = self._process_simulation_results(task, year)
                                if results_df is not None:
                                    simulation_results[archetype_id] = results_df
                                    self.logger.info(f"Archetype {archetype_id} simulé avec succès [{completed}/{len(tasks)}]")
                                else:
                                    self.logger.warning(f"Résultats non disponibles pour l'archétype {archetype_id}")
                            else:
                                self.logger.warning(f"Échec de la simulation pour l'archétype {archetype_id}")
                        except Exception as e:
                            self.logger.error(f"Erreur lors de la simulation de {task}: {str(e)}")
            else:
                self.logger.warning(f"Aucune tâche de simulation à exécuter pour le scénario {scenario_key}")
            
            # MODIFICATION: Ne pas supprimer la référence aux archétypes transformés
            # et retourner une structure complète avec les résultats et les archétypes transformés
            return {
                'results': simulation_results,
                'transformed_archetypes': self.archetype_manager.transformed_archetypes_df,
                'is_future_scenario': True
            }
                
        except Exception as e:
            self.logger.error(f"Error in parallel simulation for scenario {scenario_key}: {str(e)}")
            # Cleanup
            if hasattr(self.archetype_manager, 'transformed_archetypes_df'):
                delattr(self.archetype_manager, 'transformed_archetypes_df')
            return {'results': {}, 'is_future_scenario': True}