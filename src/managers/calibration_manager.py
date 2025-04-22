"""
CalibrationManager handles calibration of the model using various approaches:
1. Sensitivity analysis to identify influential parameters
2. Metamodel-based calibration for efficient parameter estimation
3. Hierarchical multi-level calibration for improved accuracy
4. Transfer learning for accelerated calibration of new years/scenarios

The PSO+MCMC approach has been removed in favor of more efficient methods.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import glob
import json

import pandas as pd
import numpy as np

from ..config import config
from ..utils import ensure_dir, save_results, process_hydro_data, calculate_metrics
from .simulation_manager import SimulationManager
from .aggregation_manager import AggregationManager
# Supprimé: import des calibrateurs PSO et MCMC
from ..calibration.sensitivity_analyzer import SensitivityAnalyzer
from ..calibration.metamodel_calibrator import MetamodelCalibrator
from ..calibration.hierarchical_calibrator import HierarchicalCalibrator
from ..calibration.transfer_learning_manager import TransferLearningManager

class CalibrationManager:
    """Manages calibration of the UBEM model using various approaches."""
    
    def __init__(self):
        """Initialize the CalibrationManager."""
        self.logger = logging.getLogger(__name__)
        self.simulation_manager = SimulationManager()
        self.aggregation_manager = AggregationManager(self.simulation_manager)
        # Supprimé: initialisation des calibrateurs PSO et MCMC
        self.sensitivity_analyzer = SensitivityAnalyzer(
            self.simulation_manager,
            self.aggregation_manager
        )
        self.metamodel_calibrator = MetamodelCalibrator(
            self.simulation_manager,
            self.aggregation_manager
        )
        self.hierarchical_calibrator = HierarchicalCalibrator(
            self.simulation_manager,
            self.aggregation_manager,
            self.metamodel_calibrator
        )
        # Initialize transfer learning manager
        self.transfer_learning_manager = TransferLearningManager()
        self.hydro_data = None
        self.current_campaign = None
    
    def _load_hydro_data(self, year: int) -> None:
        """
        Load and process Hydro-Quebec consumption data.
        
        Args:
            year: Year to load data for
        """
        try:
            hydro_file = config.paths['input'].get_hydro_file(year)
            raw_data = pd.read_csv(hydro_file)
            self.hydro_data = process_hydro_data(raw_data, year)
            
        except Exception as e:
            self.logger.error(f"Error loading Hydro-Quebec data: {str(e)}")
            raise
    
    def _initialize_campaign(self, year: int) -> Path:
        """
        Initialize a new calibration campaign.
        
        Args:
            year: Calibration year
            
        Returns:
            Path to campaign directory
        """
        campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_campaign = config.paths['output'].get_calibration_dir(
            year, campaign_id
        )
        return ensure_dir(self.current_campaign)
    
    def _simulate_with_params(self, params: Dict[str, Dict[int, float]], 
                            year: int,
                            use_stochastic_schedules: bool = False,
                            output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Run simulation with given parameters and return provincial results.
        
        Args:
            params: Parameter values by archetype {param_name: {archetype_id: value}}
            year: Simulation year
            use_stochastic_schedules: Whether to use stochastic schedules
            output_dir: Optional output directory
            
        Returns:
            DataFrame with provincial results
        """
        # Create scenario name
        scenario = f"calib_{datetime.now().strftime('%H%M%S')}"
        
        # Use provided output directory or create one
        if output_dir is None and self.current_campaign is not None:
            output_dir = ensure_dir(self.current_campaign / 'simulations')
        
        try:
            # Run simulations
            simulation_results = self.simulation_manager.run_parallel_simulations(
                year=year,
                scenario=scenario,
                parameters=params,
                output_dir=output_dir,
                use_stochastic_schedules=use_stochastic_schedules
            )
            
            # Aggregate results
            provincial_results, _, _ = self.aggregation_manager.aggregate_results(
                year, simulation_results, scenario, 
                output_dir=output_dir,
                skip_mrc_aggregation=True  # Skip MRC aggregation during calibration
            )
            
            return provincial_results
            
        except Exception as e:
            self.logger.error(f"Error in simulation with parameters: {str(e)}")
            raise
    
    def validate_on_year(self, params: Dict[str, Dict[int, float]], 
                       year: int,
                       use_stochastic_schedules: bool = False) -> Dict[str, Any]:
        """
        Validate parameters on a specific year.
        
        Args:
            params: Parameters by archetype
            year: Year to validate on
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Load year Hydro data
            hydro_file = config.paths['input'].get_hydro_file(year)
            hydro_data = pd.read_csv(hydro_file)
            hydro_data = process_hydro_data(hydro_data, year)
            
            # Create validation directory
            if self.current_campaign:
                validation_dir = ensure_dir(self.current_campaign / 'validation')
                output_dir = validation_dir / str(year)
            else:
                output_dir = None
            
            # Run simulation with parameters
            results = self._simulate_with_params(
                params=params,
                year=year,
                use_stochastic_schedules=use_stochastic_schedules,
                output_dir=output_dir
            )
            
            # Calculate validation metrics
            validation_metrics = calculate_metrics(results, hydro_data)
            
            # Add validation info
            validation_metrics.update({
                'year': year,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save validation results
            if output_dir:
                save_results(
                    validation_metrics,
                    output_dir / f'metrics.json',
                    'json'
                )
                save_results(
                    results,
                    output_dir / f'hourly.csv',
                    'csv'
                )
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            return None
    
    # Supprimé: Méthode run_calibration pour PSO+MCMC
    
    def run_transfer_learning_calibration(self, 
                                       parameters: Dict[str, Dict],
                                       year: int,
                                       doe_size: int = 50,  # Réduit car meilleur point de départ
                                       metamodel_type: str = "gpr",
                                       use_stochastic_schedules: bool = False) -> Dict:
        """
        Run calibration with transfer learning from previous calibrations.
        
        Args:
            parameters: Dictionary of parameters to calibrate
            year: Year to calibrate for
            doe_size: Size of design of experiments
            metamodel_type: Type of metamodel ("gpr" or "rf")
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Dictionary with calibration results
        """
        # Initialize campaign
        campaign_dir = self._initialize_campaign(year)
        transfer_dir = ensure_dir(campaign_dir / 'transfer_learning')
        
        # Check if we can apply transfer learning
        self.logger.info("Checking for previous calibrations...")
        
        # Discover previous calibrations if needed
        if len(self.transfer_learning_manager.knowledge_base) == 0:
            discovered = self.transfer_learning_manager.discover_previous_calibrations()
            self.logger.info(f"Discovered {discovered} previous calibrations")
        
        # Get predictions from transfer learning model
        self.logger.info("Predicting starting parameters from previous calibrations")
        predicted_params = self.transfer_learning_manager.predict_optimal_parameters(
            parameters=parameters,
            year=year,
            use_stochastic_schedules=use_stochastic_schedules
        )
        
        # Save predictions
        save_results(predicted_params, transfer_dir / 'predicted_params.json', 'json')
        
        # Configure MetamodelCalibrator with initial parameters
        self.metamodel_calibrator.doe_size = doe_size
        self.metamodel_calibrator.metamodel_type = metamodel_type
        self.metamodel_calibrator.initial_params = predicted_params
        
        # Load Hydro-Quebec data if needed
        if self.hydro_data is None:
            self._load_hydro_data(year)
        self.metamodel_calibrator.hydro_data = self.hydro_data
        
        # Run calibration
        self.logger.info(
            f"Starting transfer learning calibration with {len(parameters)} parameters, "
            f"DOE size {doe_size}, metamodel type {metamodel_type.upper()}"
        )
        
        calibration_results = self.metamodel_calibrator.run_calibration(
            parameters=parameters,
            year=year,
            use_stochastic_schedules=use_stochastic_schedules,
            output_dir=campaign_dir
        )
        
        # Add calibration result to knowledge base
        self.transfer_learning_manager.add_calibration_result(
            optimal_params=calibration_results['best_params'],
            metrics=calibration_results['best_metrics'],
            year=year,
            use_stochastic_schedules=use_stochastic_schedules
        )
        
        # Analyze knowledge base
        analysis_results = self.transfer_learning_manager.analyze_knowledge_base(
            output_dir=transfer_dir
        )
        
        # Log results
        self.logger.info("Transfer learning calibration completed")
        self.logger.info(f"Best RMSE: {calibration_results['best_metrics']['rmse']:.4f}")
        
        # Save current campaign information
        self.current_campaign = campaign_dir
        
        return calibration_results
    
    def get_calibration_results(self, campaign_id: str) -> Dict:
        """
        Get results for a specific calibration campaign.
        
        Args:
            campaign_id: ID of the calibration campaign
            
        Returns:
            Dictionary with campaign results
        """
        try:
            campaign_dir = self.current_campaign
            if campaign_dir is None or campaign_dir.name != campaign_id:
                campaign_dir = next(
                    p for p in config.paths['output'].CALIBRATION.iterdir()
                    if p.name == campaign_id
                )
            
            results = {
                'campaign_id': campaign_id,
                'results': pd.read_json(campaign_dir / 'results' / 'calibration.json'),
                'hourly': pd.read_csv(campaign_dir / 'results' / 'hourly.csv')
            }
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Error loading calibration results: {str(e)}"
            )
            return None
    
    def run_sensitivity_analysis(self, 
                               param_subset=None,
                               n_trajectories=10,
                               year=2022) -> Dict:
        """
        Run sensitivity analysis to identify influential parameters.
        
        Args:
            param_subset: Optional subset of parameters to analyze
            n_trajectories: Number of Morris trajectories
            year: Year to analyze
            
        Returns:
            Dictionary with sensitivity results
        """
        # Initialize campaign
        campaign_dir = self._initialize_campaign(year)
        sensitivity_dir = ensure_dir(campaign_dir / 'sensitivity')
        
        # Run sensitivity analysis
        self.logger.info("Starting sensitivity analysis")
        
        # Load Hydro-Quebec data if needed
        if self.hydro_data is None:
            self._load_hydro_data(year)
        self.sensitivity_analyzer.hydro_data = self.hydro_data
        
        sensitivity_results = self.sensitivity_analyzer.run_morris_analysis(
            param_subset=param_subset,
            n_trajectories=n_trajectories,
            year=year,
            output_dir=sensitivity_dir
        )
        
        # Log top parameters
        self.logger.info("Sensitivity analysis completed")
        self.logger.info("Top 5 parameters:")
        for i, param in enumerate(sensitivity_results['param_sensitivity'][:5]):
            self.logger.info(f"  {i+1}. {param['display_name']} (influence: {param['mu_star']:.4f})")
        
        return sensitivity_results
    
    def load_latest_sensitivity_results(self) -> Optional[Dict]:
        """
        Load the most recent sensitivity analysis results.
        
        Returns:
            Dictionary with sensitivity results or None if not found
        """
        try:
            # Find all sensitivity results
            sensitivity_files = []
            for year_dir in config.paths['output'].CALIBRATION.glob('*'):
                if year_dir.is_dir():
                    for campaign_dir in year_dir.glob('*'):
                        if campaign_dir.is_dir():
                            sensitivity_file = campaign_dir / 'sensitivity' / 'sensitivity_results.json'
                            if sensitivity_file.exists():
                                sensitivity_files.append((sensitivity_file, campaign_dir.stat().st_mtime))
            
            if not sensitivity_files:
                self.logger.warning("No sensitivity analysis results found")
                return None
            
            # Sort by modification time (newest first)
            sensitivity_files.sort(key=lambda x: x[1], reverse=True)
            latest_file = sensitivity_files[0][0]
            
            # Load results
            with open(latest_file, 'r') as f:
                param_sensitivity = json.load(f)
            
            # Reconstruct full results
            results = {
                'param_sensitivity': param_sensitivity,
                'top_parameters': [p['name'] for p in param_sensitivity[:5]]
            }
            
            self.logger.info(f"Loaded sensitivity results from {latest_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading sensitivity results: {str(e)}")
            return None
    
    def run_metamodel_calibration(self, 
                                parameters: Dict[str, Dict],
                                year: int,
                                doe_size: int = 100,
                                metamodel_type: str = "gpr",
                                use_stochastic_schedules: bool = False) -> Dict:
        """
        Run metamodel-based calibration.
        
        Args:
            parameters: Dictionary of parameters to calibrate with their bounds
            year: Year to calibrate for
            doe_size: Size of design of experiments
            metamodel_type: Type of metamodel ("gpr" or "rf")
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Dictionary with calibration results
        """
        # Initialize campaign
        campaign_dir = self._initialize_campaign(year)
        
        # Configure metamodel calibrator
        self.metamodel_calibrator.doe_size = doe_size
        self.metamodel_calibrator.metamodel_type = metamodel_type
        
        # Load Hydro-Quebec data if needed
        if self.hydro_data is None:
            self._load_hydro_data(year)
        self.metamodel_calibrator.hydro_data = self.hydro_data
        
        # Run calibration
        self.logger.info(
            f"Starting metamodel calibration with {len(parameters)} parameters, "
            f"DOE size {doe_size}, metamodel type {metamodel_type.upper()}"
        )
        
        calibration_results = self.metamodel_calibrator.run_calibration(
            parameters=parameters,
            year=year,
            use_stochastic_schedules=use_stochastic_schedules,
            output_dir=campaign_dir
        )
        
        # Analyze metamodel for parameter importance
        analysis_results = self.metamodel_calibrator.analyze_metamodel(
            output_dir=campaign_dir
        )
        
        # Log results
        self.logger.info("Metamodel calibration completed")
        self.logger.info(f"Best RMSE: {calibration_results['best_metrics']['rmse']:.4f}")
        
        # Save current campaign information
        self.current_campaign = campaign_dir
        
        return calibration_results
        
    def run_hierarchical_calibration(self, 
                                   parameters: Dict[str, Dict],
                                   year: int,
                                   use_stochastic_schedules: bool = False) -> Dict:
        """
        Run hierarchical multi-level calibration.
        
        Args:
            parameters: Dictionary of parameters to calibrate
            year: Year to calibrate for
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Dictionary with calibration results
        """
        # Initialize campaign
        campaign_dir = self._initialize_campaign(year)
        
        # Load sensitivity results if available
        sensitivity_results = self.load_latest_sensitivity_results()
        
        # Load transfer learning predictions if available
        transfer_params = None
        if hasattr(self, 'transfer_learning_manager'):

            self.hierarchical_calibrator.simulation_manager.transfer_learning_manager = self.transfer_learning_manager
            transfer_params = self.transfer_learning_manager.predict_optimal_parameters(
                parameters=parameters,
                year=year,
                use_stochastic_schedules=use_stochastic_schedules
            )
            self.logger.info("Paramètres initiaux obtenus du transfer learning")
        
        # Load Hydro-Quebec data if needed
        if self.hydro_data is None:
            self._load_hydro_data(year)
        self.hierarchical_calibrator.hydro_data = self.hydro_data
        
        # Run calibration
        self.logger.info(
            f"Starting hierarchical calibration with {len(parameters)} parameters"
        )
        
        calibration_results = self.hierarchical_calibrator.run_calibration(
            parameters=parameters,
            year=year,
            use_stochastic_schedules=use_stochastic_schedules,
            output_dir=campaign_dir,
            sensitivity_results=sensitivity_results,
            transfer_learning_params=transfer_params
        )
        
        # Add calibration result to knowledge base if available
        if hasattr(self, 'transfer_learning_manager') and calibration_results:
            self.transfer_learning_manager.add_calibration_result(
                optimal_params=calibration_results['best_params'],
                metrics=calibration_results['best_metrics'],
                year=year,
                use_stochastic_schedules=use_stochastic_schedules
            )
        
        # Log results
        self.logger.info("Hierarchical calibration completed")
        self.logger.info(f"Final RMSE: {calibration_results['best_metrics']['rmse']:.4f}")
        
        # Save current campaign information
        self.current_campaign = campaign_dir
        
        return calibration_results