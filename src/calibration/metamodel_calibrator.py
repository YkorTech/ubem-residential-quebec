"""
Module de calibration par métamodélisation pour UBEM Québec.
Utilise un design d'expériences et un métamodèle pour calibrer efficacement les paramètres.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pyDOE2 import lhs
from scipy.optimize import minimize
from scipy.stats import norm

from ..config import config
from ..utils import ensure_dir, save_results, calculate_metrics

class MetamodelCalibrator:
    """Calibrate model parameters using metamodeling approach."""
    
    def __init__(self, simulation_manager, aggregation_manager):
        """
        Initialize the metamodel calibrator.
        
        Args:
            simulation_manager: Instance of SimulationManager
            aggregation_manager: Instance of AggregationManager
        """
        self.logger = logging.getLogger("src.calibration.metamodel_calibrator")
        self.simulation_manager = simulation_manager
        self.aggregation_manager = aggregation_manager
        self.hydro_data = None
        self.metamodel = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Paramètres du métamodèle
        self.metamodel_type = "gpr"  # Options: "gpr" ou "rf"
        self.doe_size = 100  # Nombre de points dans le design d'expériences
        self.optimization_iterations = 1000  # Itérations pour l'optimisation bayésienne
        self.n_validation_points = 5  # Nombre de points à valider avec simulation complète
        self.initial_params = None  # Pour transfer learning
        
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
        return calculate_metrics(simulation_results, self.hydro_data)
    
    def _run_simulation_with_params(self, params: Dict, 
                               year: int,
                               use_stochastic_schedules: bool = False,
                               output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Run simulation with given parameters and calculate metrics.
        
        Args:
            params: Dictionary of parameter values
            year: Simulation year
            use_stochastic_schedules: Whether to use stochastic schedules
            output_dir: Optional output directory
            
        Returns:
            Dictionary of error metrics
        """
        # Convert params to nested format for simulation_manager if needed
        if not any(isinstance(v, dict) for v in params.values()):
            nested_params = self._convert_to_nested_params(params)
        else:
            nested_params = params
        
        # Generate a unique scenario name for this simulation
        scenario = f"metamodel_{datetime.now().strftime('%H%M%S')}"
        
        # Run simulations
        simulation_results = self.simulation_manager.run_parallel_simulations(
            year=year,
            scenario=scenario,
            parameters=nested_params,
            use_stochastic_schedules=use_stochastic_schedules,
            output_dir=output_dir,
            cleanup_after=True  # Activer le nettoyage
        )
        
        # Aggregate results - Skip MRC aggregation during calibration iterations
        provincial_results, _, _ = self.aggregation_manager.aggregate_results(
            year, simulation_results, scenario, 
            output_dir=output_dir,
            skip_mrc_aggregation=True  # Skip MRC aggregation to save time
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
    
    def _generate_lhs_samples(self, parameters: Dict[str, Dict], n_samples: int) -> List[Dict]:
        """
        Generate Latin Hypercube Sampling design with optional inclusion of initial parameters.
        
        Args:
            parameters: Dictionary of parameters and their bounds
            n_samples: Number of sample points
            
        Returns:
            List of parameter dictionaries for each sample
        """
        # Extract parameter names and bounds
        param_names = list(parameters.keys())
        bounds = np.array([parameters[p]['bounds'] for p in param_names])
        
        # Generate normalized LHS samples (between 0 and 1)
        lhs_samples = lhs(len(param_names), samples=n_samples, criterion="maximin")
        
        # Scale to parameter bounds
        scaled_samples = lhs_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        # Convert to list of parameter dictionaries
        samples = []
        for i in range(n_samples):
            sample = {name: scaled_samples[i, j] for j, name in enumerate(param_names)}
            samples.append(sample)
        
        # Add initial parameters if provided (replace the first sample)
        if self.initial_params is not None:
            self.logger.info("Adding predicted parameters from transfer learning to DOE")
            
            # Create clean sample with all parameters from initial_params
            initial_sample = {}
            for param in param_names:
                if param in self.initial_params:
                    value = self.initial_params[param]
                    # Ensure value is within bounds
                    min_val, max_val = parameters[param]['bounds']
                    value = max(min_val, min(max_val, value))
                    initial_sample[param] = value
                else:
                    # Use random value from bounds for missing parameters
                    min_val, max_val = parameters[param]['bounds']
                    initial_sample[param] = np.random.uniform(min_val, max_val)
            
            # Replace first sample with initial parameters
            samples[0] = initial_sample
            
            # Also add some variations around the initial parameters
            for i in range(1, min(5, n_samples)):
                variation = initial_sample.copy()
                
                # MODIFICATION ICI: Vérifier le nombre de paramètres
                # Calculer le nombre de paramètres à perturber (maximum 2, minimum 1, pas plus que disponible)
                n_params_to_perturb = min(2, len(param_names))
                
                # Perturb parameters randomly (only if there are enough parameters)
                if n_params_to_perturb > 0:
                    for param in np.random.choice(param_names, size=n_params_to_perturb, replace=False):
                        bounds = parameters[param]['bounds']
                        width = (bounds[1] - bounds[0]) * 0.1  # 10% of range
                        variation[param] += np.random.uniform(-width, width)
                        # Ensure within bounds
                        variation[param] = max(bounds[0], min(bounds[1], variation[param]))
                
                samples[i] = variation
        
        return samples
    
    def _build_metamodel(self, X: np.ndarray, y: np.ndarray, param_names: List[str] = None):
        """
        Build metamodel using training data.
        
        Args:
            X: Input feature matrix
            y: Target values
            param_names: Names of parameters (features)
        """
        # Store parameter names for later use in analyze_metamodel
        self.param_names = param_names
        
        # Standardize inputs and outputs
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        if self.metamodel_type == "gpr":
            # Gaussian Process Regression (Krigeage)
            n_dimensions = X.shape[1]
            kernel = ConstantKernel() * Matern(
                nu=2.5, 
                length_scale=np.ones(n_dimensions),  # Initialiser avec un array distinct pour chaque dimension
                length_scale_bounds=(1e-5, 1e5)
            )
            self.metamodel = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=15,  # Augmenté pour meilleure optimisation
                normalize_y=False,  # Already normalized
                random_state=42
            )
        else:
            # Random Forest Regression
            self.metamodel = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        
        # Fit the model
        self.metamodel.fit(X_scaled, y_scaled)
        
        # Log model info
        if self.metamodel_type == "gpr":
            self.logger.info(f"Fitted GPR model with kernel: {self.metamodel.kernel_}")
            # Log des longueurs caractéristiques pour diagnostic
            if hasattr(self.metamodel.kernel_, 'k2') and hasattr(self.metamodel.kernel_.k2, 'length_scale'):
                length_scales = self.metamodel.kernel_.k2.length_scale
                self.logger.info(f"Length scales after fitting: {length_scales}")
        else:
            self.logger.info(f"Fitted RF model with feature importances: {self.metamodel.feature_importances_}")
    
    def _predict_with_metamodel(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the metamodel.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Tuple of (predictions, std_devs)
        """
        X_scaled = self.scaler_X.transform(X)
        
        if self.metamodel_type == "gpr":
            y_pred_scaled, y_std_scaled = self.metamodel.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_std = y_std_scaled * self.scaler_y.scale_
            return y_pred, y_std
        else:
            y_pred_scaled = self.metamodel.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            # Random Forest doesn't provide std directly, use a constant
            return y_pred, np.ones_like(y_pred) * 0.1
    
    def _objective_function(self, x: np.ndarray, param_names: List[str]) -> float:
        """
        Objective function for optimization, using the metamodel.
        
        Args:
            x: Parameter values array
            param_names: List of parameter names
            
        Returns:
            Negative RMSE (to maximize for minimizers)
        """
        # Convert array to parameter dictionary
        params = {name: value for name, value in zip(param_names, x)}
        
        # Predict with metamodel
        X = np.array(list(params.values())).reshape(1, -1)
        y_pred, _ = self._predict_with_metamodel(X)
        
        # Return negative RMSE (we want to minimize RMSE)
        return y_pred[0]
    
    def _acquisition_function(self, x: np.ndarray, param_names: List[str], 
                            xi: float = 0.01) -> float:
        """
        Expected Improvement acquisition function for Bayesian optimization.
        
        Args:
            x: Parameter values array
            param_names: List of parameter names
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Negative expected improvement (for minimization)
        """
        # Convert array to parameter dictionary
        params = {name: value for name, value in zip(param_names, x)}
        
        # Predict with metamodel
        X = np.array(list(params.values())).reshape(1, -1)
        mu, sigma = self._predict_with_metamodel(X)
        
        # If sigma is zero, return a large negative value
        if sigma[0] <= 0.0:
            return 0.0
        
        # Calculate improvement
        imp = mu[0] - self.current_best_rmse - xi
        
        # Calculate expected improvement
        if self.metamodel_type == "gpr":
            z = imp / sigma[0]
            ei = imp * norm.cdf(z) + sigma[0] * norm.pdf(z)
        else:
            # For RF, use a simpler formula
            ei = max(0, imp)
        
        # Return negative EI (for minimization)
        return -ei
    
    def run_calibration(self, 
                       parameters: Dict[str, Dict],
                       year: int,
                       use_stochastic_schedules: bool = False,
                       output_dir: Optional[Path] = None) -> Dict:
        """
        Run metamodel-based calibration.
        
        Args:
            parameters: Dictionary of parameters to calibrate with their bounds
            year: Year to calibrate for
            use_stochastic_schedules: Whether to use stochastic schedules
            output_dir: Optional output directory
            
        Returns:
            Dictionary with calibration results
        """
        try:
            
            if self.hydro_data is None:
                self._load_hydro_data(year)
            
            # Create output directory
            if output_dir is None:
                campaign_dir = config.paths['output'].get_calibration_dir(
                    year, datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                output_dir = ensure_dir(campaign_dir)
            
            doe_dir = ensure_dir(output_dir / 'doe')
            metamodel_dir = ensure_dir(output_dir / 'metamodel')
            validation_dir = ensure_dir(output_dir / 'validation')
            
            param_names = list(parameters.keys())
            
            self.logger.info(f"Starting metamodel calibration with {len(param_names)} parameters")
            self.logger.info(f"Parameters: {param_names}")

            # Adapter la taille du DOE en fonction de l'expérience
            if hasattr(self.simulation_manager, 'transfer_learning_manager'):
                kb = self.simulation_manager.transfer_learning_manager.knowledge_base
                if kb and len(kb) > 0:
                    # Réduire progressivement la taille du DOE avec l'expérience
                    kb_size = len(kb)
                    if kb_size >= 10:
                        # Réduction drastique pour beaucoup d'expérience
                        self.doe_size = max(20, self.doe_size // 3)
                        self.logger.info(f"Taille du DOE réduite à {self.doe_size} (expérience importante)")
                    elif kb_size >= 5:
                        # Réduction moyenne
                        self.doe_size = max(30, self.doe_size // 2)
                        self.logger.info(f"Taille du DOE réduite à {self.doe_size} (expérience moyenne)")
                    elif kb_size >= 2:
                        # Légère réduction
                        self.doe_size = max(40, int(self.doe_size * 0.7))
                        self.logger.info(f"Taille du DOE réduite à {self.doe_size} (expérience limitée)")
            
            # Step 1: Generate Latin Hypercube Sampling design
            self.logger.info(f"Generating LHS design with {self.doe_size} samples")
            doe_samples = self._generate_lhs_samples(parameters, self.doe_size)
            
            # Save DOE samples
            save_results(doe_samples, doe_dir / 'doe_samples.json', 'json')
            
            # Step 2: Run simulations for DOE samples
            doe_results = []
            for i, sample in enumerate(doe_samples):
                self.logger.info(f"Running simulation {i+1}/{len(doe_samples)}")
                
                try:
                    metrics = self._run_simulation_with_params(
                        params=sample,
                        year=year,
                        use_stochastic_schedules=use_stochastic_schedules,
                        output_dir=doe_dir / f"simulation_{i}"
                    )
                    
                    # Store results with parameters
                    result = {
                        'parameters': sample,
                        'metrics': metrics,
                        'rmse': metrics['rmse'],
                        'simulation_id': i
                    }
                    doe_results.append(result)
                    
                    self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                except Exception as e:
                    self.logger.error(f"Error in simulation {i+1}: {str(e)}")
            
            # Save DOE results
            save_results(doe_results, doe_dir / 'doe_results.json', 'json')
            
            # Identify best DOE point for initial comparison
            best_doe_result = min(doe_results, key=lambda x: x['rmse'])
            self.logger.info(f"Best DOE result: RMSE={best_doe_result['rmse']:.4f}")
            
            # Step 3: Prepare data for metamodel
            X = np.array([[sample[param] for param in param_names] for sample in doe_samples])
            y = np.array([result['rmse'] for result in doe_results])
            
            # Step 4: Build metamodel
            self.logger.info(f"Building {self.metamodel_type.upper()} metamodel")
            self._build_metamodel(X, y, param_names)
            
            # Step 5: Optimize using the metamodel
            self.logger.info(f"Running optimization with {self.optimization_iterations} iterations")
            
            # Track current best for acquisition function
            self.current_best_rmse = best_doe_result['rmse']
            
            # Run multiple optimizations from different starting points
            best_predicted_params = None
            best_predicted_rmse = float('inf')
            
            # Try 10 different starting points
            for i in range(10):
                # Start from a random DOE point
                start_idx = np.random.randint(0, len(doe_samples))
                x0 = np.array([doe_samples[start_idx][param] for param in param_names])
                
                # Define bounds
                bounds = [(parameters[param]['bounds'][0], parameters[param]['bounds'][1])
                         for param in param_names]
                
                # Run optimization
                result = minimize(
                    lambda x: self._acquisition_function(x, param_names),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.optimization_iterations // 10}
                )
                
                # Convert to parameter dictionary
                optimized_params = {param: value for param, value in zip(param_names, result.x)}
                
                # Predict performance
                X_pred = np.array(list(optimized_params.values())).reshape(1, -1)
                predicted_rmse, _ = self._predict_with_metamodel(X_pred)
                
                # Check if this is the best so far
                if predicted_rmse[0] < best_predicted_rmse:
                    best_predicted_rmse = predicted_rmse[0]
                    best_predicted_params = optimized_params
                    
                self.logger.info(f"Optimization run {i+1}/10: predicted RMSE={predicted_rmse[0]:.4f}")
            
            self.logger.info(f"Best predicted: RMSE={best_predicted_rmse:.4f}")
            
            # Save metamodel results
            metamodel_results = {
                'best_predicted_params': best_predicted_params,
                'best_predicted_rmse': float(best_predicted_rmse),
                'best_doe_result': best_doe_result,
                'metamodel_type': self.metamodel_type
            }
            save_results(metamodel_results, metamodel_dir / 'metamodel_results.json', 'json')
            
            # Step 6: Validate with actual simulations
            self.logger.info("Validating top predictions with full simulations")
            
            # Create additional samples near the best predicted point for validation
            validation_samples = [best_predicted_params]
            
            # Add some variations around the best point
            for i in range(self.n_validation_points - 1):
                sample = best_predicted_params.copy()
                
                # MODIFIER CETTE PARTIE - Calculer le nombre de paramètres à perturber
                n_params_to_perturb = min(2, len(param_names))
                
                if n_params_to_perturb > 0:
                    # Perturb randomly chosen parameters
                    for param in np.random.choice(param_names, size=n_params_to_perturb, replace=False):
                        bounds = parameters[param]['bounds']
                        width = (bounds[1] - bounds[0]) * 0.1  # 10% of range
                        # Add random perturbation
                        sample[param] += np.random.uniform(-width, width)
                        # Ensure it's within bounds
                        sample[param] = max(bounds[0], min(bounds[1], sample[param]))
                
                validation_samples.append(sample)
            
            # Run validation simulations
            validation_results = []
            for i, sample in enumerate(validation_samples):
                self.logger.info(f"Running validation simulation {i+1}/{len(validation_samples)}")
                
                try:
                    metrics = self._run_simulation_with_params(
                        params=sample,
                        year=year,
                        use_stochastic_schedules=use_stochastic_schedules,
                        output_dir=validation_dir / f"validation_{i}"
                    )
                    
                    result = {
                        'parameters': sample,
                        'metrics': metrics,
                        'rmse': metrics['rmse'],
                        'validation_id': i
                    }
                    validation_results.append(result)
                    
                    self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                except Exception as e:
                    self.logger.error(f"Error in validation simulation {i+1}: {str(e)}")
            
            # Save validation results
            save_results(validation_results, validation_dir / 'validation_results.json', 'json')
            
            # Find best validated result
            best_validation_result = min(validation_results, key=lambda x: x['rmse'])
            self.logger.info(f"Best validation result: RMSE={best_validation_result['rmse']:.4f}")
            
            # Compare with best DOE result
            if best_validation_result['rmse'] < best_doe_result['rmse']:
                self.logger.info("Optimization improved RMSE by "
                               f"{best_doe_result['rmse'] - best_validation_result['rmse']:.4f}")
                best_result = best_validation_result
            else:
                self.logger.info("Best DOE point was already optimal")
                best_result = best_doe_result
            
            # Prepare final results
            calibration_results = {
                'best_params': best_result['parameters'],
                'best_metrics': best_result['metrics'],
                'doe_size': self.doe_size,
                'metamodel_type': self.metamodel_type,
                'optimization_iterations': self.optimization_iterations,
                'timestamp': datetime.now().isoformat(),
                'stochastic': use_stochastic_schedules,
                'campaign_dir': str(output_dir)
            }
            
            # Save final calibration results
            save_results(calibration_results, output_dir / 'calibration_results.json', 'json')
            
            # Run final simulation with best parameters for hourly results
            self.logger.info("Running final simulation with best parameters")
            final_simulation = self.simulation_manager.run_parallel_simulations(
                year=year,
                scenario="final",
                parameters=self._convert_to_nested_params(best_result['parameters']),
                use_stochastic_schedules=use_stochastic_schedules,
                output_dir=ensure_dir(output_dir / 'final_simulation'),
                cleanup_after=False  # Conserver cette simulation finale
            )

            # Aggregate final results
            provincial_results, _, _ = self.aggregation_manager.aggregate_results(
                year, final_simulation, "final",
                output_dir=ensure_dir(output_dir / 'final_simulation'),
                skip_mrc_aggregation=False  # Don't skip MRC aggregation for final simulation
            )

            # Save hourly results
            save_results(provincial_results, output_dir / 'hourly.csv', 'csv')

            # Clean up simulation directories to save space
            self._cleanup_simulation_directories(output_dir)

            self.logger.info("Metamodel calibration completed successfully")
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Error in metamodel calibration: {str(e)}")
            raise
            
    def analyze_metamodel(self, output_dir: Path) -> Dict:
        """
        Analyze metamodel to gain insights into parameter importance.
        
        Args:
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary with analysis results
        """
        if self.metamodel is None:
            self.logger.error("No metamodel available for analysis")
            return {}
            
        analysis_dir = ensure_dir(output_dir / 'analysis')
        param_names = self.param_names
        
        if self.metamodel_type == "gpr":
            # For GPR, analyze length scales
            try:
                # Pour un noyau produit (ConstantKernel * Matern)
                if hasattr(self.metamodel.kernel_, 'k2') and hasattr(self.metamodel.kernel_.k2, 'length_scale'):
                    length_scales = self.metamodel.kernel_.k2.length_scale
                # Pour un noyau Matern simple
                elif hasattr(self.metamodel.kernel_, 'length_scale'):
                    length_scales = self.metamodel.kernel_.length_scale
                else:
                    # Exploration de la structure du noyau pour diagnostic
                    self.logger.warning(f"Kernel structure unexpected: {self.metamodel.kernel_}")
                    self.logger.warning(f"Kernel attributes: {dir(self.metamodel.kernel_)}")
                    # Fallback: égale importance
                    length_scales = np.ones(len(param_names))
                    
                # Conversion en array si scalaire
                if np.isscalar(length_scales):
                    self.logger.warning("Length scales is a scalar, creating uniform array")
                    length_scales = np.ones(len(param_names)) * length_scales
                    
                # Inverse of length scale indicates importance
                importance = 1.0 / length_scales
                # Normalize to sum to 1
                importance = importance / importance.sum()
                
            except Exception as e:
                self.logger.error(f"Error extracting length scales: {str(e)}")
                importance = np.ones(len(param_names)) / len(param_names)
        else:
            # For RF, use built-in feature importance
            importance = self.metamodel.feature_importances_
            
        # Create importance dictionary
        importance_dict = {name: float(imp) for name, imp in zip(param_names, importance)}
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Save results
        analysis_results = {
            'parameter_importance': importance_dict,
            'sorted_importance': sorted_importance,
            'metamodel_type': self.metamodel_type
        }
        save_results(analysis_results, analysis_dir / 'metamodel_analysis.json', 'json')
        
        # Generate analysis report
        report = self._generate_analysis_report(sorted_importance)
        with open(analysis_dir / 'metamodel_analysis.md', 'w') as f:
            f.write(report)
            
        return analysis_results
    
    def _generate_analysis_report(self, sorted_importance: List[Tuple[str, float]]) -> str:
        """Generate a markdown report of metamodel analysis."""
        report = "# Analyse du métamodèle de calibration\n\n"
        report += f"Type de métamodèle: {self.metamodel_type.upper()}\n\n"
        report += "## Importance des paramètres\n\n"
        report += "| Paramètre | Importance relative |\n"
        report += "|-----------|---------------------|\n"
        
        for param, importance in sorted_importance:
            report += f"| {param} | {importance:.4f} |\n"
        
        report += "\n\n## Interprétation\n\n"
        
        if self.metamodel_type == "gpr":
            report += "Pour un modèle GPR (Krigeage), l'importance est basée sur l'inverse des longueurs caractéristiques.\n"
            report += "Une longueur caractéristique plus courte indique que le paramètre a un impact plus fort sur la sortie.\n"
        else:
            report += "Pour un modèle Random Forest, l'importance est calculée à partir de la réduction d'impureté apportée par chaque paramètre.\n"
            report += "Une valeur plus élevée indique que le paramètre contribue davantage à la prédiction.\n"
        
        return report
    
    def _cleanup_simulation_directories(self, output_dir: Path) -> None:
        """
        Clean up simulation directories after calibration is complete.
        
        Args:
            output_dir: Output directory containing simulation folders
        """
        try:
            # Clean up DOE directories
            doe_dir = output_dir / 'doe'
            if doe_dir.exists():
                for sim_dir in doe_dir.glob("simulation_*"):
                    if sim_dir.is_dir():
                        import shutil
                        shutil.rmtree(sim_dir)
                        self.logger.debug(f"Cleaned up {sim_dir}")
            
            # Clean up validation directories
            validation_dir = output_dir / 'validation'
            if validation_dir.exists():
                for val_dir in validation_dir.glob("validation_*"):
                    if val_dir.is_dir():
                        import shutil
                        shutil.rmtree(val_dir)
                        self.logger.debug(f"Cleaned up {val_dir}")
                        
            self.logger.info(f"Cleaned up simulation directories in {output_dir}")
        except Exception as e:
            self.logger.error(f"Error cleaning up simulation directories: {str(e)}")