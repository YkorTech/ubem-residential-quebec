"""
Main entry point for UBEM Quebec residential sim and calib.
"""
import argparse
import logging
from pathlib import Path
import sys

from .managers.simulation_manager import SimulationManager
from .managers.calibration_manager import CalibrationManager
from .dashboard import run_dashboard
from .config import config
from .managers.aggregation_manager import AggregationManager

def setup_logging(level: str = 'INFO') -> None:
    """Setup logging config."""
    # Config root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Config formatter
    formatter = logging.Formatter(
        '%(asctime)s%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid dupes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handler
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('src.utils').setLevel(logging.WARNING)
    logging.getLogger('src.managers.simulation_manager').setLevel(logging.INFO)
    logging.getLogger('src.managers.aggregation_manager').setLevel(logging.WARNING)

def run_simulation(args):
    """Run simulation mode."""
    simulation_manager = SimulationManager()
    aggregation_manager = AggregationManager(simulation_manager)
    
    # If a future scenario is specified
    if args.future_scenario:
        simulation_data = simulation_manager.run_parallel_simulations_for_scenario(
            scenario_key=args.future_scenario
        )
        
        if simulation_data and ('results' in simulation_data and simulation_data['results']):
            # Extract year from scenario
            year = config.future_scenarios.get_scenario_year(args.future_scenario)
            
            # Aggregate results
            provincial_results, zonal_results, mrc_results = aggregation_manager.aggregate_results(
                year=year,
                simulation_results=simulation_data,
                scenario=args.future_scenario
            )
            
            # Log success
            if provincial_results is not None:
                logging.info(
                    f"Sim and agg completed for future scenario {args.future_scenario}. "
                    f"Results saved in data/outputs/simulations/{year}/{args.future_scenario}/"
                )
            else:
                logging.warning(
                    f"Sim completed but agg failed for future scenario {args.future_scenario}."
                )
        else:
            logging.error(f"Sim failed for future scenario {args.future_scenario}")
            sys.exit(1)
    else:
        # Regular simulation
        results = simulation_manager.run_parallel_simulations(
            year=args.year,
            scenario=args.scenario,
            use_stochastic_schedules=args.stochastic
        )
        if results:
            # Aggregate results
            provincial_results, zonal_results, mrc_results = aggregation_manager.aggregate_results(
                year=args.year,
                simulation_results=results,
                scenario=args.scenario
            )
            
            # Log success
            if provincial_results is not None:
                logging.info(
                    f"Sim and agg completed. "
                    f"Results saved in data/outputs/simulations/{args.year}/{args.scenario}/"
                )
            else:
                logging.warning(
                    f"Sim completed but agg failed."
                )
        else:
            logging.error("Sim failed")
            sys.exit(1)

def run_sensitivity_analysis(args):
    """Run sensitivity analysis mode."""
    calibration_manager = CalibrationManager()
    results = calibration_manager.run_sensitivity_analysis(
        param_subset=args.params,
        n_trajectories=args.trajectories,
        year=args.year
    )
    if results:
        logging.info(
            f"Sensitivity analysis completed. Results saved in "
            f"data/outputs/calibration/{args.year}/<timestamp>/sensitivity/"
        )
    else:
        logging.error("Sensitivity analysis failed")
        sys.exit(1)

def run_metamodel_calibration(args):
    """Run metamodel calibration mode."""
    calibration_manager = CalibrationManager()
    
    # Determine params to calibrate
    if args.parameters == 'sensitivity':
        # Load results from recent sensitivity analysis
        sensitivity_results = calibration_manager.load_latest_sensitivity_results()
        if not sensitivity_results:
            logging.error("No sensitivity analysis results found. Run 'sensitivity' command first.")
            sys.exit(1)
            
        # Use top 5 parameters
        parameters = {}
        for param in sensitivity_results['top_parameters'][:5]:
            parameters[param] = {
                'name': param,
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    else:
        # Parse comma-separated list
        param_names = args.parameters.split(',')
        parameters = {}
        for param in param_names:
            parameters[param.strip()] = {
                'name': param.strip(),
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    
    logging.info(f"Calibrating {len(parameters)} parameters: {list(parameters.keys())}")
    
    results = calibration_manager.run_metamodel_calibration(
        parameters=parameters,
        year=args.year,
        doe_size=args.doe_size,
        metamodel_type=args.metamodel,
        use_stochastic_schedules=args.stochastic
    )
    
    if results:
        logging.info(
            f"Metamodel calib completed. Results saved in "
            f"data/outputs/calibration/{args.year}/<timestamp>/"
        )
    else:
        logging.error("Metamodel calib failed")
        sys.exit(1)

def run_transfer_learning_calibration(args):
    """Run calib with transfer learning from previous calibs."""
    calibration_manager = CalibrationManager()
    
    # Determine params to calibrate
    if args.parameters == 'sensitivity':
        # Load results from recent sensitivity analysis
        sensitivity_results = calibration_manager.load_latest_sensitivity_results()
        if not sensitivity_results:
            logging.error("No sensitivity analysis results found. Run 'sensitivity' command first.")
            sys.exit(1)
            
        # Use top 5 parameters
        parameters = {}
        for param in sensitivity_results['top_parameters'][:5]:
            parameters[param] = {
                'name': param,
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    else:
        # Parse comma-separated list
        param_names = args.parameters.split(',')
        parameters = {}
        for param in param_names:
            parameters[param.strip()] = {
                'name': param.strip(),
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    
    logging.info(f"Calibrating {len(parameters)} parameters: {list(parameters.keys())}")
    
    results = calibration_manager.run_transfer_learning_calibration(
        parameters=parameters,
        year=args.year,
        doe_size=args.doe_size,
        metamodel_type=args.metamodel,
        use_stochastic_schedules=args.stochastic
    )
    
    if results:
        logging.info(
            f"Transfer learning calib completed. Results saved in "
            f"data/outputs/calibration/{args.year}/<timestamp>/"
        )
    else:
        logging.error("Transfer learning calib failed")
        sys.exit(1)

def run_hierarchical_calibration(args):
    """Run hierarchical multi-level calibration mode."""
    calibration_manager = CalibrationManager()
    
    # Determine params to calibrate
    if args.parameters == 'all':
            # Full params list
            parameters = {
                'infiltration_rate': {'name': 'Infiltration Rate', 'bounds': [-0.3, 0.3]},
                'wall_rvalue': {'name': 'Wall R-Value', 'bounds': [-0.3, 0.3]},
                'ceiling_rvalue': {'name': 'Ceiling R-Value', 'bounds': [-0.3, 0.3]},
                'window_ufactor': {'name': 'Window U-Factor', 'bounds': [-0.3, 0.3]},
                'heating_efficiency': {'name': 'Heating Efficiency', 'bounds': [-0.3, 0.3]},
                'heating_setpoint': {'name': 'Heating Setpoint', 'bounds': [-0.2, 0.2]},
                'cooling_setpoint': {'name': 'Cooling Setpoint', 'bounds': [-0.2, 0.2]},
                # Schedule params
                'occupancy_scale': {'name': 'Occupancy Factor', 'bounds': [-0.3, 0.3]},
                'lighting_scale': {'name': 'Lighting Factor', 'bounds': [-0.3, 0.3]},
                'appliance_scale': {'name': 'Appliance Factor', 'bounds': [-0.3, 0.3]},
                'temporal_diversity': {'name': 'Temporal Diversity', 'bounds': [0.0, 1.0]}
            }
    elif args.parameters == 'sensitivity':
        # Load results from recent sensitivity analysis
        sensitivity_results = calibration_manager.load_latest_sensitivity_results()
        if not sensitivity_results:
            logging.error("No sensitivity analysis results found. Run 'sensitivity' command first.")
            sys.exit(1)
            
        # Use top 7 parameters
        parameters = {}
        for param in sensitivity_results['top_parameters'][:7]:
            parameters[param] = {
                'name': param,
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    else:
        # Parse comma-separated list
        param_names = args.parameters.split(',')
        parameters = {}
        for param in param_names:
            parameters[param.strip()] = {
                'name': param.strip(),
                'bounds': [-0.3, 0.3]  # Default bounds
            }
    
    logging.info(f"Calibrating {len(parameters)} parameters: {list(parameters.keys())}")
    
    results = calibration_manager.run_hierarchical_calibration(
        parameters=parameters,
        year=args.year,
        use_stochastic_schedules=args.stochastic
    )
    
    if results:
        logging.info(
            f"Hierarchical calib completed. Results saved in "
            f"data/outputs/calibration/{args.year}/<timestamp>/"
        )
    else:
        logging.error("Hierarchical calib failed")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='UBEM Quebec Residential Simulation and Calibration'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Simulation command
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Run simulation mode'
    )
    sim_parser.add_argument(
        '--year',
        type=int,
        default=2023,  # Reference year
        help='Year to simulate'
    )
    sim_parser.add_argument(
        '--scenario',
        default='baseline',
        help='Scenario name'
    )
    sim_parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic schedules for occupancy and energy use'
    )
    # Add future scenario option
    sim_parser.add_argument(
        '--future-scenario',
        type=str,
        choices=config.future_scenarios.SCENARIOS,
        help='Future scenario to simulate'
    )
    sim_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Sensitivity analysis command
    sens_parser = subparsers.add_parser(
        'sensitivity',
        help='Run sensitivity analysis to identify influential parameters'
    )
    sens_parser.add_argument(
        '--year',
        type=int,
        default=2022,
        help='Year to analyze'
    )
    sens_parser.add_argument(
        '--trajectories',
        type=int,
        default=10,
        help='Number of Morris trajectories'
    )
    sens_parser.add_argument(
        '--params',
        nargs='+',
        help='Specific parameters to analyze (optional)'
    )
    sens_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Metamodel calibration command
    meta_parser = subparsers.add_parser(
        'metamodel',
        help='Run metamodel-based calibration'
    )
    meta_parser.add_argument(
        '--year',
        type=int,
        default=2022,
        help='Year to calibrate'
    )
    meta_parser.add_argument(
        '--doe-size',
        type=int,
        default=100,
        help='Size of design of experiments (number of simulations)'
    )
    meta_parser.add_argument(
        '--metamodel',
        choices=['gpr', 'rf'],
        default='gpr',
        help='Type of metamodel (Gaussian Process or Random Forest)'
    )
    meta_parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic schedules for occupancy and energy use'
    )
    meta_parser.add_argument(
        '--parameters',
        default='sensitivity',
        help='Either "sensitivity" to use top parameters from sensitivity analysis, '
             'or a comma-separated list of parameter names'
    )
    meta_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Hierarchical calibration command
    hier_parser = subparsers.add_parser(
        'hierarchical',
        help='Run hierarchical multi-level calibration'
    )
    hier_parser.add_argument(
        '--year',
        type=int,
        default=2022,
        help='Year to calibrate'
    )
    hier_parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic schedules for occupancy and energy use'
    )
    hier_parser.add_argument(
        '--parameters',
        default='all',
        help='Either "all" to use all parameters, "sensitivity" to use top parameters '
             'from sensitivity analysis, or a comma-separated list of parameter names'
    )
    hier_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Transfer learning calibration command
    transfer_parser = subparsers.add_parser(
        'transfer',
        help='Run calibration with transfer learning from previous calibrations'
    )
    transfer_parser.add_argument(
        '--year',
        type=int,
        default=2022,
        help='Year to calibrate'
    )
    transfer_parser.add_argument(
        '--doe-size',
        type=int,
        default=50,
        help='Size of design of experiments (number of simulations)'
    )
    transfer_parser.add_argument(
        '--metamodel',
        choices=['gpr', 'rf'],
        default='gpr',
        help='Type of metamodel (Gaussian Process or Random Forest)'
    )
    transfer_parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic schedules for occupancy and energy use'
    )
    transfer_parser.add_argument(
        '--parameters',
        default='sensitivity',
        help='Either "sensitivity" to use top parameters from sensitivity analysis, '
             'or a comma-separated list of parameter names'
    )
    transfer_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Knowledge base management command
    kb_parser = subparsers.add_parser(
        'knowledge',
        help='Manage knowledge base of previous calibrations'
    )
    kb_parser.add_argument(
        '--discover',
        action='store_true',
        help='Discover previous calibrations'
    )
    kb_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze knowledge base'
    )
    kb_parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for analysis results'
    )
    kb_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    # Dashboard command
    dash_parser = subparsers.add_parser(
        'dashboard',
        help='Launch interactive dashboard'
    )
    dash_parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run dashboard on'
    )
    dash_parser.add_argument(
        '--debug',
        action='store_true',
        help='Run dashboard in debug mode'
    )
    dash_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
        type=str.upper
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        if args.command == 'simulate':
            run_simulation(args)
        elif args.command == 'sensitivity':
            run_sensitivity_analysis(args)
        elif args.command == 'metamodel':
            run_metamodel_calibration(args)
        elif args.command == 'hierarchical':
            run_hierarchical_calibration(args)
        elif args.command == 'transfer':
            run_transfer_learning_calibration(args)
        elif args.command == 'knowledge':
            calibration_manager = CalibrationManager()
            if args.discover:
                # Discover previous calibrations
                n_discovered = calibration_manager.transfer_learning_manager.discover_previous_calibrations()
                logging.info(f"Discovered {n_discovered} new calibrations")
            
            if args.analyze:
                # Analyze knowledge base
                output_dir = args.output_dir if args.output_dir else None
                if output_dir:
                    output_dir = Path(output_dir)
                analysis_results = calibration_manager.transfer_learning_manager.analyze_knowledge_base(
                    output_dir=output_dir
                )
                
                if analysis_results:
                    logging.info(f"Knowledge base analysis completed. {analysis_results['n_entries']} entries analyzed.")
                else:
                    logging.error("Knowledge base analysis failed")
                    sys.exit(1)
        elif args.command == 'dashboard':
            run_dashboard(debug=args.debug, port=args.port)
            
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
