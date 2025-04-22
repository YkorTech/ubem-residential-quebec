"""
Config module for Quebec Residential UBEM.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from dataclasses import dataclass

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

@dataclass
class InputPaths:
    """Input data paths"""
    ROOT: Path = DATA_DIR / 'inputs'
    ARCHETYPES: Path = ROOT / 'archetypes'
    WEATHER: Path = ROOT / 'weather'
    HYDRO: Path = ROOT / 'hydro'
    EVALUATION: Path = ROOT / 'evaluation'
    ZONES_SHP: Path = ROOT / 'zone_shp'
    
    # Key files
    BASE_ARCHETYPES: Path = ARCHETYPES /  "selected_archetypes_future_zones.csv"
    ZONES_SHAPEFILE: Path = ZONES_SHP / 'quebec_scenario_zones.shp'
    
    # Future weather paths
    WEATHER_FUTURE: Path = WEATHER / 'future'
    WEATHER_FUTURE_MAPPING: Path = WEATHER_FUTURE / 'zones_futur_fichiers_epw.csv'
    
    def get_weather_mapping(self, year: int = None) -> Path:
        """Get weather mapping file path"""
        return self.WEATHER / 'historic' / 'h2k_epw_avec_zones_meteo.csv'
    
    def get_weather_dir(self, year: int = None) -> Path:
        """Get weather dir"""
        return self.WEATHER / 'historic' / 'epw'
    
    def get_evaluation_file(self, year: int) -> Path:
        """Get eval file (using 2024 data)"""
        return self.EVALUATION / '2024_future_weather_zones.csv'
    
    def get_hydro_file(self, year: int) -> Path:
        """Get HQ consumption file"""
        return self.HYDRO / f'conso_residentielle_{year}.csv'
        
    def get_future_weather_dir(self, climate_type: str) -> Path:
        """Get future weather dir for climate type"""
        return self.WEATHER_FUTURE / climate_type
    
    def get_future_weather_file(self, zone: int, scenario_key: str) -> Optional[Path]:
        """Get future weather file for scenario/zone"""
        # Init ScenarioManager
        from .scenario_manager import ScenarioManager
        scenario_manager = ScenarioManager()
        
        # Get weather file path
        relative_path = scenario_manager.get_weather_file(zone, scenario_key)
        if not relative_path:
            return None
            
        return self.WEATHER / relative_path

@dataclass
class OpenStudioPaths:
    """OpenStudio paths"""
    ROOT: Path = DATA_DIR / 'openstudio'
    MEASURES: Path = ROOT / 'measures'
    TEMPLATES: Path = ROOT / 'templates'
    
    # Measure paths
    BUILD_RESIDENTIAL: Path = MEASURES / 'BuildResidentialHPXML'
    BUILD_SCHEDULE: Path = MEASURES / 'BuildResidentialScheduleFile'
    HPXML_TO_OS: Path = MEASURES / 'HPXMLtoOpenStudio'
    
    # Template files
    BASE_WORKFLOW: Path = TEMPLATES / 'base_workflow.osw'
    STOCHASTIC_WORKFLOW: Path = TEMPLATES / 'stochastic_workflow.osw'

@dataclass
class OutputPaths:
    """Output data paths"""
    ROOT: Path = DATA_DIR / 'outputs'
    SIMULATIONS: Path = ROOT / 'simulations'
    CALIBRATION: Path = ROOT / 'calibration'
    
    def get_simulation_dir(self, year: int, scenario: str = 'baseline') -> Path:
        """Get sim dir for year/scenario"""
        return self.SIMULATIONS / str(year) / scenario
    
    def get_simulation_figures_dir(self, year: int, scenario: str = 'baseline') -> Path:
        """Get figures dir for sim"""
        return self.get_simulation_dir(year, scenario) / 'figures'
    
    def get_simulation_profiles_dir(self, year: int, scenario: str = 'baseline') -> Path:
        """Get profiles figures dir"""
        return self.get_simulation_figures_dir(year, scenario) / 'profiles'
    
    def get_calibration_dir(self, year: int, campaign_id: str) -> Path:
        """Get calib dir for year/campaign"""
        return self.CALIBRATION / str(year) / campaign_id
    
    def get_calibration_figures_dir(self, year: int, campaign_id: str) -> Path:
        """Get figures dir for calib campaign"""
        return self.get_calibration_dir(year, campaign_id) / 'figures'
    
    def get_calibration_seasonal_dir(self, year: int, campaign_id: str) -> Path:
        """Get seasonal perf figs dir"""
        return self.get_calibration_figures_dir(year, campaign_id) / 'seasonal'
    
    def get_calibration_convergence_dir(self, year: int, campaign_id: str) -> Path:
        """Get convergence diags figs dir"""
        return self.get_calibration_figures_dir(year, campaign_id) / 'convergence'

class SimulationConfig:
    """Sim config settings"""
    # CPU usage (0.8 = 80% of avail CPUs)
    CPU_USAGE = 0.8
    
    # Max workers for parallel proc
    MAX_WORKERS = None  # None = auto-config based on CPU_USAGE
    
    # Sim modes
    MODES = ['simulation', 'calibration']
    
    # Cleanup settings
    CLEANUP_SIMULATIONS = True
    
    # Default sim params
    DEFAULTS = {
        'timestep': 'hourly',
        'run_period': 'annual'
    }

class FutureScenarioConfig:
    """Future scenarios config"""
    
    # Available scenarios list
    SCENARIOS = [
        # 2035
        "2035_warm_PV_E", "2035_warm_PV_EE", 
        "2035_warm_UB_E", "2035_warm_UB_EE",
        "2035_typical_PV_E", "2035_typical_PV_EE", 
        "2035_typical_UB_E", "2035_typical_UB_EE",
        "2035_cold_PV_E", "2035_cold_PV_EE", 
        "2035_cold_UB_E", "2035_cold_UB_EE",
        # 2050
        "2050_warm_PV_E", "2050_warm_PV_EE", 
        "2050_warm_UB_E", "2050_warm_UB_EE",
        "2050_typical_PV_E", "2050_typical_PV_EE", 
        "2050_typical_UB_E", "2050_typical_UB_EE",
        "2050_cold_PV_E", "2050_cold_PV_EE", 
        "2050_cold_UB_E", "2050_cold_UB_EE"
    ]
    
    # Human readable scenario names
    SCENARIO_NAMES = {
        # 2035
        "2035_warm_PV_E": "2035 - Warm year - Predicted Value - Standard efficiency",
        "2035_warm_PV_EE": "2035 - Warm year - Predicted Value - Maximum efficiency",
        "2035_warm_UB_E": "2035 - Warm year - Upper Boundary - Standard efficiency",
        "2035_warm_UB_EE": "2035 - Warm year - Upper Boundary - Maximum efficiency",
        "2035_typical_PV_E": "2035 - Typical year - Predicted Value - Standard efficiency",
        "2035_typical_PV_EE": "2035 - Typical year - Predicted Value - Maximum efficiency",
        "2035_typical_UB_E": "2035 - Typical year - Upper Boundary - Standard efficiency",
        "2035_typical_UB_EE": "2035 - Typical year - Upper Boundary - Maximum efficiency",
        "2035_cold_PV_E": "2035 - Cold year - Predicted Value - Standard efficiency",
        "2035_cold_PV_EE": "2035 - Cold year - Predicted Value - Maximum efficiency",
        "2035_cold_UB_E": "2035 - Cold year - Upper Boundary - Standard efficiency",
        "2035_cold_UB_EE": "2035 - Cold year - Upper Boundary - Maximum efficiency",
        # 2050
        "2050_warm_PV_E": "2050 - Warm year - Predicted Value - Standard efficiency",
        "2050_warm_PV_EE": "2050 - Warm year - Predicted Value - Maximum efficiency",
        "2050_warm_UB_E": "2050 - Warm year - Upper Boundary - Standard efficiency",
        "2050_warm_UB_EE": "2050 - Warm year - Upper Boundary - Maximum efficiency",
        "2050_typical_PV_E": "2050 - Typical year - Predicted Value - Standard efficiency",
        "2050_typical_PV_EE": "2050 - Typical year - Predicted Value - Maximum efficiency",
        "2050_typical_UB_E": "2050 - Typical year - Upper Boundary - Standard efficiency",
        "2050_typical_UB_EE": "2050 - Typical year - Upper Boundary - Maximum efficiency",
        "2050_cold_PV_E": "2050 - Cold year - Predicted Value - Standard efficiency",
        "2050_cold_PV_EE": "2050 - Cold year - Predicted Value - Maximum efficiency",
        "2050_cold_UB_E": "2050 - Cold year - Upper Boundary - Standard efficiency",
        "2050_cold_UB_EE": "2050 - Cold year - Upper Boundary - Maximum efficiency"
    }
    
    @staticmethod
    def is_future_scenario(scenario_key: str) -> bool:
        """Check if key is future scenario"""
        return scenario_key in FutureScenarioConfig.SCENARIOS
    
    @staticmethod
    def get_scenario_year(scenario_key: str) -> int:
        """Extract year from scenario key"""
        if not FutureScenarioConfig.is_future_scenario(scenario_key):
            return 2023  # Ref year
        
        year_str = scenario_key.split('_')[0]
        return int(year_str)

class ProjectConfig:
    """Main config class"""
    def __init__(self):
        self.paths = {
            'input': InputPaths(),
            'openstudio': OpenStudioPaths(),
            'output': OutputPaths()
        }
        self.simulation = SimulationConfig()
        self.future_scenarios = FutureScenarioConfig()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure project logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(BASE_DIR / 'ubem.log')
            ]
        )
    
    def save_run_config(self, output_dir: Path, config_data: Dict[str, Any]):
        """Save run config"""
        config_file = output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

# Global instance
config = ProjectConfig()
