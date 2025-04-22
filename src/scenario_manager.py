"""
Module for Quebec Residential UBEM future scenario management.
Defines scenario parameters and transformations.
"""
import os
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from .config import config

class ScenarioManager:
    """
    Future scenario manager for Quebec Residential UBEM.
    Defines, transforms and applies future scenarios (2035, 2050).
    """
    
    def __init__(self):
        """Initialize ScenarioManager and define scenarios."""
        self.logger = logging.getLogger(__name__)
        self._initialize_scenarios()
        self._load_weather_mapping()
        
    def _initialize_scenarios(self):
        """
        Init 24 future scenarios with their parameters.
        Structure: {year}_{climate_type}_{electrification_growth}_{efficiency}
        """
        # Scenarios dict
        self.scenarios = {}
        
        # Define base parameters
        electrification_params = {
            # 2035
            "2035_PV": {  # Predicted Value 2035
                "electric_heating_pct": 0.95,
                "heat_pump_pct": 0.50,
                "gas_heating_pct": 0.05,
                "oil_heating_pct": 0.00
            },
            "2035_UB": {  # Upper Boundary 2035
                "electric_heating_pct": 1.00,
                "heat_pump_pct": 0.65,
                "gas_heating_pct": 0.00,
                "oil_heating_pct": 0.00
            },
            # 2050
            "2050_PV": {  # Predicted Value 2050
                "electric_heating_pct": 1.00,
                "heat_pump_pct": 0.80,
                "gas_heating_pct": 0.00,
                "oil_heating_pct": 0.00
            },
            "2050_UB": {  # Upper Boundary 2050
                "electric_heating_pct": 1.00,
                "heat_pump_pct": 0.95,
                "gas_heating_pct": 0.00,
                "oil_heating_pct": 0.00
            }
        }
        
        growth_params = {
            # 2035
            "2035_PV": {  # Predicted Value 2035
                "total_growth": 0.10,
                "apartment_growth": 0.10, 
                "detached_reduction": 0.05
            },
            "2035_UB": {  # Upper Boundary 2035
                "total_growth": 0.12,
                "apartment_growth": 0.05,
                "detached_reduction": -0.05  # Growth
            },
            # 2050
            "2050_PV": {  # Predicted Value 2050
                "total_growth": 0.20,
                "apartment_growth": 0.20,
                "detached_reduction": 0.15
            },
            "2050_UB": {  # Upper Boundary 2050
                "total_growth": 0.30,
                "apartment_growth": 0.10,
                "detached_reduction": -0.10  # Growth
            }
        }
        
        efficiency_params = {
            # 2035
            "2035_E": {  # Standard efficiency 2035
                "insulation_improvement": 0.10,
                "infiltration_reduction": 0.10,
                "cop_heat_pump": 3.5
            },
            "2035_EE": {  # Maximum efficiency 2035
                "insulation_improvement": 0.20,
                "infiltration_reduction": 0.20,
                "cop_heat_pump": 4.0
            },
            # 2050
            "2050_E": {  # Standard efficiency 2050
                "insulation_improvement": 0.20,
                "infiltration_reduction": 0.20,
                "cop_heat_pump": 4.0
            },
            "2050_EE": {  # Maximum efficiency 2050
                "insulation_improvement": 0.8,
                "infiltration_reduction": 0.50,
                "cop_heat_pump": 5.0
            }
        }
        
        climate_params = {
            # 2035
            "2035_warm": {
                "type": "warm",
                "co2_increase": 1.5
            },
            "2035_typical": {
                "type": "typical",
                "co2_increase": 1.0
            },
            "2035_cold": {
                "type": "cold", 
                "co2_increase": 0.5
            },
            # 2050
            "2050_warm": {
                "type": "warm",
                "co2_increase": 3.0
            },
            "2050_typical": {
                "type": "typical",
                "co2_increase": 2.0
            },
            "2050_cold": {
                "type": "cold",
                "co2_increase": 1.0
            }
        }
        
        # Generate all 24 scenarios (2 years × 3 climates × 2 growth paths × 2 efficiency levels)
        for year in ["2035", "2050"]:
            for climate in ["warm", "typical", "cold"]:
                for growth in ["PV", "UB"]:
                    for efficiency in ["E", "EE"]:
                        # Build scenario key
                        scenario_key = f"{year}_{climate}_{growth}_{efficiency}"
                        
                        # Combine parameters
                        electrification_key = f"{year}_{growth}"
                        growth_key = f"{year}_{growth}"
                        efficiency_key = f"{year}_{efficiency}"
                        climate_key = f"{year}_{climate}"
                        
                        # Create the scenario
                        self.scenarios[scenario_key] = {
                            "electrification": electrification_params[electrification_key],
                            "growth": growth_params[growth_key],
                            "efficiency": efficiency_params[efficiency_key],
                            "climate": climate_params[climate_key],
                            "year": int(year)
                        }
                        
        self.logger.info(f"Initialized {len(self.scenarios)} future scenarios")
    
    def _load_weather_mapping(self):
        """Load weather files mapping by zone."""
        try:
            mapping_file = config.paths['input'].WEATHER_FUTURE_MAPPING
            self.weather_mapping_df = pd.read_csv(mapping_file)
            self.logger.info(f"Future weather mapping loaded: {len(self.weather_mapping_df)} zones")
        except Exception as e:
            self.logger.error(f"Error loading weather mapping: {str(e)}")
            self.weather_mapping_df = pd.DataFrame()
    
    def get_scenario_params(self, scenario_key: str) -> Dict:
        """
        Get parameters for a scenario.
        
        Args:
            scenario_key: Scenario key (e.g. '2035_warm_PV_E')
            
        Returns:
            Dict of scenario parameters or empty dict if not found
        """
        return self.scenarios.get(scenario_key, {})
    
    def transform_archetype(self, archetype: pd.Series, scenario_key: str) -> pd.Series:
        """
        Transform an archetype according to future scenario parameters.
        
        Args:
            archetype: Pandas Series with archetype data
            scenario_key: Scenario key (e.g. '2035_warm_PV_E')
            
        Returns:
            Transformed archetype
        """
        # Get scenario parameters
        scenario = self.get_scenario_params(scenario_key)
        if not scenario:
            self.logger.warning(f"Scenario {scenario_key} not found, using original archetype")
            return archetype
            
        # Create copy to avoid modifying original
        transformed = archetype.copy()
        
        # 1. Transform envelope (efficiency)
        transformed = self._transform_envelope(transformed, scenario)
        
        # 2. Other scenario-specific transformations
        transformed = self._transform_other_params(transformed, scenario)
        
        # 3. Transform heating systems (electrification)
        transformed = self._transform_heating_system(transformed, scenario)
        
        return transformed
    
    def _transform_envelope(self, archetype: pd.Series, scenario: Dict) -> pd.Series:
        """
        Transform archetype envelope based on efficiency parameters.
        
        Args:
            archetype: Pandas Series with archetype data
            scenario: Scenario parameters
            
        Returns:
            Modified pandas Series
        """
        # Create copy for modifications
        transformed = archetype.copy()
        
        # Get efficiency parameters
        efficiency = scenario.get('efficiency', {})
        if not efficiency:
            return transformed
            
        # Insulation improvement
        insulation_params = [
            'dominantWallRVal', 
            'dominantCeilingRVal', 
            'dominantBasementWallRVal',
            'dominantFoundationFloorRVal',
            'dominantSlabRVal'
        ]
        
        for param in insulation_params:
            if param in transformed and pd.notna(transformed[param]):
                # Calculate new value with improvement
                current_value = float(transformed[param])
                improved_value = current_value * (1 + efficiency['insulation_improvement'])
                transformed[param] = improved_value
        
        # Window improvement
        if 'dominantWindowRVal' in transformed and pd.notna(transformed['dominantWindowRVal']):
            # For windows, improvement is less (50% of insulation improvement)
            current_value = float(transformed['dominantWindowRVal'])
            improved_value = current_value * (1 + efficiency['insulation_improvement'] * 0.5)
            transformed['dominantWindowRVal'] = improved_value
            
        # Air infiltration reduction
        if 'ach' in transformed and pd.notna(transformed['ach']):
            # Reduce infiltration according to scenario
            current_value = float(transformed['ach'])
            reduced_value = current_value * (1 - efficiency['infiltration_reduction'])
            transformed['ach'] = reduced_value
            
        return transformed
    
    def _transform_other_params(self, archetype: pd.Series, scenario: Dict) -> pd.Series:
        """
        Apply other transformations to archetype based on scenario.
        
        Args:
            archetype: Pandas Series with archetype data
            scenario: Scenario parameters
            
        Returns:
            Modified pandas Series
        """
        # Create copy for modifications
        transformed = archetype.copy()
        
        # 1. DHW efficiency improvement
        if 'primaryDhwEff' in transformed and pd.notna(transformed['primaryDhwEff']):
            efficiency = scenario.get('efficiency', {})
            # Less improvement than envelope (30% of insulation improvement)
            improvement_factor = efficiency.get('insulation_improvement', 0) * 0.3
            current_value = float(transformed['primaryDhwEff'])
            improved_value = min(99.0, current_value * (1 + improvement_factor))  # Cap at 99%
            transformed['primaryDhwEff'] = improved_value
            
        # 2. Temperature setpoints modification
        # Increase cooling setpoint and decrease heating setpoint slightly
        # to reflect more energy-efficient behavior
        if 'coolingSetPoint' in transformed and pd.notna(transformed['coolingSetPoint']):
            transformed['coolingSetPoint'] = float(transformed['coolingSetPoint']) + 0.5  # +0.5°C
            
        if 'dayHeatingSetPoint' in transformed and pd.notna(transformed['dayHeatingSetPoint']):
            transformed['dayHeatingSetPoint'] = float(transformed['dayHeatingSetPoint']) - 0.5  # -0.5°C
            
        return transformed
    
    def _transform_heating_system(self, archetype: pd.Series, scenario: Dict) -> pd.Series:
        """
        Improve heating system efficiency without changing system type.
        
        Args:
            archetype: Pandas Series with archetype data
            scenario: Scenario parameters
            
        Returns:
            Modified pandas Series
        """
        # Create copy for modifications
        transformed = archetype.copy()
        
        # Get efficiency parameters
        efficiency = scenario.get('efficiency', {})
        
        # 1. If heat pump exists, improve its COP
        heat_pump_type = transformed.get('heatPumpType')
        if pd.notna(heat_pump_type):
            # Make sure efficiency type is explicitly defined
            # To avoid conversion errors in ArchetypeManager.prepare_archetype
            if 'heatPumpHeatingEffType' not in transformed or pd.isna(transformed.get('heatPumpHeatingEffType')):
                transformed['heatPumpHeatingEffType'] = 'COP'
            
            if 'heatPumpCoolingEffType' not in transformed or pd.isna(transformed.get('heatPumpCoolingEffType')):
                transformed['heatPumpCoolingEffType'] = True
            
            # Improve efficiency with relative variation
            if pd.notna(transformed.get('heatPumpHeatingEff')):
                # Get target COP from scenario and current value
                target_cop = efficiency.get('cop_heat_pump', 5.0)
                current_cop = float(transformed['heatPumpHeatingEff'])
                
                # Construction year factor (for different improvement rates)
                year_factor = 1.0
                if pd.notna(transformed.get('vintageExact')):
                    year = float(transformed['vintageExact'])
                    # Newer systems need less improvement
                    year_norm = min(1.0, max(0.0, (year - 1950) / 70.0))  # 0-1 for years 1950-2020
                    year_factor = 0.8 + (year_norm * 0.4)  # 0.8-1.2
                
                # Random factor for diversity (±10%)
                random_factor = 0.9 + (np.random.random() * 0.2)
                
                # Calculate relative improvement
                if current_cop < target_cop:
                    # If current efficiency is lower than target:
                    # - Larger gap means more improvement
                    # - Year and random factors affect improvement amount
                    improvement_ratio = (target_cop / current_cop) - 1.0  # Ratio of needed improvement
                    applied_improvement = improvement_ratio * year_factor * random_factor * 0.7  # 70% of possible improvement
                    
                    # Calculate new value (progressive improvement)
                    new_cop = current_cop * (1.0 + applied_improvement)
                    
                    # Ensure improvement is significant but not excessive
                    new_cop = min(target_cop * 1.1, max(current_cop * 1.05, new_cop))
                    
                    transformed['heatPumpHeatingEff'] = new_cop
                    self.logger.debug(f"Heat pump improvement: {current_cop:.2f} → {new_cop:.2f} (target: {target_cop:.2f})")
                else:
                    # If already above target, small possible improvement
                    improvement = min(0.1, (target_cop * 0.1) / current_cop)  # Max 10% improvement
                    new_cop = current_cop * (1.0 + improvement * random_factor * 0.5)
                    transformed['heatPumpHeatingEff'] = new_cop
                    self.logger.debug(f"Small improvement for already efficient heat pump: {current_cop:.2f} → {new_cop:.2f}")
                
                # Also improve cooling efficiency with some variation
                if pd.notna(transformed.get('heatPumpCoolingEff')):
                    current_cooling = float(transformed['heatPumpCoolingEff'])
                    cooling_factor = 0.85 + (np.random.random() * 0.1)  # 0.85-0.95
                    transformed['heatPumpCoolingEff'] = new_cop * cooling_factor
            
            self.logger.debug(f"Heat pump efficiency improvement: COP = {transformed.get('heatPumpHeatingEff')}")
            return transformed
        
        # 2. Electric system, small improvement
        if transformed.get('spaceHeatingFuel') == 'Electric':
            if pd.notna(transformed.get('spaceHeatingEff')):
                # Electric systems already efficient, limited improvement
                current_eff = float(transformed['spaceHeatingEff'])
                transformed['spaceHeatingEff'] = min(100.0, current_eff * 1.05)
                self.logger.debug(f"Electric heating efficiency improvement: {current_eff} → {transformed['spaceHeatingEff']}")
        
        # 3. Gas/oil systems, more significant improvement (more efficient tech)
        elif transformed.get('spaceHeatingFuel') in ['Natural gas', 'Oil']:
            if pd.notna(transformed.get('spaceHeatingEff')):
                current_eff = float(transformed['spaceHeatingEff'])
                transformed['spaceHeatingEff'] = min(98.0, current_eff * 1.15)
                self.logger.debug(f"{transformed.get('spaceHeatingFuel')} heating efficiency improvement: {current_eff} → {transformed['spaceHeatingEff']}")
        
        # 4. Wood systems, moderate improvement
        elif transformed.get('spaceHeatingFuel') in ['Mixed Wood', 'Hardwood', 'Wood Pellets']:
            if pd.notna(transformed.get('spaceHeatingEff')):
                current_eff = float(transformed['spaceHeatingEff'])
                transformed['spaceHeatingEff'] = min(85.0, current_eff * 1.12)
                self.logger.debug(f"Wood heating efficiency improvement: {current_eff} → {transformed['spaceHeatingEff']}")
                
        # 5. Dual systems - improve both components
        if pd.notna(transformed.get('supplHeatingFuel')):
            if pd.notna(transformed.get('supplHeatingEff')):
                current_eff = float(transformed['supplHeatingEff'])
                transformed['supplHeatingEff'] = min(98.0, current_eff * 1.1)
                self.logger.debug(f"Supplementary heating efficiency improvement: {current_eff} → {transformed['supplHeatingEff']}")
                
        return transformed
    
    def get_weather_file(self, weather_zone: int, scenario_key: str) -> Optional[str]:
        """
        Get weather file for scenario and zone.
        
        Args:
            weather_zone: Weather zone ID
            scenario_key: Scenario key (e.g. '2035_warm_PV_E')
            
        Returns:
            Relative path to EPW file or None if not found
        """
        if self.weather_mapping_df.empty:
            self.logger.error("Weather mapping not available")
            return None
            
        # Extract scenario info
        scenario = self.get_scenario_params(scenario_key)
        if not scenario:
            self.logger.warning(f"Scenario {scenario_key} not found for weather selection")
            return None
            
        # Extract year, climate type and CO2 increase
        year = str(scenario.get('year', ''))
        climate_type = scenario['climate']['type']
        co2_increase = scenario['climate']['co2_increase']  # Keep as float
        
        # Exact format with decimal point for columns
        co2_str = str(co2_increase)  # Keep decimal point: 1.5 stays "1.5"
        
        # Build exact column name we're looking for
        target_column = f"EPW_{year}_{climate_type}_{co2_str}"
        
        self.logger.info(f"Looking for specific column: {target_column} for scenario {scenario_key}")
        
        try:
            # Find EPW file for zone
            row = self.weather_mapping_df[self.weather_mapping_df['weather_zo'] == weather_zone]
            if row.empty:
                self.logger.warning(f"Weather zone {weather_zone} not found in mapping")
                return None
            
            # Debug available columns
            self.logger.debug(f"Available columns in mapping: {list(row.columns)}")
            
            # Check if target column exists
            if target_column in row.columns:
                self.logger.info(f"Target column found: {target_column}")
                epw_file = row[target_column].iloc[0]
            else:
                # If exact column doesn't exist, try alternatives for same climate
                climate_columns = [col for col in row.columns if f"{year}_{climate_type}" in col]
                
                if climate_columns:
                    # Use first column found for this climate
                    column_to_use = climate_columns[0]
                    self.logger.info(f"Using alternative column for same climate: {column_to_use}")
                    epw_file = row[column_to_use].iloc[0]
                else:
                    # Last resort, look for any column for this year
                    year_columns = [col for col in row.columns if year in col and col.startswith('EPW')]
                    if year_columns:
                        column_to_use = year_columns[0]
                        self.logger.warning(f"No column for {climate_type} found, using {column_to_use}")
                        epw_file = row[column_to_use].iloc[0]
                    else:
                        self.logger.error(f"No weather column found for year {year}")
                        return None
            
            if pd.isna(epw_file):
                self.logger.warning(f"EPW file not defined for zone {weather_zone}")
                return None
            
            # Build relative path
            epw_path = f"future/{climate_type}/{epw_file}"
            self.logger.info(f"Using weather file: {epw_path} for scenario {scenario_key}")
            return epw_path
            
        except Exception as e:
            self.logger.error(f"Error selecting weather file: {str(e)}")
            return None
    
    def _determine_building_type(self, row: pd.Series) -> str:
        """
        Determine building type from property data.
        Same as function in AggregationManager for consistency.
        
        Args:
            row: Pandas Series with building data
            
        Returns:
            Building type (Maisons unifamiliales, Maisons individuelles attenantes, Appartements)
        """
        # Default values to avoid NaN
        lien_physique = row.get('lien_physique_code', 0) 
        nb_logements = row.get('nombre_logements', 0)
        
        # RULE 1: Based on physical link
        if lien_physique == 1:  # Detached
            if nb_logements <= 2:
                return "Maisons unifamiliales"
            else:
                return "Appartements"
                
        elif lien_physique in [2, 3, 4]:  # Semi-detached or row
            if nb_logements <= 2:
                return "Maisons individuelles attenantes"
            else:
                return "Appartements"
                
        elif lien_physique == 5:  # Integrated
            return "Appartements"
            
        # RULE 2: If physical link missing, use unit count
        elif nb_logements == 1:
            return "Maisons unifamiliales"
        elif nb_logements == 2:
            return "Maisons individuelles attenantes"
        elif nb_logements > 2:
            return "Appartements"
        
        # RULE 3: Undetermined cases
        else:
            genre = row.get('genre_construction_code', 0)
            if genre in [1, 3]:  # Single-story or unimodular
                return "Maisons unifamiliales"
            elif genre in [4, 5]:  # Mansard or full stories
                return "Appartements"
            else:
                return "Maisons unifamiliales"  # Default value
    
    def transform_property_data(self, data: pd.DataFrame, scenario_key: str) -> pd.DataFrame:
        """
        Transform property data according to growth scenario.
        Based on total area rather than building count.
        With improved temporal distribution and nuanced removal criteria.
        
        Args:
            data: DataFrame of property evaluation data
            scenario_key: Future scenario key
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Get scenario parameters
            scenario = self.get_scenario_params(scenario_key)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_key} not found for property data transformation")
                return data
                
            # Create copy and reset index to avoid issues
            transformed = data.copy().reset_index(drop=True)
            
            # Add building classification
            # Using apply doesn't cause warning because it's a vectorized operation
            transformed['building_type'] = transformed.apply(self._determine_building_type, axis=1)
            
            # Get growth parameters
            growth = scenario.get('growth', {})
            total_growth = growth.get('total_growth', 0)
            apartment_growth = growth.get('apartment_growth', 0)
            detached_reduction = growth.get('detached_reduction', 0)
            
            # Explicitly log growth parameters for verification
            self.logger.info(f"Growth parameters for {scenario_key}:")
            self.logger.info(f"  → Total growth: {total_growth:.2f}")
            self.logger.info(f"  → Apartment growth: {apartment_growth:.2f}")
            self.logger.info(f"  → Detached reduction/growth: {detached_reduction:.2f}")
            
            # Adjust to differentiate PV and UB scenarios
            if "UB" in scenario_key:  # Upper Boundary
                # Emphasize single-family home growth for UB
                if detached_reduction < 0:  # If already growth (negative value)
                    detached_reduction *= 1.2  # Further increase growth
                else:  # If reduction
                    detached_reduction *= 0.7  # Reduce reduction amount
                
                self.logger.info(f"Upper Boundary (UB) scenario: Adjustment to single-family transformation ({detached_reduction:.2f})")
            
            # For each building type, apply modifications
            # 1. Single-family homes
            # Use copy to avoid warnings
            detached_mask = transformed['building_type'] == 'Maisons unifamiliales'
            detached = transformed[detached_mask].copy()
            
            # Calculate current total area
            detached_total_area = detached['aire_etages'].sum()
            
            # Reduction or growth of detached houses based on area
            if detached_reduction > 0:  # Reduction
                # Calculate area to remove
                area_to_remove = detached_total_area * detached_reduction
                
                self.logger.info(f"SINGLE-FAMILY REDUCTION: Area to remove: {area_to_remove:.1f} m² out of {detached_total_area:.1f} m²")
                
                # Create priority scores for removal
                # Use .loc for all modifications
                detached.loc[:, 'age_score'] = (2023 - detached['annee_construction']) / 100  # Normalize age
                
                # Simple approach: use area as energy intensity proxy
                detached.loc[:, 'size_score'] = detached['aire_etages'] / detached['aire_etages'].max()
                
                # Combined score (70% age, 30% size)
                detached.loc[:, 'removal_score'] = 0.7 * detached['age_score'] + 0.3 * detached['size_score']
                
                # Sort by removal score (descending)
                detached_sorted = detached.sort_values('removal_score', ascending=False).copy()
                
                # Calculate cumulative area
                detached_sorted.loc[:, 'cumulative_area'] = detached_sorted['aire_etages'].cumsum()
                
                # Select buildings until area to remove is reached
                to_remove_mask = detached_sorted['cumulative_area'] <= area_to_remove
                
                # If not enough buildings, add next one crossing threshold
                if detached_sorted[to_remove_mask]['aire_etages'].sum() < area_to_remove and len(detached_sorted) > sum(to_remove_mask):
                    # Add next row to selection
                    next_idx = sum(to_remove_mask)
                    to_remove_mask.iloc[next_idx] = True
                
                # Get indices of rows to remove from original DataFrame
                to_remove_indices = detached_sorted[to_remove_mask].index
                
                # Create boolean mask to keep all buildings EXCEPT those to remove
                keep_mask = ~transformed.index.isin(to_remove_indices)
                transformed = transformed[keep_mask].copy().reset_index(drop=True)
                
                self.logger.info(
                    f"Scenario {scenario_key}: Removed {len(to_remove_indices)} single-family houses "
                    f"({detached_sorted[to_remove_mask]['aire_etages'].sum():.1f} m²)"
                )
            else:  # Growth
                # Area to add
                growth_factor = abs(detached_reduction)
                area_to_add = detached_total_area * growth_factor
                
                self.logger.info(f"SINGLE-FAMILY GROWTH: Area to add: {area_to_add:.1f} m² (factor {growth_factor:.2f})")
                
                if area_to_add > 0 and not detached.empty:
                    # Use all recent buildings as templates
                    recent_buildings = detached.sort_values('annee_construction', ascending=False).copy()
                    
                    # Calculate average area of recent buildings
                    recent_avg_area = recent_buildings['aire_etages'].mean()
                    
                    # Approximate number of buildings to add
                    n_to_add = int(area_to_add / recent_avg_area)
                    
                    # Randomly select buildings to duplicate, with replacement if needed
                    to_duplicate = recent_buildings.sample(n=min(n_to_add, len(recent_buildings)), replace=True).copy()
                    
                    # Calculate year range
                    reference_year = 2023
                    target_year = scenario['year']  # 2035 or 2050
                    year_span = target_year - reference_year
                    
                    # Duplicate and modify slightly
                    new_buildings = []
                    total_area_added = 0
                    
                    for i, (_, building) in enumerate(to_duplicate.iterrows()):
                        # Stop if target area reached
                        if total_area_added >= area_to_add:
                            break
                            
                        # Create dict for new building (avoids reference problems)
                        new_building = building.to_dict()
                        
                        # Progressive construction years distribution
                        construction_year = reference_year + int((i / len(to_duplicate)) * year_span)
                        new_building['annee_construction'] = construction_year
                        
                        # Slight area variation (±10%)
                        new_building['aire_etages'] = building['aire_etages'] * (0.9 + np.random.random() * 0.2)
                        
                        total_area_added += new_building['aire_etages']
                        new_buildings.append(new_building)
                    
                    # Add new buildings
                    if new_buildings:
                        new_df = pd.DataFrame(new_buildings)
                        transformed = pd.concat([transformed, new_df], ignore_index=True)
                        self.logger.info(
                            f"Scenario {scenario_key}: Added {len(new_buildings)} new single-family houses "
                            f"({total_area_added:.1f} m²)"
                        )
            
            # 2. Apartments
            # Recalculate mask as transformed may have changed
            apartments_mask = transformed['building_type'] == 'Appartements'
            apartments = transformed[apartments_mask].copy()
            apartments_total_area = apartments['aire_etages'].sum()
            
            # Adjust to differentiate PV and UB scenarios
            if "UB" in scenario_key:  # For UB, reduce emphasis on apartments
                apartment_growth *= 0.7
                self.logger.info(f"Upper Boundary (UB) scenario: Adjustment to apartment growth ({apartment_growth:.2f})")
            elif "PV" in scenario_key:  # For PV, increase emphasis on apartments
                apartment_growth *= 1.1
                self.logger.info(f"Predicted Value (PV) scenario: Adjustment to apartment growth ({apartment_growth:.2f})")
            
            # Apartment growth based on area
            if apartment_growth > 0 and apartments_total_area > 0:
                # Area to add
                area_to_add = apartments_total_area * apartment_growth
                
                self.logger.info(f"APARTMENT GROWTH: Area to add: {area_to_add:.1f} m² (factor {apartment_growth:.2f})")
                
                if area_to_add > 0:
                    # Use recent buildings as templates
                    recent_apartments = apartments.sort_values('annee_construction', ascending=False).copy()
                    
                    # Calculate average area of recent apartments
                    recent_avg_area = recent_apartments['aire_etages'].mean()
                    
                    # Approximate number of apartments to add
                    n_to_add = int(area_to_add / recent_avg_area)
                    
                    # Randomly select buildings to duplicate, with replacement if needed
                    to_duplicate = recent_apartments.sample(n=min(n_to_add, len(recent_apartments)), replace=True).copy()
                    
                    # Calculate year range
                    reference_year = 2023
                    target_year = scenario['year']  # 2035 or 2050
                    year_span = target_year - reference_year
                    
                    # Duplicate and modify slightly
                    new_apartments = []
                    total_area_added = 0
                    
                    for i, (_, building) in enumerate(to_duplicate.iterrows()):
                        # Stop if target area reached
                        if total_area_added >= area_to_add:
                            break
                            
                        # Create dict for new building (avoids reference problems)
                        new_building = building.to_dict()
                        
                        # Progressive construction years distribution
                        construction_year = reference_year + int((i / len(to_duplicate)) * year_span)
                        new_building['annee_construction'] = construction_year
                        
                        # Slight area variation (±10%)
                        new_building['aire_etages'] = building['aire_etages'] * (0.9 + np.random.random() * 0.2)
                        
                        # Slightly increase unit count (higher density)
                        if 'nombre_logements' in new_building and pd.notna(building.get('nombre_logements')):
                            try:
                                nb_logements = building.get('nombre_logements')
                                if pd.notna(nb_logements):
                                    new_building['nombre_logements'] = max(3, int(nb_logements * (1 + np.random.random() * 0.3)))
                                else:
                                    new_building['nombre_logements'] = 3
                            except Exception as e:
                                new_building['nombre_logements'] = 3
                        elif 'nombre_logements' not in new_building:
                            new_building['nombre_logements'] = 3
                        
                        total_area_added += new_building['aire_etages']
                        new_apartments.append(new_building)
                    
                    # Add new buildings
                    if new_apartments:
                        new_df = pd.DataFrame(new_apartments)
                        transformed = pd.concat([transformed, new_df], ignore_index=True)
                        self.logger.info(
                            f"Scenario {scenario_key}: Added {len(new_apartments)} new apartments "
                            f"({total_area_added:.1f} m²)"
                        )
            
            # Calculate current total area after specific transformations
            current_total_area = transformed['aire_etages'].sum()
            
            # Calculate target total area after growth
            target_total_area = data['aire_etages'].sum() * (1 + total_growth)
            
            # Missing area after specific modifications
            missing_area = target_total_area - current_total_area
            
            # More coherent distribution of missing growth
            if missing_area > 0:
                self.logger.info(f"Missing area for total growth: {missing_area:.1f} m²")
                
                # Determine which building types can receive additional growth
                allowed_growth_types = []
                
                # If detached_reduction is positive (PV), don't add single-family houses
                if detached_reduction <= 0:  # Only when negative (growth) or zero
                    allowed_growth_types.append('Maisons unifamiliales')
                
                # Always allow attached houses
                allowed_growth_types.append('Maisons individuelles attenantes')
                
                # Always allow apartments
                allowed_growth_types.append('Appartements')
                
                # Strongly prefer apartments for PV, houses for UB
                type_weights = {}
                if "PV" in scenario_key:
                    # PV scenario: strong densification
                    type_weights = {
                        'Maisons unifamiliales': 0,  # No additional growth if positive reduction
                        'Maisons individuelles attenantes': 0.3,
                        'Appartements': 0.7
                    }
                else:  # UB
                    # UB scenario: more balanced development
                    type_weights = {
                        'Maisons unifamiliales': 0.5 if detached_reduction <= 0 else 0,
                        'Maisons individuelles attenantes': 0.3,
                        'Appartements': 0.2
                    }
                
                # Recalculate sum of weights for allowed types
                weight_sum = sum(type_weights[typ] for typ in allowed_growth_types)
                if weight_sum == 0:
                    # For safety, distribute uniformly if no weights
                    for typ in allowed_growth_types:
                        type_weights[typ] = 1 / len(allowed_growth_types)
                else:
                    # Normalize weights for allowed types
                    for typ in type_weights:
                        if typ in allowed_growth_types:
                            type_weights[typ] /= weight_sum
                
                # Log final weights
                self.logger.info(f"Missing growth distribution: {type_weights}")
                
                # Distribute growth among allowed types and prepare representative buildings
                buildings_by_type = {}
                for building_type in allowed_growth_types:
                    # Filter recent buildings of this type
                    type_recent = transformed[transformed['building_type'] == building_type].sort_values('annee_construction', ascending=False)
                    if not type_recent.empty:
                        buildings_by_type[building_type] = type_recent
                
                # Create new buildings respecting weights
                remaining_area = missing_area
                new_buildings = []
                
                for building_type, weight in type_weights.items():
                    if building_type not in allowed_growth_types or weight <= 0:
                        continue
                        
                    # Area to add for this type
                    area_to_add = missing_area * weight
                    if area_to_add <= 0:
                        continue
                    
                    # Check if we have buildings of this type to duplicate
                    if building_type not in buildings_by_type or buildings_by_type[building_type].empty:
                        self.logger.warning(f"No buildings of type {building_type} found for duplication")
                        continue
                    
                    # Calculate average size for this building type
                    avg_size = buildings_by_type[building_type]['aire_etages'].mean()
                    
                    # Number of buildings to add
                    n_to_add = int(area_to_add / avg_size) + 1  # +1 to compensate rounding
                    
                    # Select buildings to duplicate
                    to_duplicate = buildings_by_type[building_type].sample(
                        n=min(n_to_add, len(buildings_by_type[building_type])), 
                        replace=True
                    )
                    
                    # Calculate year range
                    reference_year = 2023
                    target_year = int(scenario_key.split('_')[0])
                    year_span = target_year - reference_year
                    
                    # Add new buildings
                    type_area_added = 0
                    for i, (_, base_building) in enumerate(to_duplicate.iterrows()):
                        if type_area_added >= area_to_add:
                            break
                            
                        # Create new building
                        new_building = base_building.copy()
                        
                        # Progressive construction year
                        construction_year = reference_year + int((i / len(to_duplicate)) * year_span)
                        new_building['annee_construction'] = construction_year
                        
                        # Slight size variation
                        size_factor = 0.9 + (np.random.random() * 0.2)  # 0.9 to 1.1
                        new_building['aire_etages'] = base_building['aire_etages'] * size_factor
                        
                        type_area_added += new_building['aire_etages']
                        new_buildings.append(new_building)
                    
                    self.logger.info(f"Added {len(new_buildings)} buildings of type {building_type} ({type_area_added:.1f} m²)")
                    remaining_area -= type_area_added
                
                # Add new buildings to DataFrame
                if new_buildings:
                    new_df = pd.DataFrame(new_buildings)
                    transformed = pd.concat([transformed, new_df], ignore_index=True)
            
            # Final statistics
            final_counts = transformed.groupby('building_type').agg({'aire_etages': ['count', 'sum']})
            self.logger.info(
                f"Scenario {scenario_key}: Final building distribution:\n"
                f"{final_counts.to_string()}"
            )
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming property data: {str(e)}")
            # Show error trace for debugging
            import traceback
            self.logger.debug(f"Error trace: {traceback.format_exc()}")
            return data

    def apply_scenario_to_calibration_params(self, 
                                          calibration_params: Dict[str, Dict[int, float]], 
                                          scenario_key: str) -> Dict[str, Dict[int, float]]:
        """
        Adapt calibrated parameters for future scenario.
        
        Args:
            calibration_params: Calibrated parameters {param_name: {archetype_id: value}}
            scenario_key: Scenario key
            
        Returns:
            Calibrated parameters adapted to scenario
        """
        # Check that calibration_params is valid dict
        if not calibration_params or not isinstance(calibration_params, dict):
            self.logger.warning(f"Invalid or empty calibration parameters for scenario {scenario_key}")
            return {}
            
        # Get scenario parameters
        scenario = self.get_scenario_params(scenario_key)
        if not scenario:
            self.logger.warning(f"Scenario {scenario_key} not found, returning original parameters")
            return calibration_params
            
        # Copy to avoid modifying original
        modified_params = {}
        for param_name, archetype_values in calibration_params.items():
            # If value is not a dict but a number, likely data error
            # Ignore it instead of warning
            if not isinstance(archetype_values, dict):
                if isinstance(archetype_values, (int, float)):
                    self.logger.debug(f"Ignoring scalar parameter {param_name} with value {archetype_values}")
                else:
                    self.logger.debug(f"Ignoring non-dictionary parameter {param_name}: {type(archetype_values)}")
                continue
                
            modified_params[param_name] = {}
            
            # For each archetype
            for archetype_id, value in archetype_values.items():
                # Check that value is a number
                if not isinstance(value, (int, float)):
                    self.logger.debug(f"Ignoring non-numeric value for {param_name}[{archetype_id}]: {value}")
                    continue
                    
                # Apply modifications based on parameter and scenario
                try:
                    if param_name in ['wall_rvalue', 'ceiling_rvalue', 'door_rvalue', 'foundation_r']:
                        # Insulation improvement - reduce relative adjustment
                        # Since buildings are already improved in scenario
                        efficiency_factor = scenario['efficiency']['insulation_improvement']
                        # Reduce relative adjustment (as already improved by scenario)
                        adjusted_value = value / (1 + efficiency_factor * 0.5)
                        modified_params[param_name][archetype_id] = adjusted_value
                    
                    elif param_name == 'infiltration_rate':
                        # Infiltration reduction - reduce relative adjustment
                        reduction_factor = scenario['efficiency']['infiltration_reduction']
                        # Reduce relative adjustment (as already improved by scenario)
                        adjusted_value = value / (1 - reduction_factor * 0.5)
                        modified_params[param_name][archetype_id] = adjusted_value
                    
                    # NOTE: We don't specifically handle heating_efficiency here
                    # because heat pumps are already transformed directly in the scenario
                    # and we don't calibrate this parameter for heat pumps.
                    # Incorrect handling would cause abnormally low efficiency values.
                    
                    else:
                        # For other parameters, keep calibrated value
                        modified_params[param_name][archetype_id] = value
                except Exception as e:
                    self.logger.debug(f"Error adjusting {param_name}[{archetype_id}]: {str(e)}")
                    # Keep original value in case of error
                    modified_params[param_name][archetype_id] = value
        
        return modified_params
        
    def transform_selected_archetypes(self, archetypes_df: pd.DataFrame, scenario_key: str, 
                                    transform_percentage: float = 0.3) -> pd.DataFrame:
        """
        Transform a subset of archetypes according to scenario parameters.
        
        Args:
            archetypes_df: DataFrame containing all archetypes
            scenario_key: Future scenario key
            transform_percentage: Percentage of archetypes to transform (0.0-1.0)
            
        Returns:
            DataFrame with transformed archetypes
        """
        # Get scenario parameters
        scenario = self.get_scenario_params(scenario_key)
        if not scenario:
            self.logger.warning(f"Scenario {scenario_key} not found for archetype transformation")
            return archetypes_df
            
        # Make deep copy to avoid modifying original
        transformed_df = archetypes_df.copy()
        
        # Apply envelope and other parameter transformations to ALL archetypes
        # Use .loc to avoid SettingWithCopyWarning
        self.logger.info(f"Applying envelope and efficiency improvements to all archetypes")
        
        for idx in transformed_df.index:
            # Create temporary copy for transformations
            archetype_temp = transformed_df.loc[idx].copy()
            
            # Apply transformations to temporary copy
            archetype_temp = self._transform_envelope(archetype_temp, scenario)
            archetype_temp = self._transform_other_params(archetype_temp, scenario)
            archetype_temp = self._transform_heating_system(archetype_temp, scenario)
            
            # Copy modified values to original DataFrame
            for col in archetype_temp.index:
                if col in transformed_df.columns:
                    transformed_df.loc[idx, col] = archetype_temp[col]
            
            # Log first few archetypes for verification
            if idx < 3:  # Limit to 3 to avoid log flooding
                self.logger.debug(f"Archetype {idx} - After envelope transformation:")
                self.logger.debug(f"  → dominantWallRVal: {transformed_df.loc[idx, 'dominantWallRVal']}")
                self.logger.debug(f"  → dominantCeilingRVal: {transformed_df.loc[idx, 'dominantCeilingRVal']}")
                self.logger.debug(f"  → ach: {transformed_df.loc[idx, 'ach']}")
        
        # THEN selectively transform fossil heating systems
        # Extract electrification parameters
        electrification = scenario.get('electrification', {})
        
        # Fossil systems to replace
        fossil_systems = ['Natural gas', 'Oil']
        
        # Identify archetypes with fossil systems
        fossil_archetypes = []
        for idx in transformed_df.index:
            if transformed_df.loc[idx, 'spaceHeatingFuel'] in fossil_systems:
                # Calculate priority score for transformation
                # Newer and more efficient buildings get priority
                year = float(transformed_df.loc[idx, 'vintageExact'])
                # Normalize year (0-1)
                year_norm = (year - 1900) / (2020 - 1900)
                
                # Insulation (use dominantWallRVal as efficiency proxy)
                r_value = float(transformed_df.loc[idx, 'dominantWallRVal'])
                r_norm = min(1.0, r_value / 5.0)  # Normalize (0-1)
                
                # Combined score (higher = more priority)
                priority_score = 0.7 * year_norm + 0.3 * r_norm
                
                fossil_archetypes.append({
                    'index': idx,
                    'score': priority_score,
                    'system': transformed_df.loc[idx, 'spaceHeatingFuel']
                })
        
        # Sort by priority score (descending)
        fossil_archetypes.sort(key=lambda x: x['score'], reverse=True)
        
        # Number of archetypes to transform
        # Adjust percentage based on scenario type (PV vs UB)
        if "UB" in scenario_key:  # Upper Boundary = more aggressive electrification
            transform_percentage = transform_percentage * 1.5  # 45% instead of 30%
            self.logger.info(f"Upper Boundary (UB) scenario: more aggressive electrification ({transform_percentage*100:.1f}%)")
        
        n_to_transform = int(len(fossil_archetypes) * transform_percentage)
        self.logger.info(f"Transforming {n_to_transform}/{len(fossil_archetypes)} archetypes with fossil systems")
        
        # Select priority archetypes to transform
        to_transform = fossil_archetypes[:n_to_transform]
        
        # Stats for tracking
        stats = {
            'total_fossil': len(fossil_archetypes),
            'transformed': n_to_transform,
            'to_heat_pump': 0,
            'to_electric': 0,
            'unchanged': len(fossil_archetypes) - n_to_transform
        }
        
        # Transform selected archetypes
        for item in to_transform:
            idx = item['index']
            
            # Determine whether to transform to heat pump or standard electric
            # Adjust heat pump probability based on scenario type
            heat_pump_probability = electrification.get('heat_pump_pct', 0.7)
            
            if np.random.random() < heat_pump_probability:
                # Transform to heat pump - use .loc for all modifications
                # Use .loc for each individual modification
                transformed_df.loc[idx, 'heatPumpType'] = 'AirHeatPump'

                # Explicitly define efficiency type to avoid conversion errors
                # For air-air heat pumps, expected type by HPXML is HSPF
                transformed_df.loc[idx, 'heatPumpHeatingEffType'] = 'COP'  # Storage value (COP)
                transformed_df.loc[idx, 'heatPumpCoolingEffType'] = True   # Boolean value for cooling efficiency

                # Add diversity to heat pump efficiencies
                # 1. Variation factor based on building characteristics
                building_quality = 0.0
                
                # Use insulation as building quality indicator
                if pd.notna(transformed_df.loc[idx, 'dominantWallRVal']):
                    r_value = float(transformed_df.loc[idx, 'dominantWallRVal'])
                    r_norm = min(1.0, r_value / 5.0)  # Normalize (0-1)
                    building_quality += r_norm * 0.4  # 40% weight
                    
                # Use air tightness as second indicator
                if pd.notna(transformed_df.loc[idx, 'ach']):
                    ach = float(transformed_df.loc[idx, 'ach'])
                    ach_inv = max(0, 1.0 - (ach / 10.0))  # Invert and normalize (less ach = better)
                    building_quality += ach_inv * 0.3  # 30% weight
                    
                # Add construction year factor
                if pd.notna(transformed_df.loc[idx, 'vintageExact']):
                    year = float(transformed_df.loc[idx, 'vintageExact'])
                    year_norm = min(1.0, max(0.0, (year - 1950) / 70.0))  # Normalize 1950-2020
                    building_quality += year_norm * 0.3  # 30% weight
                    
                # If missing data, use average value
                if building_quality == 0.0:
                    building_quality = 0.5
                    
                # 2. Random factor for additional variation (±10%)
                random_factor = 0.9 + (np.random.random() * 0.2)
                
                # 3. Calculate final efficiency (85-115% of base based on quality + random variation)
                base_cop = scenario['efficiency']['cop_heat_pump']
                quality_factor = 0.85 + (building_quality * 0.3)  # 0.85-1.15 based on quality
                final_cop = base_cop * quality_factor * random_factor
                
                # Heating efficiency (COP)
                transformed_df.loc[idx, 'heatPumpHeatingEff'] = final_cop
                
                # Cooling efficiency (COP * proportional factor)
                cooling_factor = 0.85 + (np.random.random() * 0.1)  # 0.85-0.95
                transformed_df.loc[idx, 'heatPumpCoolingEff'] = final_cop * cooling_factor
                
                # Clear original system properties
                transformed_df.loc[idx, 'spaceHeatingType'] = np.nan
                transformed_df.loc[idx, 'spaceHeatingFuel'] = np.nan
                
                stats['to_heat_pump'] += 1
                
                # Log first archetypes for verification
                self.logger.debug(f"Archetype {idx} - Transformed to heat pump:")
                self.logger.debug(f"  → Original system: {item['system']} → heat pump")
                self.logger.debug(f"  → Heating COP: {final_cop:.2f} (base: {base_cop}, quality: {quality_factor:.2f}, random: {random_factor:.2f})")
                self.logger.debug(f"  → Cooling COP: {final_cop * cooling_factor:.2f}")
            else:
                # Transform to standard electric heating - use .loc
                transformed_df.loc[idx, 'spaceHeatingType'] = 'Baseboards'
                transformed_df.loc[idx, 'spaceHeatingFuel'] = 'Electric'
                transformed_df.loc[idx, 'spaceHeatingEff'] = 100.0  # Max efficiency for electric
                
                stats['to_electric'] += 1
                
                # Add log for verification
                self.logger.debug(f"Archetype {idx} - Transformed to electric heating:")
                self.logger.debug(f"  → Original system: {item['system']} → Electric")
        
        # Log results
        self.logger.info(f"Transformation stats: {stats}")
        self.logger.info(f"New heating system mix: " + 
                       f"{self.analyze_heating_systems_distribution(transformed_df)}")
        
        return transformed_df

    def analyze_heating_systems_distribution(self, archetypes_df: pd.DataFrame) -> dict:
        """
        Analyze heating system distribution in archetypes.
        
        Args:
            archetypes_df: Archetypes DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Count main heating systems
        heating_systems = {}
        total = len(archetypes_df)
        
        # First count heat pumps
        heat_pumps = 0
        for _, archetype in archetypes_df.iterrows():
            if pd.notna(archetype.get('heatPumpType')):
                heat_pumps += 1
        
        heating_systems['heat_pump'] = heat_pumps / total if total > 0 else 0
        
        # Count other systems
        system_counts = {}
        for _, archetype in archetypes_df.iterrows():
            # Skip archetypes with heat pumps
            if pd.notna(archetype.get('heatPumpType')):
                continue
                
            fuel = archetype.get('spaceHeatingFuel')
            if pd.notna(fuel):
                system_counts[fuel] = system_counts.get(fuel, 0) + 1
        
        # Convert to percentages
        for system, count in system_counts.items():
            heating_systems[system] = count / total if total > 0 else 0
        
        return heating_systems