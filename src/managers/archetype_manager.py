"""
ArchetypeManager handles loading and processing of building archetypes.
"""
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Callable
import json
import re
from dataclasses import dataclass
import numpy as np
from src.config import config
from src.utils import ensure_dir
from src.schedule_generator import ScheduleGenerator

@dataclass
class BuildingConfig:
    """Configuration pour un bâtiment HPXML"""
    building_type: str
    geometry_building_num_units: int = 1
    whole_sfa_or_mf_building_sim: bool = False

@dataclass
class HPXMLMapping:
    """Définit le mapping d'une variable archetype vers HPXML"""
    hpxml_name: str
    archetype_name: str
    conversion_func: Callable = None
    mapping_dict: Dict = None
    default_value: Any = None

class UnitConverter:
    """Classe utilitaire pour les conversions d'unités"""

    @staticmethod
    def temperature_to_schedule(value: float, is_heating: bool = True, 
                              archetype_id: int = 0, 
                              temporal_diversity: float = 0.0) -> str:
        """Convertit une température en horaire avec courbes paraboliques lisses."""
        # S'assurer que value est un nombre
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                # Si la conversion échoue, retourner la valeur d'origine
                return value
                
        # Conversion C à F si nécessaire
        temp_f = value if value > 50 else (value * 1.8 + 32)
        
        # Générer un décalage unique pour cet archétype
        import hashlib
        import random
        import math
        
        seed = int(hashlib.md5(f"{archetype_id}_{value}".encode()).hexdigest(), 16) % 10000
        r = random.Random(seed)  # Générateur pseudo-aléatoire avec graine fixe
        
        # PARAMÈTRES PARABOLIQUES AMÉLIORÉS
        # Amplitude de base (°F)
        base_amplitude = 0.5
        # Amplitude additionnelle selon diversité (°F)
        max_additional_amplitude = 2.0
        # Amplitude totale
        amplitude = base_amplitude + (max_additional_amplitude * temporal_diversity) * r.random()
        
        # Décalage horaire (0-5h selon la diversité temporelle)
        time_shift = int(5 * temporal_diversity * r.random())
        
        # Température de référence (point central)
        base_temp = temp_f
        
        # Générer les 24 valeurs horaires avec une véritable courbe parabolique lisse
        schedule = []
        
        # Paramètres de la courbe pour chauffage/climatisation
        if is_heating:
            # Pour le chauffage: courbe en forme de cloche inversée
            # Plus froid la nuit (1-6h), plus chaud le jour (10-16h)
            morning_hour = 6 + r.randint(-1, 1) * int(temporal_diversity + 1)
            peak_hour = 13 + r.randint(-2, 2) * int(temporal_diversity + 1)
            evening_hour = 20 + r.randint(-1, 1) * int(temporal_diversity + 1)
            
            for hour in range(24):
                # Appliquer décalage horaire
                h = (hour + time_shift) % 24
                
                # Création d'une courbe sinusoïdale sur 24h
                # Convertir l'heure en position angulaire (0h = 0°, 24h = 360°)
                # Phase ajustée pour que le minimum soit autour de 3h du matin
                phase_shift = math.pi * (3.0 / 12.0)
                angle = (h / 24.0) * 2 * math.pi - phase_shift
                
                # Fonction sinusoïdale: -cos car minimum à 0° (3h), maximum à 180° (15h)
                # Oscillation entre -1 et 1
                sin_value = -math.cos(angle)
                
                # Convertir en ajustement de température
                if is_heating:
                    # Pour chauffage: minimum la nuit (-amplitude), maximum le jour (+0)
                    adj = -amplitude * (1 - sin_value) / 2
                else:
                    # Pour climatisation: maximum la nuit (+0), minimum le jour (-amplitude)
                    adj = -amplitude * (1 + sin_value) / 2
                
                # Ajouter à la liste
                schedule.append(base_temp + adj)
        else:
            # Pour la climatisation: courbe en forme de cloche
            # Plus chaud la nuit (1-6h), plus frais le jour (10-16h)
            morning_hour = 6 + r.randint(-1, 1) * int(temporal_diversity + 1)
            peak_hour = 13 + r.randint(-2, 2) * int(temporal_diversity + 1)
            evening_hour = 20 + r.randint(-1, 1) * int(temporal_diversity + 1)
            
            for hour in range(24):
                # Appliquer décalage horaire
                h = (hour + time_shift) % 24
                
                # Même approche que pour le chauffage mais inversée
                phase_shift = math.pi * (3.0 / 12.0)
                angle = (h / 24.0) * 2 * math.pi - phase_shift
                sin_value = -math.cos(angle)
                
                # Convertir en ajustement de température
                # Pour climatisation: maximum la nuit (+0), minimum le jour (-amplitude)
                adj = amplitude * (1 - sin_value) / 2
                
                # Ajouter à la liste
                schedule.append(base_temp + adj)
        
        # Arrondir à un chiffre après la virgule et joindre avec des virgules
        return ", ".join([f"{round(temp, 1)}" for temp in schedule])
    
    @staticmethod
    def c_to_f(value: float) -> str:
        """Convertit Celsius en Fahrenheit"""
        return str(value * 1.8 + 32)

    @staticmethod
    def m2_to_ft2(value: float) -> float:
        """Convertit mètres carrés en pieds carrés"""
        return value * 10.7639

    @staticmethod
    def m_to_ft(value: float) -> float:
        """Convertit mètres en pieds"""
        return value * 3.28084

    @staticmethod
    def l_to_gal(value: float) -> float:
        """Convertit litres en gallons"""
        return value * 0.264172 if value > 0 else None

    @staticmethod
    def rsi_to_rvalue(value: float) -> Optional[float]:
        """Convertit RSI (m²·K/W) en R-value (h·ft²·°F/Btu)"""
        return value * 5.678263337 if value > 0 else None

    @staticmethod
    def w_to_btu_hr(value: float) -> float:
        """Convertit Watts en BTU/hr"""
        return value * 3.412142

    @staticmethod
    def kw_to_btu_hr(value: float) -> float:
        """Convertit kW en BTU/hr"""
        return value * 1000 * 3.412142

    @staticmethod
    def rsi_to_ufactor(value: float) -> Optional[float]:
        """Convertit RSI en U-factor"""
        if value > 0:
            return 1.0 / (value * 5.678263337)
        return None

    @staticmethod
    def cop_to_efficiency(cop: float, system_type: str) -> float:
        """Convertit COP en SEER/EER selon le type"""
        if system_type in ['Central split system', 'Mini-split']:
            return cop * 3.792  # COP -> SEER
        else:  # Room AC, PTAC
            return cop * 3.412  # COP -> EER

    @staticmethod
    def calculate_occupants(n_adults: float, n_children: float) -> float:
        """
        Calcule le nombre total d'occupants.
        
        Args:
            n_adults: Nombre d'adultes
            n_children: Nombre d'enfants
            
        Returns:
            Nombre total d'occupants
        """
        return n_adults + n_children

    @staticmethod
    def convert_vintage(value: float) -> Optional[int]:
        """Convertit l'année de construction"""
        return int(value) if value > 0 else None

    @staticmethod
    def convert_efficiency(value: float) -> float:
        """Convertit l'efficacité en pourcentage à décimal"""
        return float(value/100)

    @staticmethod
    def convert_vent_flow(value: float) -> float:
        """Convertit L/s en CFM"""
        return value * 2.11888

    @staticmethod
    def heating_schedule(value: float, archetype_id: int = 0, temporal_diversity: float = 0.0) -> str:
        """Crée un horaire de chauffage à partir d'une température de base"""
        # Convertir en nombre si c'est une chaîne
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return value
        return UnitConverter.temperature_to_schedule(value, is_heating=True, 
                                                    archetype_id=archetype_id, 
                                                    temporal_diversity=temporal_diversity)

    @staticmethod
    def cooling_schedule(value: float, archetype_id: int = 0, temporal_diversity: float = 0.0) -> str:
        """Crée un horaire de climatisation à partir d'une température de base"""
        # Convertir en nombre si c'est une chaîne
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return value
        return UnitConverter.temperature_to_schedule(value, is_heating=False, 
                                                   archetype_id=archetype_id, 
                                                   temporal_diversity=temporal_diversity)
    
    @staticmethod
    def convert_heatpump_heating_efficiency(value, eff_type, hp_type):
        """
        Convertit l'efficacité de chauffage de la thermopompe au format requis par HPXML.
        
        Args:
            value: Valeur d'efficacité originale
            eff_type: Type d'efficacité d'origine ("COP", etc.)
            hp_type: Type de thermopompe ('AirHeatPump', 'GroundHeatPump', etc.)
            
        Returns:
            Valeur d'efficacité convertie
        """
        if pd.isna(value):
            return None
            
        # Si la valeur est déjà en COP et que le type de thermopompe nécessite COP
        if eff_type == "COP" and hp_type in ['GroundHeatPump', 'WaterHeatPump']:
            return value
            
        # Conversion COP à HSPF pour air-to-air/mini-split (facteur approximatif)
        if eff_type == "COP" and hp_type == 'AirHeatPump':
            return value * 3.41  # Facteur approximatif COP → HSPF
            
        # Valeur par défaut si aucune conversion spécifique
        return value

    @staticmethod
    def convert_heatpump_cooling_efficiency(value, eff_type, hp_type):
        """
        Convertit l'efficacité de refroidissement de la thermopompe au format requis par HPXML.
        
        Args:
            value: Valeur d'efficacité originale
            eff_type: Type d'efficacité d'origine (True/False/NaN)
            hp_type: Type de thermopompe ('AirHeatPump', 'GroundHeatPump', etc.)
            
        Returns:
            Valeur d'efficacité convertie
        """
        if pd.isna(value):
            return None
            
        # Pour les thermopompes air-air, conversion en SEER si eff_type=True
        if eff_type is True and hp_type == 'AirHeatPump':
            # SEER ≈ EER * 1.1 (approximation)
            return value * 1.1
            
        # Pour les thermopompes géothermiques, garder en EER
        if hp_type in ['GroundHeatPump', 'WaterHeatPump']:
            return value
            
        # Valeur par défaut
        return value

class BuildingTypeMapper:
    """Gère la détermination du type de bâtiment HPXML"""
    def __init__(self):
        self.type_mapping = {
            'House': {
                'Single Detached': 'single-family detached',
                'Double/Semi-detached': 'single-family attached',
                'Row house, end unit': 'single-family attached',
                'Row house, middle unit': 'single-family attached',
                'Mobile Home': 'manufactured home'
            },
            'Multi-unit: one unit': {
                'Apartment': 'apartment unit',
                'Apartment Row': 'apartment unit'
            }
        }

    def get_building_types(self):
        """Retourne tous les types de bâtiments possibles"""
        types = set()
        for main_types in self.type_mapping.values():
            types.update(main_types.values())
        return types

    def get_building_hpxml_config(self, archetype: dict) -> BuildingConfig:
        """Détermine la configuration HPXML pour un archetype"""
        house_type = archetype['houseType']
        sub_type = archetype['houseSubType']

        # 1. Maisons unifamiliales (House)
        if house_type == 'House':
            if sub_type == 'Single Detached':
                return BuildingConfig(
                    building_type='single-family detached',
                    geometry_building_num_units=1
                )
            elif sub_type == 'Mobile Home':
                return BuildingConfig(
                    building_type='manufactured home',
                    geometry_building_num_units=1
                )
            else:  # Double/Semi-detached, Row houses
                return BuildingConfig(
                    building_type='single-family attached',
                    geometry_building_num_units=2  # Double = 2 unités
                )

        # 2. Multi-unit: one unit
        if house_type == 'Multi-unit: one unit':
            return BuildingConfig(
                building_type='apartment unit',
                geometry_building_num_units=int(archetype.get('murbDwellingCount', 1))
            )

        # 3. Multi-unit: whole building
        if house_type == 'Multi-unit: whole building':
            num_units = int(archetype.get('murbDwellingCount', 1))
            if 'Duplex' in sub_type:
                return BuildingConfig(
                    building_type='single-family attached',
                    geometry_building_num_units=2
                )
            return BuildingConfig(
                building_type='apartment unit',
                geometry_building_num_units=num_units
            )

class ArchetypeManager:
    """Manages building archetypes and their conversion to HPXML parameters."""
    
    def __init__(self):
        """Initialize the ArchetypeManager."""
        self.logger = logging.getLogger(__name__)
        self.archetypes_df = None
        self.weather_mapping = {}
        self.current_weather_year = None
        self.current_scenario_key = None
        self.type_mapper = BuildingTypeMapper()
        self.converter = UnitConverter()
        self.mappings = self._init_mappings()
        self.schedule_generator = ScheduleGenerator()
        self._load_archetypes()
    
    def _convert_storeys_to_int(self, storeys_str: str) -> Optional[int]:
        """Convert string representation of storeys to integer."""
        if pd.isna(storeys_str):
            return None
            
        # Convert word to number
        number_map = {
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5
        }
        
        # Extract first word and convert to lowercase
        match = re.match(r'^(\w+)', storeys_str.lower())
        if match:
            word = match.group(1)
            if word in number_map:
                return number_map[word]
        
        return None
    
    def _init_mappings(self) -> Dict[str, HPXMLMapping]:
        """Initialize HPXML mappings"""

        
        return {
            'year_built': HPXMLMapping(
                hpxml_name='year_built',
                archetype_name='vintageExact',
                conversion_func=self.converter.convert_vintage
            ),
            'geometry_unit_num_occupants': HPXMLMapping(
                hpxml_name='geometry_unit_num_occupants',
                archetype_name=['numAdults', 'numChildren'],
                conversion_func=self.converter.calculate_occupants
            ),
            'geometry_unit_cfa': HPXMLMapping(
                hpxml_name='geometry_unit_cfa',
                archetype_name='totFloorArea',
                conversion_func=self.converter.m2_to_ft2
            ),
            'hvac_control_heating_weekday_setpoint': HPXMLMapping(
                hpxml_name='hvac_control_heating_weekday_setpoint',
                archetype_name='dayHeatingSetPoint',
                conversion_func=self.converter.heating_schedule
            ),
            'hvac_control_heating_weekend_setpoint': HPXMLMapping(
                hpxml_name='hvac_control_heating_weekend_setpoint',
                archetype_name='dayHeatingSetPoint',
                conversion_func=self.converter.heating_schedule
            ),
            'hvac_control_cooling_weekday_setpoint': HPXMLMapping(
                hpxml_name='hvac_control_cooling_weekday_setpoint',
                archetype_name='coolingSetPoint',
                conversion_func=self.converter.cooling_schedule
            ),
            'hvac_control_cooling_weekend_setpoint': HPXMLMapping(
                hpxml_name='hvac_control_cooling_weekend_setpoint',
                archetype_name='coolingSetPoint',
                conversion_func=self.converter.cooling_schedule
            ),
            'heating_system_type': HPXMLMapping(
                hpxml_name='heating_system_type',
                archetype_name='spaceHeatingType',
                mapping_dict={
                    'Baseboards': 'ElectricResistance',
                    'Furnace': 'Furnace',
                    'Boiler': 'Boiler',
                    'ComboHeatDhw': 'Boiler'
                }
            ),
            'heating_system_fuel': HPXMLMapping(
                hpxml_name='heating_system_fuel',
                archetype_name='spaceHeatingFuel',
                mapping_dict={
                    'Electric': 'electricity',
                    'Natural gas': 'natural gas',
                    'Oil': 'fuel oil',
                    'Mixed Wood': 'wood',
                    'Hardwood': 'wood',
                    'Wood Pellets': 'wood pellets'
                }
            ),
            'heating_system_heating_efficiency': HPXMLMapping(
                hpxml_name='heating_system_heating_efficiency',
                archetype_name='spaceHeatingEff',
                conversion_func=self.converter.convert_efficiency
            ),
            'heating_system_heating_capacity': HPXMLMapping(
                hpxml_name='heating_system_heating_capacity',
                archetype_name='spaceHeatingCapacity', 
                conversion_func=self.converter.kw_to_btu_hr  
            ),
            #supply heating system
            'heating_system_2_fuel': HPXMLMapping(
                hpxml_name='heating_system_2_fuel',
                archetype_name='supplHeatingFuel',
                mapping_dict={
                    'Electric': 'electricity',
                    'Natural gas': 'natural gas',
                    'Oil': 'fuel oil',
                    'Mixed Wood': 'wood',
                    'Hardwood': 'wood',
                    'Wood Pellets': 'wood pellets',
                    'Propane': 'propane'
                }
            ),
            'heat_pump_type': HPXMLMapping(
                hpxml_name='heat_pump_type',
                archetype_name='heatPumpType',
                mapping_dict={
                    'AirHeatPump': 'air-to-air',  
                    'GroundHeatPump': 'ground-to-air',
                    'WaterHeatPump': 'water-to-air'
                },
                default_value='none'
            ),
            'heat_pump_heating_efficiency': HPXMLMapping(
                hpxml_name='heat_pump_heating_efficiency',
                archetype_name=['heatPumpHeatingEff', 'heatPumpHeatingEffType', 'heatPumpType'],
                conversion_func=self.converter.convert_heatpump_heating_efficiency
            ),

            'heat_pump_cooling_efficiency': HPXMLMapping(
                hpxml_name='heat_pump_cooling_efficiency',
                archetype_name=['heatPumpCoolingEff', 'heatPumpCoolingEffType', 'heatPumpType'],
                conversion_func=self.converter.convert_heatpump_cooling_efficiency
            ),
            'heat_pump_heating_efficiency_type': HPXMLMapping(
                hpxml_name='heat_pump_heating_efficiency_type',
                archetype_name='heatPumpType',
                mapping_dict={
                    'AirHeatPump': 'HSPF',
                    'GroundHeatPump': 'COP',
                    'WaterHeatPump': 'COP'
                },
                default_value='COP'
            ),

            'heat_pump_cooling_efficiency_type': HPXMLMapping(
                hpxml_name='heat_pump_cooling_efficiency_type',
                archetype_name='heatPumpType',
                mapping_dict={
                    'AirHeatPump': 'SEER',
                    'GroundHeatPump': 'EER',
                    'WaterHeatPump': 'EER'
                },
                default_value='SEER'
            ),
            'heating_system_2_heating_efficiency': HPXMLMapping(
                hpxml_name='heating_system_2_heating_efficiency',
                archetype_name='supplHeatingEff',
                conversion_func=self.converter.convert_efficiency
            ),
            'heating_system_2_heating_capacity': HPXMLMapping(
                hpxml_name='heating_system_2_heating_capacity',
                archetype_name='supplHeatingCapacity',  # En kW
                conversion_func=self.converter.kw_to_btu_hr  # kW à BTU/hr
            ),
            'cooling_system_type': HPXMLMapping(
                hpxml_name='cooling_system_type',
                archetype_name='coolingEquipType',
                mapping_dict={
                    'Central split system': 'central air conditioner',
                    'Mini-split ductless': 'mini-split',
                    'Central single package system': 'packaged terminal air conditioner'
                }
            ),
            'cooling_system_cooling_efficiency_type': HPXMLMapping(
                hpxml_name='cooling_system_cooling_efficiency_type',
                archetype_name='coolingEquipType',
                mapping_dict={
                    'Central split system': 'SEER',
                    'Mini-split ductless': 'SEER',
                    'Central single package system': 'EER'
                }
            ),
            'cooling_system_cooling_efficiency': HPXMLMapping(
                hpxml_name='cooling_system_cooling_efficiency',
                archetype_name=['coolingEff', 'coolingEquipType'],
                conversion_func=self.converter.cop_to_efficiency
            ),
            'cooling_system_cooling_capacity': HPXMLMapping(
                hpxml_name='cooling_system_cooling_capacity',
                archetype_name='coolingCapacity',  # En kW
                conversion_func=self.converter.kw_to_btu_hr  # kW à BTU/hr
            ),
            'water_heater_type': HPXMLMapping(
                hpxml_name='water_heater_type',
                archetype_name='primaryDhwTankType',
                mapping_dict={
                    'Conventional tank': 'storage water heater',
                    'Conserver tank': 'storage water heater',
                    'Instantaneous': 'instantaneous water heater',
                    'Instantaneous (condensing)': 'instantaneous water heater',
                    'Induced draft fan': 'storage water heater',
                    'Conventional tank (pilot)': 'storage water heater',
                    'Direct vent (sealed)': 'storage water heater'
                }
            ),
            'water_heater_fuel_type': HPXMLMapping(
                hpxml_name='water_heater_fuel_type',
                archetype_name='primaryDhwFuel',
                mapping_dict={
                    'Electricity': 'electricity',
                    'Natural gas': 'natural gas',
                    'Oil': 'fuel oil'
                }
            ),
            'water_heater_tank_volume': HPXMLMapping(
                hpxml_name='water_heater_tank_volume',
                archetype_name='primaryDhwTankVolume',
                conversion_func=self.converter.l_to_gal
            ),
            'mech_vent_fan_type': HPXMLMapping(
                hpxml_name='mech_vent_fan_type',
                archetype_name='hrvPresent',
                mapping_dict={
                    True: 'heat recovery ventilator',
                    False: 'none'
                }
            ),
            'mech_vent_flow_rate': HPXMLMapping(
                hpxml_name='mech_vent_flow_rate',
                archetype_name='hrvTotalSupply',
                conversion_func=self.converter.convert_vent_flow
            ),
            'air_leakage_value': HPXMLMapping(
                hpxml_name='air_leakage_value',
                archetype_name='ach'
            ),
            'wall_assembly_r': HPXMLMapping(
                hpxml_name='wall_assembly_r',
                archetype_name='dominantWallRVal',
                conversion_func=self.converter.rsi_to_rvalue
            ),
            'window_ufactor': HPXMLMapping(
                hpxml_name='window_ufactor',
                archetype_name='dominantWindowRVal',
                conversion_func=self.converter.rsi_to_ufactor
            ),
            'window_shgc': HPXMLMapping(
                hpxml_name='window_shgc',
                archetype_name='dominantWindowShgc'
            ),
            'ceiling_assembly_r': HPXMLMapping(
                hpxml_name='ceiling_assembly_r',
                archetype_name='dominantCeilingRVal',
                conversion_func=self.converter.rsi_to_rvalue
            ),
            'slab_perimeter_insulation_r': HPXMLMapping(
                hpxml_name='slab_perimeter_insulation_r',
                archetype_name='dominantSlabRVal',
                conversion_func=self.converter.rsi_to_rvalue
            ),
            'foundation_wall_assembly_r': HPXMLMapping(
                hpxml_name='foundation_wall_assembly_r',
                archetype_name='dominantBasementWallRVal',
                conversion_func=self.converter.rsi_to_rvalue
            ),
            'door_rvalue': HPXMLMapping(
                hpxml_name='door_rvalue',
                archetype_name='dominantDoorRVal',
                conversion_func=self.converter.rsi_to_rvalue
            )
        }
    
    def _load_archetypes(self) -> None:
        """Load base archetypes from CSV file."""
        try:
            self.archetypes_df = pd.read_csv(config.paths['input'].BASE_ARCHETYPES)
            # Reduced logging - only log errors
        except Exception as e:
            self.logger.error(f"Error loading archetypes: {str(e)}")
            raise
    
    def _load_weather_mapping(self, year: int) -> None:
        """Load weather mapping for a specific year."""
        try:
            mapping_file = config.paths['input'].get_weather_mapping()
            weather_df = pd.read_csv(mapping_file)
            
            # Log available columns for debugging
            self.logger.debug(f"Available columns in weather mapping: {list(weather_df.columns)}")
            
            # Try several column name formats
            potential_columns = [
                f"EPW_{year}",
                f"EPW{year}",
                # Add other potential column formats here
            ]
            
            # Find the first matching column
            epw_column = None
            for col_name in potential_columns:
                if col_name in weather_df.columns:
                    epw_column = col_name
                    break
            
            # If no exact match, try to find a column that contains the year
            if epw_column is None:
                year_columns = [col for col in weather_df.columns if str(year) in col and col.startswith('EPW')]
                if year_columns:
                    epw_column = year_columns[0]
                    self.logger.info(f"Using alternative column format: {epw_column}")
            
            # If we still couldn't find a column, try the current year or most recent year before
            if epw_column is None:
                # Get current year
                from datetime import datetime
                current_year = datetime.now().year
                
                # Try columns for the current year or earlier
                for try_year in range(current_year, 2018, -1):  # Try back to 2019
                    try_column = f"EPW_{try_year}"
                    if try_column in weather_df.columns:
                        epw_column = try_column
                        self.logger.warning(f"Using {try_year} weather data as fallback for {year}")
                        break
            
            # If still no column found, raise an error
            if epw_column is None:
                self.logger.error(f"No suitable EPW column found for year {year}")
                raise ValueError(f"No EPW data available for year {year}")
            
            self.logger.info(f"Using weather column: {epw_column}")
            
            # Create mapping dictionary using the year-specific EPW column
            self.weather_mapping = {
                (row['provinces_english'], row['cities_english']): 
                row[epw_column]
                for _, row in weather_df.iterrows()
                if pd.notna(row[epw_column])  # Skip rows with missing EPW values
            }
            
        except Exception as e:
            self.logger.error(f"Error loading weather mapping: {str(e)}")
            raise
    
    def _get_weather_file(self, archetype: pd.Series, year: int) -> Path:
        """Get weather file path for an archetype."""
        
        # Check if this is for a future scenario
        scenario_key = getattr(self, 'current_scenario_key', None)
        
        if scenario_key and config.future_scenarios.is_future_scenario(scenario_key):
            # Get future weather file
            from src.scenario_manager import ScenarioManager
            scenario_manager = ScenarioManager()
            
            relative_path = scenario_manager.get_weather_file(
                int(archetype['weather_zone']), 
                scenario_key
            )
            
            if relative_path:
                return config.paths['input'].WEATHER / relative_path
        
        # Original behavior for historical weather files
        # Check if we need to reload the weather mapping for a different year
        if not self.weather_mapping or self.current_weather_year != year:
            self._load_weather_mapping(year)
            self.current_weather_year = year
            
        key = (archetype['province_std'], archetype['location_std'])
        if key not in self.weather_mapping:
            raise ValueError(f"No weather file found for {key}")
            
        epw_file = self.weather_mapping[key]
        return config.paths['input'].get_weather_dir() / epw_file
    
    def _create_workflow(self, arch_dir: Path, hpxml_params: dict, 
                        archetype: pd.Series, year: int,
                        use_stochastic_schedules: bool = False) -> bool:
        """Create OpenStudio workflow for an archetype."""
        try:
            # Déterminer quel template utiliser (base ou stochastic)
            template_path = (config.paths['openstudio'].STOCHASTIC_WORKFLOW 
                           if use_stochastic_schedules
                           else config.paths['openstudio'].BASE_WORKFLOW)
            
            # Load template
            with open(template_path, 'r') as f:
                workflow = json.load(f)
            
            # Clean base arguments
            base_arguments = workflow['steps'][0]['arguments']
            cleaned_arguments = {}
            
            # Ne garder que les arguments essentiels du workflow de base
            essential_args = {'hpxml_path', 'weather_station_epw_filepath'}
            for arg in essential_args:
                if arg in base_arguments:
                    cleaned_arguments[arg] = base_arguments[arg]
            
            # NOUVEAU: Vérification des thermopompes pour détecter le problème de COP/HSPF
            hp_keys = [k for k in hpxml_params.keys() if k.startswith('heat_pump_')]
            if any(hp_keys):
                # Log pour déboguer les valeurs d'efficacité
                hp_type = hpxml_params.get('heat_pump_type', 'unknown')
                heat_eff = hpxml_params.get('heat_pump_heating_efficiency', 'N/A')
                heat_type = hpxml_params.get('heat_pump_heating_efficiency_type', 'N/A')
                cool_eff = hpxml_params.get('heat_pump_cooling_efficiency', 'N/A')
                cool_type = hpxml_params.get('heat_pump_cooling_efficiency_type', 'N/A')
                
                # Log détaillé pour débogage
                self.logger.debug(f"Archetype {arch_dir.name} - Configuration thermopompe:")
                self.logger.debug(f"  → Type: {hp_type}")
                self.logger.debug(f"  → Efficacité chauffage: {heat_eff} ({heat_type})")
                self.logger.debug(f"  → Efficacité refroidissement: {cool_eff} ({cool_type})")
                
                # Vérifier la cohérence des valeurs
                if heat_type == 'HSPF' and heat_eff < 6.0:
                    # Valeur probablement en COP mais type HSPF - appliquer conversion
                    original = heat_eff
                    hpxml_params['heat_pump_heating_efficiency'] = heat_eff * 3.41
                    self.logger.warning(f"Correction auto thermopompe: COP → HSPF ({original} → {hpxml_params['heat_pump_heating_efficiency']})")
            
            # Mettre à jour avec les arguments HPXML de l'archétype
            # Ne garder que les valeurs non-NaN et filtrer les paramètres virtuels
            # Liste des paramètres virtuels qui ne sont pas de vrais paramètres HPXML
            virtual_params = ['occupancy_scale', 'lighting_scale', 'appliance_scale']
            
            valid_hpxml_params = {
                key: value for key, value in hpxml_params.items()
                if pd.notna(value) and key not in virtual_params  # Vérifie que la valeur n'est pas NaN et n'est pas un paramètre virtuel
            }
            cleaned_arguments.update(valid_hpxml_params)
            
            # Add weather file
            weather_file = self._get_weather_file(archetype, year)
            cleaned_arguments['weather_station_epw_filepath'] = str(weather_file)
            
            # Update workflow
            workflow['steps'][0]['arguments'] = cleaned_arguments
            workflow['run_directory'] = str(arch_dir / 'run')
            
            # Save workflow
            workflow_path = arch_dir / 'workflow.osw'
            with open(workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            return False
    
    def _generate_schedule_for_archetype(self, archetype_id: int, 
                                    archetype: pd.Series,
                                    output_dir: Path,
                                    scale_factors: Optional[Dict[str, float]] = None,
                                    temporal_diversity: float = 0.0) -> Optional[Path]:
        """
        Génère un schedule pour un archétype spécifique.
        
        Args:
            archetype_id: ID de l'archétype
            archetype: Données de l'archétype
            output_dir: Répertoire de sortie
            scale_factors: Facteurs d'échelle pour différents types de profils
            temporal_diversity: Facteur de diversité temporelle entre archétypes [0.0-1.0]
            
        Returns:
            Chemin vers le fichier de schedule généré
        """
        try:
            # Liste standard des schedules à générer
            profile_types = [
                'occupants',
                'lighting_interior',
                'lighting_garage',
                'cooking_range',
                'dishwasher',
                'clothes_washer',
                'clothes_dryer',
                'ceiling_fan',
                'plug_loads_other',
                'plug_loads_tv',
                'hot_water_dishwasher',
                'hot_water_clothes_washer',
                'hot_water_fixtures'
            ]
            

            
            # Générer et sauvegarder le schedule
            schedules_dir = ensure_dir(output_dir / "schedules")
            schedule_path = self.schedule_generator.generate_and_save_schedule(
                profile_types=profile_types,
                archetype_id=archetype_id,
                archetype_data=archetype.to_dict(),
                scale_factors=scale_factors,
                temporal_diversity=temporal_diversity,
                output_dir=schedules_dir
            )
            
            return schedule_path
        
        except Exception as e:
            self.logger.error(f"Error generating schedule for archetype {archetype_id}: {str(e)}")
            return None
            
    def prepare_archetype(self, archetype_id: int, output_dir: Path, 
                     year: int,
                     calibration_params: Optional[Dict[str, Dict[int, float]]] = None,
                     use_stochastic_schedules: bool = False) -> Optional[Path]:
        """
        Prepare an archetype for simulation.
        
        Args:
            archetype_id: ID of the archetype to prepare
            output_dir: Directory to store the prepared files
            year: Simulation year
            calibration_params: Optional calibration parameters by archetype
                            {param_name: {archetype_id: value}}
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Path to the prepared archetype directory if successful, None otherwise
        """
        try:
            # Get archetype data
            archetype = self.archetypes_df.iloc[archetype_id]
            
            # Create archetype directory
            arch_dir = ensure_dir(output_dir / f"archetype_{archetype_id}")
            
            # Get building configuration
            building_config = self.type_mapper.get_building_hpxml_config(archetype.to_dict())
            
            # Initialize HPXML parameters
            hpxml_params = {
                'geometry_unit_type': building_config.building_type,
                'geometry_building_num_units': building_config.geometry_building_num_units,
                'air_leakage_units': 'ACH',
                'air_leakage_house_pressure': 50.0
            }
            # Vérifier si l'archétype a une thermopompe
            has_heat_pump = pd.notna(archetype.get('heatPumpType'))

            if has_heat_pump:
                # Forcer heating_system_type et cooling_system_type à "none" pour éviter les conflits
                hpxml_params['heating_system_type'] = 'none'
                hpxml_params['cooling_system_type'] = 'none'
                self.logger.debug(f"Archétype {archetype_id} a une thermopompe - systèmes conventionnels désactivés")

            # Préparer les facteurs d'échelle pour les schedules
            schedule_scale_factors = None
            temporal_diversity_value = 0.0
            
            if calibration_params is not None:
                schedule_scale_factors = {}
                # Récupérer les facteurs d'échelle s'ils existent
                for scale_param in ['occupancy_scale', 'lighting_scale', 'appliance_scale', 'temporal_diversity']:
                    if scale_param in calibration_params and archetype_id in calibration_params[scale_param]:
                        if scale_param == 'temporal_diversity':
                            # Stocker la valeur séparément pour le paramètre temporal_diversity
                            temporal_diversity_value = calibration_params[scale_param][archetype_id]
                            self.logger.debug(f"Archétype {archetype_id}: Diversité temporelle appliquée = {temporal_diversity_value:.3f} pour les setpoints de chauffage/climatisation")
                        else:
                            schedule_scale_factors[scale_param] = calibration_params[scale_param][archetype_id]
            
            # Générer le fichier de schedule si demandé
            if use_stochastic_schedules:
                schedule_path = self._generate_schedule_for_archetype(
                    archetype_id, archetype, arch_dir, schedule_scale_factors, temporal_diversity_value
                )
                if schedule_path is not None:
                    # Ajouter le chemin du schedule aux paramètres HPXML
                    # (quel que soit le template utilisé)
                    hpxml_params['schedules_filepaths'] = str(schedule_path)
                    self.logger.debug(f"Added schedule file to archetype {archetype_id}: {schedule_path}")
            
            # Add air_leakage_type for specific building types
            if building_config.building_type in ['single-family attached', 'apartment unit']:
                hpxml_params['air_leakage_type'] = 'unit total'
            
            # Handle number of floors
            if building_config.building_type == 'apartment unit':
                # Force 1 floor for apartment units
                hpxml_params['geometry_unit_num_floors_above_grade'] = 1
            elif pd.notna(archetype.get('storeys')):
                # For other building types, use storeys if available
                floors = self._convert_storeys_to_int(archetype['storeys'])
                if floors is not None:
                    hpxml_params['geometry_unit_num_floors_above_grade'] = floors
            
            # Apply calibration parameters if provided
            if calibration_params is not None:
                # NOUVEAU: Log détaillé des paramètres de calibration
                self.logger.debug(f"Application des paramètres de calibration à l'archétype {archetype_id}:")
                
                # Map calibration parameters to HPXML parameters and original values
                param_mapping = {
                    # Paramètres originaux
                    'infiltration_rate': ('air_leakage_value', 'ach', None),
                    'wall_rvalue': ('wall_assembly_r', 'dominantWallRVal', self.converter.rsi_to_rvalue),
                    'ceiling_rvalue': ('ceiling_assembly_r', 'dominantCeilingRVal', self.converter.rsi_to_rvalue),
                    'window_ufactor': ('window_ufactor', 'dominantWindowRVal', self.converter.rsi_to_ufactor),
                    
                    # Nouveaux paramètres comportementaux (pour schedules stochastiques)
                    'occupancy_scale': ('', '', None),  # Traité séparément pour les schedules
                    'lighting_scale': ('', '', None),   # Traité séparément pour les schedules
                    'appliance_scale': ('', '', None),  # Traité séparément pour les schedules
                    'heating_setpoint': ('hvac_control_heating_weekday_setpoint', 'dayHeatingSetPoint', self.converter.c_to_f),
                    'cooling_setpoint': ('hvac_control_cooling_weekday_setpoint', 'coolingSetPoint', self.converter.c_to_f),
                    
                    # Paramètres des systèmes
                    # Paramètres des systèmes - adaptation pour thermopompes
                    'heating_efficiency': ('heating_system_heating_efficiency' 
                        if not has_heat_pump else 'heat_pump_heating_efficiency',
                        'spaceHeatingEff' if not has_heat_pump else 'heatPumpHeatingEff',
                        self.converter.convert_efficiency),
                    'dhw_volume': ('water_heater_tank_volume', 'primaryDhwTankVolume', self.converter.l_to_gal),
                    'foundation_r': ('foundation_wall_assembly_r', 'dominantBasementWallRVal', self.converter.rsi_to_rvalue)
                }
                
                applied_params = []
                
                for param_name, (hpxml_name, original_col, converter) in param_mapping.items():
                    if param_name in calibration_params and archetype_id in calibration_params[param_name]:
                        # Get adjustment factor (-0.3 to +0.3)
                        adjustment = calibration_params[param_name][archetype_id]
                        
                        # Log l'ajustement reçu
                        self.logger.debug(f"  → Calibration {param_name}: ajustement = {adjustment}")
                        
                        # Skip virtual parameters used only for schedule generation
                        # These aren't actual HPXML parameters and shouldn't be included in the workflow
                        if param_name in ['occupancy_scale', 'lighting_scale', 'appliance_scale', 'temporal_diversity']:
                            self.logger.debug(f"Skipping virtual parameter {param_name} in HPXML workflow")
                            continue
                            
                        # For regular parameters, get original value
                        original_value = archetype[original_col]
                        if pd.isna(original_value):
                            self.logger.debug(f"  → Valeur originale '{original_col}' est NaN, paramètre ignoré")
                            continue
                        
                        # Log la valeur originale
                        self.logger.debug(f"  → Valeur originale '{original_col}': {original_value}")
                            
                        # Apply adjustment to original value
                        adjusted_value = original_value * (1 + adjustment)
                        self.logger.debug(f"  → Valeur ajustée: {adjusted_value}")
                        
                        # Vérifier les contraintes spécifiques à certains paramètres
                        if param_name == 'heating_efficiency' and adjusted_value > 100:
                            self.logger.debug(f"Heating efficiency ajustée ({adjusted_value}) > 100%, limitée à 100%")
                            adjusted_value = 100.0
                        
                        # Apply conversion if needed
                        if converter is not None:
                            converted_value = converter(adjusted_value)
                            self.logger.debug(f"  → Valeur convertie: {converted_value}")
                            adjusted_value = converted_value
                            
                            # Vérification supplémentaire après conversion
                            if param_name == 'heating_efficiency' and adjusted_value > 1.0:
                                self.logger.debug(f"Heating efficiency convertie ({adjusted_value}) > 1.0, limitée à 1.0")
                                adjusted_value = 1.0
                            
                        # Add to HPXML parameters (only for valid HPXML parameters)
                        if pd.notna(adjusted_value) and hpxml_name:
                            hpxml_params[hpxml_name] = adjusted_value
                            applied_params.append(f"{hpxml_name}={adjusted_value}")
                            
                            # Appliquer les mêmes valeurs aux consignes de week-end
                            if hpxml_name == 'hvac_control_heating_weekday_setpoint':
                                adjusted_value = self.converter.heating_schedule(adjusted_value, archetype_id, temporal_diversity_value)
                                hpxml_params[hpxml_name] = adjusted_value
                                hpxml_params['hvac_control_heating_weekend_setpoint'] = adjusted_value
                            elif hpxml_name == 'hvac_control_cooling_weekday_setpoint':
                                adjusted_value = self.converter.cooling_schedule(adjusted_value, archetype_id, temporal_diversity_value)
                                hpxml_params[hpxml_name] = adjusted_value
                                hpxml_params['hvac_control_cooling_weekend_setpoint'] = adjusted_value
                        
                        # Traitement spécial pour les thermopompes après la calibration
                        if param_name == 'heating_efficiency' and has_heat_pump:
                            # Si le paramètre n'a pas été correctement appliqué, le forcer ici
                            hp_type = archetype.get('heatPumpType')
                            hp_equip_type = archetype.get('heatPumpEquipType')
                            
                            # Pour les thermopompes air-air, conversion COP vers HSPF si nécessaire
                            if hp_type == 'AirHeatPump' or hp_equip_type == 'Mini-split ductless':
                                if archetype.get('heatPumpHeatingEffType') == 'COP':
                                    adjusted_value *= 3.41  # COP vers HSPF
                            
                            hpxml_params['heat_pump_heating_efficiency'] = adjusted_value
                            self.logger.debug(f"Thermopompe {archetype_id}: Efficacité chauffage calibrée à {adjusted_value}")
                
                # NOUVEAU: Résumé des paramètres appliqués
                if applied_params:
                    self.logger.debug(f"Paramètres calibrés appliqués à l'archétype {archetype_id}: {', '.join(applied_params)}")
                else:
                    self.logger.debug(f"Aucun paramètre de calibration appliqué à l'archétype {archetype_id}")
            
            # Apply mappings
            for mapping in self.mappings.values():
                try:
                    # Skip parameters that are being calibrated
                    # Map HPXML names to calibration parameter names
                    hpxml_to_calibration = {
                        'air_leakage_value': 'infiltration_rate',
                        'wall_assembly_r': 'wall_rvalue',
                        'ceiling_assembly_r': 'ceiling_rvalue',
                        'window_ufactor': 'window_ufactor',
                        'geometry_unit_num_occupants': 'occupancy',
                        'hvac_control_heating_weekday_setpoint': 'heating_setpoint',
                        'hvac_control_heating_weekend_setpoint': 'heating_setpoint',
                        'hvac_control_cooling_weekday_setpoint': 'cooling_setpoint',
                        'hvac_control_cooling_weekend_setpoint': 'cooling_setpoint',
                        'heating_system_heating_efficiency': 'heating_efficiency',
                        'heat_pump_heating_efficiency': 'heating_efficiency', 
                        'water_heater_tank_volume': 'dhw_volume',
                        'foundation_wall_assembly_r': 'foundation_r'
                    }
                    
                    # Special handling for weekend setpoints
                    # If weekday setpoint is calibrated, skip weekend setpoint too
                    if calibration_params is not None:
                        # Heating weekend setpoint
                        if (mapping.hpxml_name == 'hvac_control_heating_weekend_setpoint' and
                            'heating_setpoint' in calibration_params):
                            continue
                        
                        # Cooling weekend setpoint
                        if (mapping.hpxml_name == 'hvac_control_cooling_weekend_setpoint' and
                            'cooling_setpoint' in calibration_params):
                            continue
                    
                    # Skip only if this specific parameter is being calibrated
                    if (calibration_params is not None and 
                        mapping.hpxml_name in hpxml_to_calibration and
                        hpxml_to_calibration[mapping.hpxml_name] in calibration_params):
                        continue
                    

                    if isinstance(mapping.archetype_name, list):
                        # Multiple input values (like occupants)
                        values = []
                        valid_values = True
                        for name in mapping.archetype_name:
                            if name not in archetype:
                                valid_values = False
                                break
                            val = archetype[name]
                            if pd.isna(val):
                                valid_values = False
                                break
                            values.append(float(val))  # Convert to float for numeric operations
                        
                        if not valid_values:
                            continue
                            
                        # Apply conversion function with unpacked values
                        value = mapping.conversion_func(*values)
                        
                        
                    else:
                        # Single input value
                        if mapping.archetype_name not in archetype:
                            continue
                        value = archetype[mapping.archetype_name]
                        if pd.isna(value):
                            continue
                            
                        if mapping.mapping_dict is not None:
                            value = mapping.mapping_dict[value]
                        
                        if mapping.conversion_func is not None:
                            value = mapping.conversion_func(value)
                    
                    # Add value only if not None or NaN
                    if pd.notna(value):
                        hpxml_params[mapping.hpxml_name] = value
                        
                        
                except (KeyError, ValueError, TypeError) as e:
                    if mapping.default_value is not None:
                        hpxml_params[mapping.hpxml_name] = mapping.default_value
            
            # Traitement spécial pour les types de thermopompes spécifiques
            if has_heat_pump:
                # Forcer heating_system_type et cooling_system_type à "none"
                hpxml_params['heating_system_type'] = 'none'
                hpxml_params['cooling_system_type'] = 'none'
                
                # Récupérer le type de thermopompe
                hp_type = archetype.get('heatPumpType')
                hp_equip_type = archetype.get('heatPumpEquipType')
                
                # Établir le type HPXML de thermopompe
                if hp_equip_type == 'Mini-split ductless':
                    hpxml_params['heat_pump_type'] = 'mini-split'
                elif hp_type == 'AirHeatPump':
                    hpxml_params['heat_pump_type'] = 'air-to-air'
                elif hp_type == 'GroundHeatPump':
                    hpxml_params['heat_pump_type'] = 'ground-to-air'
                elif hp_type == 'WaterHeatPump':
                    hpxml_params['heat_pump_type'] = 'water-to-air'
                
                # Définir les types d'efficacité en fonction du type de thermopompe
                if hp_type == 'AirHeatPump' or hp_equip_type == 'Mini-split ductless':
                    hpxml_params['heat_pump_heating_efficiency_type'] = 'HSPF'
                    hpxml_params['heat_pump_cooling_efficiency_type'] = 'SEER'
                else:  # Géothermique ou autres
                    hpxml_params['heat_pump_heating_efficiency_type'] = 'COP'
                    hpxml_params['heat_pump_cooling_efficiency_type'] = 'EER'
                
                # S'assurer que les efficacités sont cohérentes avec les types
                if 'heat_pump_heating_efficiency' in hpxml_params and 'heat_pump_heating_efficiency_type' in hpxml_params:
                    hp_heating_eff = hpxml_params['heat_pump_heating_efficiency']
                    hp_heating_type = hpxml_params['heat_pump_heating_efficiency_type']
                    
                    # Vérifier la cohérence et convertir si nécessaire
                    if hp_heating_type == 'HSPF' and hp_heating_eff < 6:  # Valeur trop faible pour HSPF
                        # Probable COP, convertir en HSPF
                        hpxml_params['heat_pump_heating_efficiency'] = hp_heating_eff * 3.41
                
                # Ajouter directement les valeurs d'efficacité pour les thermopompes
                if 'heat_pump_heating_efficiency' not in hpxml_params and pd.notna(archetype.get('heatPumpHeatingEff')):
                    heating_eff = float(archetype['heatPumpHeatingEff'])
                    
                    # Convertir en HSPF pour air-to-air si nécessaire
                    if hp_type == 'AirHeatPump' or hp_equip_type == 'Mini-split ductless':
                        if archetype.get('heatPumpHeatingEffType') == 'COP':
                            heating_eff *= 3.41  # COP vers HSPF
                    
                    hpxml_params['heat_pump_heating_efficiency'] = heating_eff
                    self.logger.debug(f"Thermopompe {archetype_id}: Efficacité chauffage {heating_eff}")

                if 'heat_pump_cooling_efficiency' not in hpxml_params and pd.notna(archetype.get('heatPumpCoolingEff')):
                    cooling_eff = float(archetype['heatPumpCoolingEff'])
                    
                    # Convertir selon le type de thermopompe
                    if hp_type == 'AirHeatPump' or hp_equip_type == 'Mini-split ductless':
                        # Si en EER, conversion approximative vers SEER
                        if archetype.get('heatPumpCoolingEffType') is True:
                            cooling_eff *= 1.1  # EER vers SEER approximation
                    
                    hpxml_params['heat_pump_cooling_efficiency'] = cooling_eff
                    self.logger.debug(f"Thermopompe {archetype_id}: Efficacité refroidissement {cooling_eff}")
                
                self.logger.debug(f"Archétype {archetype_id}: Thermopompe configurée - Type: {hpxml_params.get('heat_pump_type')}")

            # Create workflow
            if not self._create_workflow(arch_dir, hpxml_params, archetype, year, use_stochastic_schedules):
                return None
            
            # self.logger.info(f"Prepared archetype {archetype_id}")
            return arch_dir
            
        except Exception as e:
            self.logger.error(f"Error preparing archetype {archetype_id}: {str(e)}")
            return None
    
    def get_zones(self) -> List[float]:
        """Get list of unique weather zones."""
        if self.archetypes_df is None:
            self._load_archetypes()
        return sorted(self.archetypes_df['weather_zone'].unique())
    
    def get_archetypes_by_zone(self, zone: float) -> pd.DataFrame:
        """Get archetypes for a specific weather zone."""
        if self.archetypes_df is None:
            self._load_archetypes()
        return self.archetypes_df[self.archetypes_df['weather_zone'] == zone]

    def prepare_archetype_for_scenario(self, 
                                 archetype_id: int, 
                                 scenario_key: str, 
                                 output_dir: Path,
                                 calibration_params: Optional[Dict[str, Dict[int, float]]] = None,
                                 use_stochastic_schedules: bool = True) -> Optional[Path]:
        """
        Prepare an archetype for a future scenario using the hybrid approach.
        
        Args:
            archetype_id: ID of the archetype to prepare
            scenario_key: Future scenario key (e.g., '2035_warm_PV_E')
            output_dir: Directory to store prepared files
            calibration_params: Optional calibration parameters by archetype
            use_stochastic_schedules: Whether to use stochastic schedules
            
        Returns:
            Path to prepared archetype directory if successful, None otherwise
        """
        try:
            # Verify the scenario key is valid
            if not config.future_scenarios.is_future_scenario(scenario_key):
                self.logger.error(f"Invalid future scenario key: {scenario_key}")
                return None
            
            # Access the pre-transformed archetype DataFrame
            # This would be prepared by SimulationManager before calling this method
            if not hasattr(self, 'transformed_archetypes_df'):
                self.logger.warning("No pre-transformed archetypes DataFrame found - using standard approach")
                # Fall back to original method
                original_archetype = self.archetypes_df.iloc[archetype_id].copy()
                
                # Transform archetype according to scenario
                from ..scenario_manager import ScenarioManager
                scenario_manager = ScenarioManager()
                
                # Transform archetype
                transformed_archetype = scenario_manager.transform_archetype(original_archetype, scenario_key)
                
                # NOUVEAU: Vérifier et ajuster les types d'efficacité si c'est une thermopompe
                if pd.notna(transformed_archetype.get('heatPumpType')):
                    # S'assurer que les types d'efficacité sont correctement définis
                    if pd.isna(transformed_archetype.get('heatPumpHeatingEffType')):
                        transformed_archetype['heatPumpHeatingEffType'] = 'COP'
                        
                    if pd.isna(transformed_archetype.get('heatPumpCoolingEffType')):
                        transformed_archetype['heatPumpCoolingEffType'] = True
                        
                    self.logger.debug(f"Archetype {archetype_id} - Types d'efficacité définis: Chauffage={transformed_archetype['heatPumpHeatingEffType']}, Refroidissement={transformed_archetype['heatPumpCoolingEffType']}")
                
                # NOUVEAU: Journaliser les changements clés pour le débogage
                self.logger.debug(f"Archetype {archetype_id} transformé avec approche hybride pour {scenario_key}")
                
                # Vérification d'attributs clés avant/après
                thermal_attributes = ['dominantWallRVal', 'dominantCeilingRVal', 'ach', 'spaceHeatingFuel', 'heatPumpType']
                for attr in thermal_attributes:
                    if attr in original_archetype and attr in transformed_archetype:
                        old_val = original_archetype.get(attr)
                        new_val = transformed_archetype.get(attr)
                        if pd.notna(old_val) and pd.notna(new_val) and old_val != new_val:
                            self.logger.debug(f"  → {attr}: {old_val} → {new_val}")
                
                # Create temporary DataFrame with transformed archetype
                temp_df = self.archetypes_df.copy()
                temp_df.iloc[archetype_id] = transformed_archetype
                
                # Save original DataFrame
                original_df = self.archetypes_df
                
                # Replace with temporary DataFrame
                self.archetypes_df = temp_df
            else:
                # Use the pre-transformed DataFrame
                original_df = self.archetypes_df  # Save current
                self.archetypes_df = self.transformed_archetypes_df  # Use transformed
                
                # NOUVEAU: Vérifier les types d'efficacité dans le DataFrame préalablement transformé
                transformed_archetype = self.transformed_archetypes_df.iloc[archetype_id]
                if pd.notna(transformed_archetype.get('heatPumpType')):
                    # S'assurer que les types d'efficacité sont correctement définis
                    if pd.isna(transformed_archetype.get('heatPumpHeatingEffType')):
                        self.transformed_archetypes_df.loc[archetype_id, 'heatPumpHeatingEffType'] = 'COP'
                        
                    if pd.isna(transformed_archetype.get('heatPumpCoolingEffType')):
                        self.transformed_archetypes_df.loc[archetype_id, 'heatPumpCoolingEffType'] = True
                        
                    self.logger.debug(f"Archetype transformé {archetype_id} - Types d'efficacité vérifiés: Chauffage={self.transformed_archetypes_df.loc[archetype_id, 'heatPumpHeatingEffType']}, Refroidissement={self.transformed_archetypes_df.loc[archetype_id, 'heatPumpCoolingEffType']}")
                
                # NOUVEAU: Journaliser pour vérification
                original_archetype = original_df.iloc[archetype_id]
            
            # Get scenario year
            year = config.future_scenarios.get_scenario_year(scenario_key)
            
            # Prepare archetype using standard method with the already-adapted parameters received by the function
            arch_dir = self.prepare_archetype(
                archetype_id, 
                output_dir, 
                year,
                calibration_params, # Use the parameters received directly
                use_stochastic_schedules=use_stochastic_schedules
            )
            
            # Restore original DataFrame
            self.archetypes_df = original_df
            
            return arch_dir
            
        except Exception as e:
            self.logger.error(f"Error preparing archetype {archetype_id} for scenario {scenario_key}: {str(e)}")
            # Ensure original DataFrame is restored
            if 'original_df' in locals():
                self.archetypes_df = original_df
            return None