"""
AggregationManager révisé pour UBEM Québec.
Stratifie et agrège les résultats de simulation avec une approche scientifiquement rigoureuse.
Cette version élimine le raisonnement circulaire de la calibration en n'utilisant que
des facteurs d'échelle basés sur la représentativité.
"""
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from datetime import datetime

from ..config import config
from ..utils import ensure_dir, save_results
from .simulation_manager import SimulationManager

# Statistiques d'intensité énergétique basées sur RNCan 2021 (kWh/m²/an)
# Utilisées uniquement pour validation/diagnostic, pas pour ajustement des résultats
TARGET_INTENSITIES = {
    'Maisons unifamiliales': 207.2,  # 192.2 PJ / 257.7 M m²
    'Maisons individuelles attenantes': 180.5,  # 27.4 PJ / 42.1 M m²
    'Appartements': 149.6,  # 102.6 PJ / 190.3 M m²
}

# Statistiques de consommation totale par type (TWh)
TARGET_CONSUMPTION = {
    'Maisons unifamiliales': 53.4,  # 192.2 PJ
    'Maisons individuelles attenantes': 7.6,  # 27.4 PJ
    'Appartements': 28.5,  # 102.6 PJ
    'Total': 89.5,  # 322.2 PJ
}

# Surfaces totales selon RNCan (millions de m²)
RNCAN_SURFACES = {
    'Maisons unifamiliales': 257.7,
    'Maisons individuelles attenantes': 42.1,
    'Appartements': 190.3
}

# Répartition des systèmes de chauffage par typologie (RNCan 2021)
HEATING_SYSTEMS_BY_TYPE = {
    'Maisons unifamiliales': {
        'electric': {'percentage': 0.475, 'fuel': ['Electric']},
        'dual_wood_electric': {'percentage': 0.226, 'fuel': ['Electric', 'Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'dual_oil_electric': {'percentage': 0.070, 'fuel': ['Electric', 'Oil']},
        'oil': {'percentage': 0.037, 'fuel': ['Oil']},
        'gas': {'percentage': 0.049, 'fuel': ['Natural gas']},
        'dual_gas_electric': {'percentage': 0.007, 'fuel': ['Electric', 'Natural gas']},
        'wood': {'percentage': 0.136, 'fuel': ['Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'heat_pump': {'percentage': 0.135, 'fuel': ['heat pump']},  
    },
    'Maisons individuelles attenantes': {
        'electric': {'percentage': 0.765, 'fuel': ['Electric']},
        'dual_wood_electric': {'percentage': 0.034, 'fuel': ['Electric', 'Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'dual_oil_electric': {'percentage': 0.053, 'fuel': ['Electric', 'Oil']},
        'oil': {'percentage': 0.036, 'fuel': ['Oil']},
        'gas': {'percentage': 0.054, 'fuel': ['Natural gas']},
        'dual_gas_electric': {'percentage': 0.000, 'fuel': ['Electric', 'Natural gas']},
        'wood': {'percentage': 0.058, 'fuel': ['Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'heat_pump': {'percentage': 0.056, 'fuel': ['heat pump']},
    },
    'Appartements': {
        'electric': {'percentage': 0.849, 'fuel': ['Electric']},
        'dual_wood_electric': {'percentage': 0.012, 'fuel': ['Electric', 'Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'dual_oil_electric': {'percentage': 0.024, 'fuel': ['Electric', 'Oil']},
        'oil': {'percentage': 0.044, 'fuel': ['Oil']},
        'gas': {'percentage': 0.055, 'fuel': ['Natural gas']},
        'dual_gas_electric': {'percentage': 0.006, 'fuel': ['Electric', 'Natural gas']},
        'wood': {'percentage': 0.010, 'fuel': ['Mixed Wood', 'Hardwood', 'Wood Pellets']},
        'heat_pump': {'percentage': 0.010, 'fuel': ['heat pump']}, 
    }
}

# Hiérarchie de similarité entre systèmes de chauffage pour substitution
HEATING_SYSTEM_SIMILARITY = {
    # Systèmes électriques
    'electric': ['heat_pump', 'dual_wood_electric', 'dual_oil_electric', 'dual_gas_electric'],
    'heat_pump': ['electric', 'dual_wood_electric', 'dual_oil_electric', 'dual_gas_electric'],
    
    # Systèmes bivalents avec électricité
    'dual_wood_electric': ['electric', 'heat_pump', 'wood', 'dual_oil_electric', 'dual_gas_electric'],
    'dual_oil_electric': ['electric', 'heat_pump', 'oil', 'dual_wood_electric', 'dual_gas_electric'],
    'dual_gas_electric': ['electric', 'heat_pump', 'gas', 'dual_wood_electric', 'dual_oil_electric'],
    
    # Systèmes fossiles et autres
    'oil': ['dual_oil_electric', 'gas', 'dual_gas_electric'],
    'gas': ['dual_gas_electric', 'oil', 'dual_oil_electric'],
    'wood': ['dual_wood_electric', 'electric', 'heat_pump']
}

class ZoneWeights:
    """Gère les poids d'agrégation par zone, typologie, période et système de chauffage."""
    
    def __init__(self, property_data: pd.DataFrame, simulation_year: int):
        """
        Initialise les poids d'agrégation.
        
        Args:
            property_data: DataFrame des données d'évaluation foncière
            simulation_year: Année de simulation
        """
        self.logger = logging.getLogger(__name__)
        
        # Filtrer les bâtiments construits après l'année de simulation
        self.property_data = property_data[
            property_data['annee_construction'] <= simulation_year
        ].copy()
        
        self.simulation_year = simulation_year
        
        # Périodes alignées avec les statistiques RNCan
        self.construction_periods = {
            'pre1960': lambda x: x < 1960,
            '1960_1980': lambda x: (x >= 1960) & (x <= 1980),
            'post1980': lambda x: (x > 1980) & (x <= simulation_year)
        }
        
        # Systèmes de chauffage par typologie
        self.heating_systems = HEATING_SYSTEMS_BY_TYPE
        
        # Intensités énergétiques cibles (pour validation uniquement)
        self.target_intensities = TARGET_INTENSITIES
        
        # Facteurs de représentativité (initialisation)
        self.representativity_factors = {
            'Maisons unifamiliales': 1.0,
            'Maisons individuelles attenantes': 1.0,
            'Appartements': 1.0
        }
        
        # Référence de surfaces originale pour RNCan (ne pas modifier cette constante)
        self.reference_surfaces = RNCAN_SURFACES.copy()
        
        # Calculer les poids
        self.weights = self._calculate_weights(self.property_data)
        
    def _calculate_weights(self, data: pd.DataFrame) -> Dict:
        """
        Calcule les poids par zone, typologie, période et système de chauffage.
        Ne tient compte que de la représentativité, pas de l'intensité énergétique.
        
        Args:
            data: DataFrame des données d'évaluation foncière
            
        Returns:
            Dictionnaire des poids structurés par zone/typologie/période/système
        """
        weights = {}
        
        # Déterminer le type de bâtiment à partir des données foncières
        data['building_type'] = data.apply(self._determine_building_type, axis=1)
        
        # Grouper par zone
        for zone in data['weather_zone'].unique():
            zone_data = data[data['weather_zone'] == zone]
            weights[zone] = {}
            
            # Grouper par typologie
            for building_type in ['Maisons unifamiliales', 'Maisons individuelles attenantes', 'Appartements']:
                type_data = zone_data[zone_data['building_type'] == building_type]
                weights[zone][building_type] = {}
                
                # Calculer les poids pour chaque période
                for period_name, condition in self.construction_periods.items():
                    period_data = type_data[condition(type_data['annee_construction'])]
                    total_area = period_data['aire_etages'].sum()
                    total_buildings = len(period_data)
                    
                    weights[zone][building_type][period_name] = {}
                    
                    # Calculer les poids pour chaque système de chauffage
                    systems = self.heating_systems[building_type]
                    for system, info in systems.items():
                        weights[zone][building_type][period_name][system] = {
                            'total_area': total_area * info['percentage'],
                            'building_count': int(total_buildings * info['percentage'])
                        }
        
        # Résumé des poids calculés
        total_area = sum(
            sum(
                sum(
                    sum(system['total_area'] for system in period.values())
                    for period in building_type.values()
                )
                for building_type in zone_weights.values()
            )
            for zone_weights in weights.values()
        )
        
        self.logger.info(f"Poids calculés pour {len(weights)} zones avec {total_area:.1f} m² de surface totale")
        
        return weights
    
    def calculate_representativity_factors(self, scenario_params: Dict = None) -> Dict[str, float]:
        """
        Calcule les facteurs de représentativité basés sur les écarts de surface 
        entre RNCan et les données foncières.
        
        Args:
            scenario_params: Paramètres du scénario futur (optionnel)
            
        Returns:
            Dictionnaire des facteurs de représentativité par typologie
        """
        # Calculer les surfaces totales représentées dans le modèle par typologie
        model_surfaces = {}
        
        # Parcourir toutes les zones et calculer la surface totale par typologie
        for building_type in self.reference_surfaces.keys():
            total_area = 0
            for zone in self.weights:
                if building_type in self.weights[zone]:
                    for period in self.weights[zone][building_type]:
                        for system, system_data in self.weights[zone][building_type][period].items():
                            total_area += system_data['total_area']
            
            # Convertir en millions de m²
            model_surfaces[building_type] = total_area / 1_000_000
        
        # Ajuster les surfaces de référence si c'est un scénario futur
        target_surfaces = self.reference_surfaces.copy()
        
        if scenario_params and 'growth' in scenario_params:
            growth = scenario_params['growth']
            
            # Lire les facteurs de croissance du scénario
            total_growth = growth.get('total_growth', 0)
            apartment_growth = growth.get('apartment_growth', 0)
            detached_reduction = growth.get('detached_reduction', 0)
            
            self.logger.info(f"Ajustement des surfaces de référence pour le scénario futur:")
            self.logger.info(f"  → Croissance totale: {total_growth:.2f}")
            self.logger.info(f"  → Croissance appartements: {apartment_growth:.2f}")
            self.logger.info(f"  → Réduction/croissance maisons: {detached_reduction:.2f}")
            
            # Appliquer les facteurs de croissance aux surfaces de référence
            # Note: detached_reduction est positif pour une réduction, négatif pour une croissance
            target_surfaces['Appartements'] *= (1 + apartment_growth)
            
            if detached_reduction > 0:  # Réduction des maisons
                target_surfaces['Maisons unifamiliales'] *= (1 - detached_reduction)
            else:  # Croissance des maisons
                target_surfaces['Maisons unifamiliales'] *= (1 + abs(detached_reduction))
            
            # Pour les maisons attenantes, utiliser un facteur intermédiaire (moyenne pondérée)
            attached_factor = ((1 + apartment_growth) + (1 - detached_reduction if detached_reduction > 0 else 1 + abs(detached_reduction))) / 2
            target_surfaces['Maisons individuelles attenantes'] *= attached_factor
            
            # Log des surfaces ajustées
            self.logger.info(f"Surfaces cibles ajustées (M m²):")
            for btype, surface in target_surfaces.items():
                self.logger.info(f"  → {btype}: {self.reference_surfaces[btype]:.1f} → {surface:.1f}")
            
        # Calculer les facteurs de représentativité en utilisant les surfaces ajustées
        for building_type in self.reference_surfaces.keys():
            # Calculer le facteur de représentativité
            if model_surfaces[building_type] > 0:
                representativity_factor = target_surfaces[building_type] / model_surfaces[building_type]
            else:
                representativity_factor = 1.0
                
            self.representativity_factors[building_type] = representativity_factor
            
        self.logger.info(f"Surfaces modèle (M m²): {model_surfaces}")
        self.logger.info(f"Surfaces cibles (M m²): {target_surfaces}")
        self.logger.info(f"Facteurs de représentativité calculés: {self.representativity_factors}")
        
        return self.representativity_factors
    
    def _determine_building_type(self, row: pd.Series) -> str:
        """
        Détermine le type de bâtiment à partir des données foncières.
        Classification simplifiée basée principalement sur lien physique et nombre de logements.
        
        Args:
            row: Série pandas avec les données d'un bâtiment
            
        Returns:
            Type de bâtiment (Maisons unifamiliales, Maisons individuelles attenantes, Appartements)
        """
        # Valeurs par défaut pour éviter les NaN
        lien_physique = row.get('lien_physique_code', 0) 
        nb_logements = row.get('nombre_logements', 0)
        
        # RÈGLE 1: Basée sur le lien physique
        if lien_physique == 1:  # Détaché
            if nb_logements <= 2:
                return "Maisons unifamiliales"
            else:
                return "Appartements"
                
        elif lien_physique in [2, 3, 4]:  # Jumelé ou en rangée
            if nb_logements <= 2:
                return "Maisons individuelles attenantes"
            else:
                return "Appartements"
                
        elif lien_physique == 5:  # Intégré
            return "Appartements"
            
        # RÈGLE 2: Si lien physique manquant, utiliser nombre de logements
        elif nb_logements == 1:
            return "Maisons unifamiliales"
        elif nb_logements == 2:
            return "Maisons individuelles attenantes"
        elif nb_logements > 2:
            return "Appartements"
        
        # RÈGLE 3: Cas indéterminés - utiliser genre de construction si disponible
        else:
            genre = row.get('genre_construction_code', 0)
            if genre in [1, 3]:  # Plain-pied ou Unimodulaire
                return "Maisons unifamiliales"
            elif genre in [4, 5]:  # Étage mansardé ou Étages entiers
                return "Appartements"
            else:
                return "Maisons unifamiliales"  # Valeur par défaut la plus probable
    
    def get_period(self, year: float) -> str:
        """
        Détermine la période de construction pour une année donnée.
        
        Args:
            year: Année de construction
            
        Returns:
            Nom de la période (pre1983, 1984_2000, post2001)
        """
        for period, condition in self.construction_periods.items():
            if condition(pd.Series([year])).iloc[0]:
                return period
        return 'post2001'  # Valeur par défaut
    
    def match_heating_system(self, archetype: pd.Series, building_type: str) -> Optional[str]:
        """
        Détermine le type de système de chauffage d'un archétype.
        
        Args:
            archetype: Série pandas contenant les données de l'archétype
            building_type: Type de bâtiment
            
        Returns:
            Nom du système de chauffage ou None si pas de correspondance
        """
        # NOUVEAU: Vérifier d'abord si c'est une thermopompe
        heat_pump_type = archetype.get('heatPumpType')
        if pd.notna(heat_pump_type):
            return 'heat_pump'
        
        # Logique existante pour les autres systèmes
        primary_type = archetype.get('spaceHeatingType')
        primary_fuel = archetype.get('spaceHeatingFuel')
        secondary_fuel = archetype.get('supplHeatingFuel')
        
        # Obtenir les systèmes pour ce type de bâtiment
        systems = self.heating_systems.get(building_type, self.heating_systems['Maisons unifamiliales'])
        
        # Parcourir les systèmes de chauffage
        for system_name, system_info in systems.items():
            # Vérifier le système principal
            if primary_fuel in system_info['fuel']:
                # Si c'est un système bivalent, vérifier le système secondaire
                if system_name.startswith('dual_') and secondary_fuel:
                    secondary_fuels = [f for f in system_info['fuel'] if f != primary_fuel]
                    if secondary_fuel in secondary_fuels:
                        return system_name
                elif not system_name.startswith('dual_') and not secondary_fuel:
                    # Système simple
                    return system_name
        
        # Si aucun match parfait, retourner le système électrique par défaut
        if primary_fuel == 'Electric':
            return 'electric'
        
        # Sinon choix par défaut selon le combustible principal
        if primary_fuel in ['Mixed Wood', 'Hardwood', 'Wood Pellets']:
            return 'wood'
        elif primary_fuel == 'Oil':
            return 'oil'
        elif primary_fuel == 'Natural gas':
            return 'gas'
        
        # Défaut si tout échoue
        return 'electric'
    
    def determine_building_type(self, archetype: pd.Series) -> str:
        """
        Détermine le type de bâtiment pour un archétype.
        
        Args:
            archetype: Série pandas avec les données d'un archétype
            
        Returns:
            Type de bâtiment (Maisons unifamiliales, Maisons individuelles attenantes, Appartements)
        """
        house_type = archetype.get('houseType', '')
        sub_type = archetype.get('houseSubType', '')
        
        # Maisons unifamiliales
        if house_type == 'House' and sub_type in ['Single Detached', 'Mobile Home']:
            return 'Maisons unifamiliales'
        
        # Maisons individuelles attenantes
        if house_type == 'House' and any(term in sub_type for term in ['Semi', 'Row', 'Duplex', 'Attached']):
            return 'Maisons individuelles attenantes'
        
        # Appartements (tout le reste)
        if 'Multi-unit' in house_type or sub_type in ['Apartment', 'Apartment Row'] or 'Triplex' in sub_type:
            return 'Appartements'
        
        # Par défaut
        return 'Maisons unifamiliales'

    def get_similar_systems(self, target_system: str, building_type: str) -> List[str]:
        """
        Renvoie une liste ordonnée de systèmes de chauffage similaires pour un système donné,
        en tenant compte de la typologie du bâtiment et de la hiérarchie de similarité.
        
        Args:
            target_system: Système de chauffage cible
            building_type: Type de bâtiment
            
        Returns:
            Liste ordonnée de systèmes similaires (excluant le système cible)
        """
        # Liste des systèmes similaires selon la hiérarchie définie
        similar_systems = HEATING_SYSTEM_SIMILARITY.get(target_system, [])
        
        # Filtrer pour ne garder que les systèmes définis pour ce type de bâtiment
        available_systems = list(self.heating_systems[building_type].keys())
        similar_systems = [s for s in similar_systems if s in available_systems]
        
        # Logger la liste des systèmes similaires
        self.logger.debug(
            f"Systèmes similaires à '{target_system}' pour '{building_type}': {similar_systems}"
        )
        
        return similar_systems

    def adjust_heating_systems_for_scenario(self, scenario_key: str) -> None:
        """
        Ajuste les distributions des systèmes selon les paramètres du scénario.
        Version simplifiée qui reflète directement les paramètres du scénario.
        
        Args:
            scenario_key: Clé du scénario futur
        """
        # Import ScenarioManager
        from ..scenario_manager import ScenarioManager
        scenario_manager = ScenarioManager()
        
        # Get scenario parameters
        scenario = scenario_manager.get_scenario_params(scenario_key)
        if not scenario:
            self.logger.warning(f"Scénario {scenario_key} non trouvé, les distributions des systèmes restent inchangées")
            return
        
        # Get electrification parameters
        electrification = scenario.get('electrification', {})
        if not electrification:
            self.logger.warning(f"Aucun paramètre d'électrification dans le scénario {scenario_key}")
            return
        
        # Récupérer directement les valeurs cibles du scénario
        target_electric_pct = electrification.get('electric_heating_pct', 0.0)
        target_heat_pump_pct = electrification.get('heat_pump_pct', 0.0)
        target_gas_pct = electrification.get('gas_heating_pct', 0.0)
        target_oil_pct = electrification.get('oil_heating_pct', 0.0)
        
        self.logger.info(f"Application des cibles d'électrification pour {scenario_key}:")
        self.logger.info(f"  → Électrique: {target_electric_pct:.2f}")
        self.logger.info(f"  → Thermopompes: {target_heat_pump_pct:.2f}")
        self.logger.info(f"  → Gaz: {target_gas_pct:.2f}")
        self.logger.info(f"  → Mazout: {target_oil_pct:.2f}")
        
        # Créer une copie profonde des distributions actuelles
        import copy
        adjusted_systems = copy.deepcopy(self.heating_systems)
        
        # Pour chaque typologie de bâtiment
        for building_type in adjusted_systems:
            # Créer une nouvelle distribution
            new_distribution = {}
            
            # Appliquer les pourcentages du scénario directement
            if 'electric' in adjusted_systems[building_type]:
                new_distribution['electric'] = {
                    'percentage': max(0.0, target_electric_pct - target_heat_pump_pct),
                    'fuel': adjusted_systems[building_type]['electric']['fuel']
                }
                
            if 'heat_pump' in adjusted_systems[building_type]:
                new_distribution['heat_pump'] = {
                    'percentage': target_heat_pump_pct,
                    'fuel': adjusted_systems[building_type]['heat_pump']['fuel']
                }
                
            if 'gas' in adjusted_systems[building_type]:
                new_distribution['gas'] = {
                    'percentage': target_gas_pct,
                    'fuel': adjusted_systems[building_type]['gas']['fuel']
                }
                
            if 'oil' in adjusted_systems[building_type]:
                new_distribution['oil'] = {
                    'percentage': target_oil_pct,
                    'fuel': adjusted_systems[building_type]['oil']['fuel']
                }
            
            # Calculer le pourcentage restant pour les autres systèmes
            used_pct = (target_electric_pct - target_heat_pump_pct) + target_heat_pump_pct + target_gas_pct + target_oil_pct
            other_pct = max(0.0, 1.0 - used_pct)
            
            # Distribuer le pourcentage restant entre les autres systèmes
            other_systems = [s for s in adjusted_systems[building_type] 
                            if s not in ['electric', 'heat_pump', 'gas', 'oil']]
            
            if other_systems and other_pct > 0:
                # Calculer la somme des pourcentages actuels des autres systèmes
                current_other_sum = sum(adjusted_systems[building_type][s]['percentage'] for s in other_systems)
                
                # Si somme > 0, répartir proportionnellement
                if current_other_sum > 0:
                    for system in other_systems:
                        current_pct = adjusted_systems[building_type][system]['percentage']
                        new_pct = (current_pct / current_other_sum) * other_pct
                        new_distribution[system] = {
                            'percentage': new_pct,
                            'fuel': adjusted_systems[building_type][system]['fuel']
                        }
                else:
                    # Sinon, répartir uniformément
                    equal_share = other_pct / len(other_systems)
                    for system in other_systems:
                        new_distribution[system] = {
                            'percentage': equal_share,
                            'fuel': adjusted_systems[building_type][system]['fuel']
                        }
            
            # Remplacer l'ancienne distribution par la nouvelle
            adjusted_systems[building_type] = new_distribution
            
            # Vérification finale: s'assurer que la somme est bien 1.0
            total = sum(adjusted_systems[building_type][s]['percentage'] for s in adjusted_systems[building_type])
            if abs(total - 1.0) > 0.001:
                self.logger.warning(
                    f"La somme des pourcentages pour {building_type} est {total}, normalisation appliquée"
                )
                # Normaliser
                for system in adjusted_systems[building_type]:
                    adjusted_systems[building_type][system]['percentage'] /= total
        
        # Remplacer les distributions
        self.heating_systems = adjusted_systems
        
        # Log des nouvelles distributions
        for building_type in self.heating_systems:
            distribution = {s: round(self.heating_systems[building_type][s]['percentage'] * 100, 1) 
                          for s in self.heating_systems[building_type]}
            self.logger.debug(f"Nouvelle distribution pour {building_type}: {distribution}%")

class AggregationManager:
    """
    Gère l'agrégation des résultats de simulation à différentes échelles spatiales.
    Cette version révisée utilise des facteurs d'échelle basés sur la représentativité,
    sans imposer artificiellement les intensités énergétiques cibles.
    """
    
    def __init__(self, simulation_manager: SimulationManager):
        """
        Initialise l'AggregationManager.
        
        Args:
            simulation_manager: Instance de SimulationManager pour accès aux archétypes
        """
        self.logger = logging.getLogger(__name__)
        self.simulation_manager = simulation_manager
        self.property_data = None
        self.zone_weights = None
        
        # Mémoire cache d'intensités calculées pour diagnostic
        self.intensity_cache = {}
    
    def _load_property_data(self, year: int) -> None:
        """
        Charge et traite les données d'évaluation foncière.
        
        Args:
            year: Année des données à charger
        """
        try:
            # Charger les données foncières (toujours utiliser 2024)
            property_file = config.paths['input'].get_evaluation_file(year)
            self.property_data = pd.read_csv(property_file)
            
            # Filtrer les enregistrements invalides
            self.property_data = self.property_data[
                (pd.notna(self.property_data['weather_zone'])) &
                (pd.notna(self.property_data['aire_etages'])) &
                (self.property_data['aire_etages'] > 0)
            ]
            
            # Conversion et statistiques
            self.property_data['weather_zone'] = self.property_data['weather_zone'].astype(int)
            
            # Statistiques par zone
            zone_stats = self.property_data.groupby('weather_zone').agg({
                'aire_etages': ['sum', 'count']
            }).reset_index()
            
            zone_stats.columns = ['zone', 'total_floor_area', 'building_count']
            total_area = zone_stats['total_floor_area'].sum()
            zone_stats['weight'] = zone_stats['total_floor_area'] / total_area
            
            self.logger.info(
                f"Données foncières chargées: {len(self.property_data):,} bâtiments, "
                f"{total_area/1e6:.1f} millions m² de surface totale"
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données foncières: {str(e)}")
            raise
    
    def _load_mrc_mapping(self) -> Dict:
        """
        Charge le mapping entre zones météo et MRCs depuis le fichier SHP.
        
        Returns:
            Dictionnaire {zone_meteo: {mrc_ids: [...], mrc_names: [...]}}
        """
        try:
            import geopandas as gpd
            
            # Charger le fichier SHP
            zones_shp_path = config.paths['input'].ZONES_SHAPEFILE
            gdf = gpd.read_file(zones_shp_path)
            
            # Créer le mapping
            mrc_mapping = {}
            for _, row in gdf.iterrows():
                zone_meteo = int(row['weather_zo'])
                mrc_id = row['CDUID']
                mrc_name = row['CDNAME']
                
                if zone_meteo not in mrc_mapping:
                    mrc_mapping[zone_meteo] = {'mrc_ids': [], 'mrc_names': []}
                    
                mrc_mapping[zone_meteo]['mrc_ids'].append(mrc_id)
                mrc_mapping[zone_meteo]['mrc_names'].append(mrc_name)
            
            self.logger.info(f"Mapping MRC chargé avec {len(mrc_mapping)} zones")
            return mrc_mapping
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du mapping MRC: {str(e)}")
            return {}
    
    def _get_archetype_type(self, archetype_id: int) -> str:
        """
        Détermine le type de bâtiment pour un archétype.
        
        Args:
            archetype_id: ID de l'archétype
            
        Returns:
            Type de bâtiment (Maisons unifamiliales, Maisons individuelles attenantes, Appartements)
        """
        archetype = self.simulation_manager.archetype_manager.archetypes_df.iloc[archetype_id]
        return self.zone_weights.determine_building_type(archetype)
    
    def _calculate_energy_intensity(self, results: pd.DataFrame, area: float) -> float:
        """
        Calcule l'intensité énergétique pour un bâtiment.
        
        Args:
            results: DataFrame de résultats horaires
            area: Surface du bâtiment en m²
            
        Returns:
            Intensité énergétique en kWh/m²/an
        """
        # S'assurer que la division par zéro est évitée
        if area <= 0:
            return 0
            
        # Consommation annuelle
        annual_consumption = results['Fuel Use: Electricity: Total'].sum()
        
        # Intensité (kWh/m²/an)
        return annual_consumption / area
    
    def _aggregate_zone(self, zone: float, 
                      simulation_results: Dict[int, pd.DataFrame],
                      apply_representativity: bool = True,
                      transformed_archetypes_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Agrège les résultats de simulation pour une zone spécifique.
        Version améliorée qui utilise des archétypes similaires lorsqu'une combinaison exacte
        n'existe pas dans la sélection d'archétypes.
        
        Args:
            zone: Zone météo à agréger
            simulation_results: Dictionnaire de résultats par ID d'archétype
            apply_representativity: Si True, applique les facteurs de représentativité
            transformed_archetypes_df: DataFrame des archétypes transformés (pour scénarios futurs)
            
        Returns:
            DataFrame avec les résultats agrégés pour la zone
        """
        try:
            zone = int(zone)  # Conversion en entier pour comparaison
            
            # MODIFICATION: Utiliser les archétypes transformés si disponibles
            archetypes_df = transformed_archetypes_df if transformed_archetypes_df is not None else self.simulation_manager.archetype_manager.archetypes_df
            
            zone_results = pd.DataFrame()
            
            # Compteurs pour les statistiques de remplacement
            total_combinations = 0
            missing_combinations = 0
            proxy_combinations = 0
            still_missing_combinations = 0
            
            # Un seul log au début de l'agrégation
            self.logger.info(f"Début agrégation zone {zone}")
            
            # Statistiques pour le rapport
            stats = {
                'zone': zone,
                'typologies': {},
                'scale_factors': [],
                'representativity': {
                    'applied': apply_representativity,
                    'factors': self.zone_weights.representativity_factors.copy() if apply_representativity else None
                },
                'missing_combinations': [],  # Liste pour stocker les combinaisons manquantes
                'proxy_combinations': []     # Liste pour stocker les combinaisons avec proxy
            }
            
            # Pour chaque typologie de bâtiment
            for building_type in ['Maisons unifamiliales', 'Maisons individuelles attenantes', 'Appartements']:
                stats['typologies'][building_type] = {'periods': {}}
                
                # Pour chaque période de construction
                for period, condition in self.zone_weights.construction_periods.items():
                    stats['typologies'][building_type]['periods'][period] = {'systems': {}}
                    
                    # Sélectionner les archétypes pour cette zone/typologie/période
                    period_archetypes = []
                    for idx, results in simulation_results.items():
                        if int(results.attrs['weather_zone']) == zone:
                            archetype = archetypes_df.iloc[idx]
                            arch_type = self.zone_weights.determine_building_type(archetype)
                            
                            if arch_type == building_type and condition(archetype['vintageExact']):
                                period_archetypes.append((idx, results, archetype))
                    
                    if not period_archetypes:
                        self.logger.debug(f"Aucun archétype pour zone {zone}, type {building_type}, période {period}")
                        # Pas de recherche alternative ici - on garde l'approche par système
                        continue
                    
                    # Pour chaque système de chauffage
                    for system, info in self.zone_weights.heating_systems[building_type].items():
                        total_combinations += 1
                        combination_key = f"zone{zone}_{building_type}_{period}_{system}"
                        
                        # Filtrer les archétypes par système de chauffage
                        system_archetypes = []
                        for idx, results, archetype in period_archetypes:
                            matched_system = self.zone_weights.match_heating_system(archetype, building_type)
                            if matched_system == system:
                                system_archetypes.append((idx, results, archetype))
                        
                        # Si aucun archétype, chercher des alternatives
                        if not system_archetypes:
                            missing_combinations += 1
                            stats['missing_combinations'].append(combination_key)
                            
                            # Log minimal pour le debug
                            self.logger.debug(f"Recherche alternatives: zone {zone}, {building_type}, {period}, {system}")
                            
                            # NIVEAU 1: Chercher des archétypes de la même combinaison mais d'une autre zone
                            for alt_zone_idx, alt_results in simulation_results.items():
                                alt_archetype = archetypes_df.iloc[alt_zone_idx]
                                alt_type = self.zone_weights.determine_building_type(alt_archetype)
                                alt_system = self.zone_weights.match_heating_system(alt_archetype, building_type)
                                
                                # Si même type/période/système mais zone différente
                                if (alt_type == building_type and 
                                    condition(alt_archetype['vintageExact']) and 
                                    alt_system == system):
                                    # Marquer comme proxy pour les statistiques
                                    alt_results = alt_results.copy()
                                    alt_results.attrs['is_proxy'] = True
                                    alt_results.attrs['original_zone'] = int(alt_results.attrs.get('weather_zone', 0))
                                    alt_results.attrs['similarity_level'] = 1
                                    alt_results.attrs['original_system'] = alt_system
                                    system_archetypes.append((alt_zone_idx, alt_results, alt_archetype))
                            
                            # NIVEAU 2: Si toujours rien, chercher des archétypes avec systèmes similaires de la même zone
                            if not system_archetypes:
                                similar_systems = self.zone_weights.get_similar_systems(system, building_type)
                                
                                for similar_system in similar_systems:
                                    # Chercher des archétypes de cette zone avec un système similaire
                                    for alt_idx, alt_results in simulation_results.items():
                                        if int(alt_results.attrs['weather_zone']) != zone:
                                            continue
                                            
                                        alt_archetype = archetypes_df.iloc[alt_idx]
                                        alt_type = self.zone_weights.determine_building_type(alt_archetype)
                                        alt_system = self.zone_weights.match_heating_system(alt_archetype, building_type)
                                        
                                        # Même zone, même type, même période, mais système similaire
                                        if (alt_type == building_type and 
                                            condition(alt_archetype['vintageExact']) and 
                                            alt_system == similar_system):
                                            
                                            alt_results = alt_results.copy()
                                            alt_results.attrs['is_proxy'] = True
                                            alt_results.attrs['similarity_level'] = 2
                                            alt_results.attrs['original_system'] = alt_system
                                            alt_results.attrs['target_system'] = system
                                            system_archetypes.append((alt_idx, alt_results, alt_archetype))
                                    
                                    # Si on a trouvé avec ce système similaire, on s'arrête là
                                    if system_archetypes:
                                        break
                            
                            # NIVEAU 3: Si toujours rien, chercher des archétypes avec systèmes similaires d'autres zones
                            if not system_archetypes:
                                for similar_system in similar_systems:
                                    # Pour chaque système similaire, chercher dans toutes les zones
                                    for alt_idx, alt_results in simulation_results.items():
                                        alt_archetype = archetypes_df.iloc[alt_idx]
                                        alt_type = self.zone_weights.determine_building_type(alt_archetype)
                                        alt_system = self.zone_weights.match_heating_system(alt_archetype, building_type)
                                        
                                        # Même type, même période, mais zone différente et système similaire
                                        if (alt_type == building_type and 
                                            condition(alt_archetype['vintageExact']) and 
                                            alt_system == similar_system):
                                            
                                            alt_results = alt_results.copy()
                                            alt_results.attrs['is_proxy'] = True
                                            alt_results.attrs['original_zone'] = int(alt_results.attrs.get('weather_zone', 0))
                                            alt_results.attrs['similarity_level'] = 3
                                            alt_results.attrs['original_system'] = alt_system
                                            alt_results.attrs['target_system'] = system
                                            system_archetypes.append((alt_idx, alt_results, alt_archetype))
                                    
                                    # Si on a trouvé avec ce système similaire, on s'arrête là
                                    if system_archetypes:
                                        break
                            
                            if system_archetypes:
                                proxy_combinations += 1
                                stats['proxy_combinations'].append(combination_key)
                                
                                # Niveau de similarité pour le debug
                                similarity_levels = set([r.attrs.get('similarity_level', '?') for _, r, _ in system_archetypes])
                                min_level = min(similarity_levels) if similarity_levels else '?'
                                self.logger.debug(f"Remplacement trouvé (niveau {min_level}): zone {zone}, {building_type}, {period}, {system}")
                            else:
                                still_missing_combinations += 1
                                self.logger.warning(f"Aucun remplacement: zone {zone}, {building_type}, {period}, {system}")
                        
                        if not system_archetypes:
                            continue
                        
                        # Calculer surface totale des archétypes pour ce système
                        arch_total_area = sum(
                            float(archetype['totFloorArea'])
                            for _, _, archetype in system_archetypes
                        )
                        
                        # Calculer consommation totale des archétypes
                        arch_results = pd.DataFrame()
                        for _, results, _ in system_archetypes:
                            if arch_results.empty:
                                arch_results = results.copy()
                            else:
                                arch_results = arch_results.add(results, fill_value=0)
                        
                        # Calcul de l'intensité énergétique de l'archétype (pour diagnostic)
                        arch_consumption = arch_results['Fuel Use: Electricity: Total'].sum()
                        arch_intensity = self._calculate_energy_intensity(arch_results, arch_total_area)
                        
                        # Surface totale réelle pour ce système
                        real_total_area = (
                            self.zone_weights.weights[zone][building_type][period][system]['total_area']
                        )
                        
                        # Application du facteur d'échelle basé sur la représentativité
                        area_factor = real_total_area / arch_total_area if arch_total_area > 0 else 0
                        
                        # Application du facteur de représentativité si demandé
                        if apply_representativity and building_type in self.zone_weights.representativity_factors:
                            representativity_factor = self.zone_weights.representativity_factors[building_type]
                            area_factor *= representativity_factor
                            
                            self.logger.debug(
                                f"Zone {zone}, {building_type}: Facteur de représentativité {representativity_factor:.2f} appliqué"
                            )
                        
                        # Application du facteur
                        if area_factor > 0:
                            system_results = arch_results * area_factor
                            
                            # Ajouter à l'agrégation de la zone
                            if zone_results.empty:
                                zone_results = system_results
                            else:
                                zone_results = zone_results.add(system_results, fill_value=0)
                            
                            # Enregistrer les statistiques pour le rapport
                            stats['scale_factors'].append(area_factor)
                            
                            # Détecter si des proxys ont été utilisés
                            used_proxy = any(hasattr(r, 'attrs') and r.attrs.get('is_proxy', False) for _, r, _ in system_archetypes)
                            
                            stats['typologies'][building_type]['periods'][period]['systems'][system] = {
                                'arch_area': arch_total_area,
                                'real_area': real_total_area,
                                'arch_consumption': arch_consumption,
                                'arch_intensity': arch_intensity,
                                'area_factor': area_factor,
                                'used_proxy': used_proxy
                            }
                
                # Statistiques globales pour la zone
                if not zone_results.empty:
                    avg_area_factor = np.mean(stats['scale_factors']) if stats['scale_factors'] else 0
                    
                    # Résumé des remplacements pour cette zone
                    self.logger.info(
                        f"RÉSUMÉ ZONE {zone}:\n"
                        f"  Total combinaisons: {total_combinations}\n"
                        f"  Combinaisons manquantes: {missing_combinations} ({missing_combinations/total_combinations*100:.1f}%)\n"
                        f"  Remplacements trouvés: {proxy_combinations} ({proxy_combinations/missing_combinations*100:.1f}% des manquants)\n"
                        f"  Toujours manquants: {still_missing_combinations} ({still_missing_combinations/missing_combinations*100:.1f}% des manquants)\n"
                        f"  Facteur d'échelle moyen: {avg_area_factor:.2f}"
                    )
                    
                    # Stocker les statistiques de remplacement dans l'objet zone_results
                    stats['replacement_stats'] = {
                        'total_combinations': total_combinations,
                        'missing_combinations': missing_combinations,
                        'proxy_combinations': proxy_combinations,
                        'still_missing_combinations': still_missing_combinations,
                        'proxy_success_rate': proxy_combinations/missing_combinations if missing_combinations > 0 else 0
                    }
                    
                    # Stocker les statistiques dans l'objet zone_results
                    zone_results.attrs['aggregation_stats'] = stats
                    return zone_results
                else:
                    self.logger.warning(f"Aucun résultat agrégé pour zone {zone}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'agrégation de la zone {zone}: {str(e)}")
            return None
    
    def _aggregate_provincial(self, zonal_results: Dict[float, pd.DataFrame]) -> pd.DataFrame:
        """
        Agrège les résultats zonaux au niveau provincial.
        
        Args:
            zonal_results: Dictionnaire des résultats par zone {zone_id: dataframe}
            
        Returns:
            DataFrame des résultats agrégés pour la province
        """
        if not zonal_results:
            self.logger.warning("Aucun résultat zonal à agréger au niveau provincial")
            return None
            
        # Initialiser avec la première zone
        provincial_results = None
        
        # Ajouter chaque zone
        for zone, zone_results in zonal_results.items():
            if provincial_results is None:
                provincial_results = zone_results.copy()
            else:
                provincial_results = provincial_results.add(zone_results, fill_value=0)
                
            # Log pour suivi
            self.logger.info(f"Zone {zone}: consommation électrique {zone_results['Fuel Use: Electricity: Total'].sum()/1e9:.2f} TWh ajoutée au total provincial")
        
        return provincial_results
    
    def _aggregate_by_mrc(self, 
                   zonal_results: Dict[float, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Agrège les résultats zonaux au niveau des MRCs.
        
        Args:
            zonal_results: Résultats par zone météo
        
        Returns:
            Dictionnaire des résultats par MRC {mrc_id: dataframe}
        """
        try:
            # Charger le mapping zone -> MRC
            mrc_mapping = self._load_mrc_mapping()
            if not mrc_mapping:
                self.logger.error("Aucun mapping MRC trouvé")
                return {}
            
            # Calculer les poids des MRCs dans chaque zone
            mrc_weights = {}
            
            for zone in zonal_results.keys():
                zone_int = int(zone)
                if zone_int not in mrc_mapping:
                    self.logger.warning(f"Zone {zone_int} non trouvée dans le mapping MRC")
                    continue
                    
                # MRCs dans cette zone
                mrc_ids = mrc_mapping[zone_int]['mrc_ids']
                
                # Pour chaque MRC dans cette zone, calculer le poids
                zone_data = self.property_data[self.property_data['weather_zone'] == zone_int]
                zone_total_area = zone_data['aire_etages'].sum()
                
                if zone_int not in mrc_weights:
                    mrc_weights[zone_int] = {}
                
                for mrc_id in mrc_ids:
                    # Extraire le code_mrc (2 derniers chiffres)
                    try:
                        code_mrc = int(mrc_id[-2:])
                    except ValueError:
                        continue
                    
                    # Filtrer les données foncières pour cette MRC
                    mrc_data = self.property_data[self.property_data['code_mrc'] == code_mrc]
                    if len(mrc_data) == 0:
                        continue
                    
                    # Calculer la surface totale des bâtiments dans cette MRC
                    mrc_area = mrc_data['aire_etages'].sum()
                    
                    # Calculer et stocker le poids
                    if zone_total_area > 0 and mrc_area > 0:
                        mrc_weights[zone_int][mrc_id] = mrc_area / zone_total_area
            
            # Appliquer les poids pour obtenir les résultats par MRC
            mrc_results = {}
            
            for zone, weights in mrc_weights.items():
                if float(zone) not in zonal_results:
                    continue
                    
                zone_results = zonal_results[float(zone)]
                
                for mrc_id, weight in weights.items():
                    # Appliquer le poids aux résultats de la zone
                    if mrc_id not in mrc_results:
                        mrc_results[mrc_id] = zone_results.copy() * weight
                    else:
                        # Cas peu probable où une MRC serait dans plusieurs zones
                        mrc_results[mrc_id] = mrc_results[mrc_id].add(zone_results * weight, fill_value=0)
            
            self.logger.info(f"Résultats agrégés pour {len(mrc_results)} MRCs")
            return mrc_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'agrégation par MRC: {str(e)}")
            return {}
    
    def validate_results(self, provincial_results: pd.DataFrame, year: int) -> Dict:
        """
        Valide les résultats de simulation agrégés en comparant aux statistiques connues.
        Cette version sert uniquement de diagnostic, sans modifier les résultats.
        
        Args:
            provincial_results: Résultats provinciaux agrégés
            year: Année de simulation
            
        Returns:
            Métriques de validation
        """
        validation = {
            'year': year,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. Vérifier la consommation totale
            total_consumption = provincial_results['Fuel Use: Electricity: Total'].sum() / 1e9  # TWh
            expected_consumption = TARGET_CONSUMPTION['Total']
            ratio = total_consumption / expected_consumption
            
            validation['total_consumption'] = {
                'simulated': float(total_consumption),
                'expected': expected_consumption,
                'ratio': float(ratio)
            }
            
            # Détection d'écarts importants (uniquement pour diagnostic, pas de correction)
            if 0.225 <= ratio <= 0.275:  # ~0.25
                self.logger.warning(f"Écart important détecté: ratio = {ratio:.3f} (sous-estimation ~4x)")
                validation['error_detected'] = "Possible sous-estimation (facteur ~4x)"
                    
            elif 3.6 <= ratio <= 4.4:  # ~4.0
                self.logger.warning(f"Écart important détecté: ratio = {ratio:.3f} (surestimation ~4x)")
                validation['error_detected'] = "Possible surestimation (facteur ~4x)"
            
            # 2. Vérifier la répartition des usages finaux
            end_uses = {
                'Heating': 'End Use: Electricity: Heating',
                'Water Heating': 'End Use: Electricity: Hot Water',
                'Appliances': 'End Use: Electricity: Plug Loads',
                'Lighting': 'End Use: Electricity: Lighting Interior',
                'Cooling': 'End Use: Electricity: Cooling'
            }
            
            end_use_validation = {}
            for use_name, column in end_uses.items():
                if column in provincial_results.columns:
                    simulated = provincial_results[column].sum() / 1e9  # TWh
                    # Rapport statistique utilise énergie pour le chauffage - adapter selon vos besoins
                    simulated_pct = simulated / total_consumption * 100
                    
                    end_use_validation[use_name] = {
                        'simulated': float(simulated),
                        'simulated_pct': float(simulated_pct)
                    }
            
            validation['end_uses'] = end_use_validation
            
            # 3. Log de l'évaluation finale
            validation_status = "PASS" if 0.8 <= ratio <= 1.2 else "WARNING" if 0.7 <= ratio <= 1.3 else "FAIL"
            validation['status'] = validation_status
            
            self.logger.info(
                f"Validation des résultats: {validation_status}\n"
                f"  Consommation: {total_consumption:.1f} TWh vs {expected_consumption:.1f} TWh attendus (ratio = {ratio:.2f})"
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des résultats: {str(e)}")
            validation['status'] = "ERROR"
            validation['error'] = str(e)
            return validation
    
    def _load_property_data_for_scenario(self, year: int, scenario_key: str) -> None:
        """
        Load and transform property data for a future scenario.
        
        Args:
            year: Base year for property data
            scenario_key: Future scenario key
        """
        try:
            # Load property data as usual
            self._load_property_data(year)
            
            # Now transform according to scenario
            if config.future_scenarios.is_future_scenario(scenario_key):
                from ..scenario_manager import ScenarioManager
                scenario_manager = ScenarioManager()
                
                # NOUVEAU: Sauvegarder les données originales pour vérification
                original_data = self.property_data.copy()
                
                # Transform property data
                self.property_data = scenario_manager.transform_property_data(
                    self.property_data, scenario_key
                )
                
                # NOUVEAU: Vérifier les transformations
                if not self.property_data.equals(original_data):
                    # Comptage par type de bâtiment avant/après
                    original_counts = original_data.groupby('building_type').size() if 'building_type' in original_data.columns else pd.Series({'Non classé': len(original_data)})
                    
                    # Déterminer le type de bâtiment pour les données transformées
                    if 'building_type' not in self.property_data.columns:
                        self.property_data['building_type'] = self.property_data.apply(self._determine_building_type, axis=1)
                        
                    transformed_counts = self.property_data.groupby('building_type').size()
                    
                    # Vérifier les changements
                    self.logger.info(f"Transformation des données foncières pour {scenario_key}:")
                    self.logger.info(f"  → Nombre de bâtiments: {len(original_data)} → {len(self.property_data)}")
                    
                    # Afficher les changements par type
                    for btype in set(list(original_counts.index) + list(transformed_counts.index)):
                        orig = original_counts.get(btype, 0)
                        trans = transformed_counts.get(btype, 0)
                        diff = trans - orig
                        diff_pct = diff / orig * 100 if orig > 0 else float('inf')
                        
                        self.logger.debug(f"  → {btype}: {orig} → {trans} ({diff_pct:+.1f}%)")
                else:
                    self.logger.warning(f"Aucune modification des données foncières détectée pour {scenario_key}")
                
                # Initialize zone weights with transformed data
                self.zone_weights = ZoneWeights(self.property_data, year)
                
                # AJOUT: Ajuster les distributions des systèmes de chauffage selon le scénario
                self.zone_weights.adjust_heating_systems_for_scenario(scenario_key)
                
                # Calculate representativity factors
                if hasattr(self.zone_weights, 'calculate_representativity_factors'):
                    self.zone_weights.calculate_representativity_factors()
                    
                self.logger.info(f"Property data and heating system distributions transformed for scenario {scenario_key}")
                
        except Exception as e:
            self.logger.error(f"Error loading property data for scenario {scenario_key}: {str(e)}")
            raise
            
    def aggregate_results(self, year: int, simulation_results: Dict[Any, Any],
                  scenario: str = 'baseline',
                  output_dir: Optional[Path] = None,
                  apply_representativity: bool = True,
                  skip_mrc_aggregation: bool = False) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Aggregate simulation results at provincial, zonal and MRC levels.
        
        Args:
            year: Simulation year
            simulation_results: Dictionary of results or structured results with transformed archetypes
            scenario: Scenario name or future scenario key
            output_dir: Optional output directory
            apply_representativity: Whether to apply representativity factors
            skip_mrc_aggregation: If True, skip MRC aggregation (useful during calibration)
            
        Returns:
            Tuple of (provincial results, zonal results, MRC results)
        """
        try:
            # MODIFICATION: Extraire les archétypes transformés si disponibles
            transformed_archetypes_df = None
            if isinstance(simulation_results, dict) and 'results' in simulation_results:
                # Format structuré avec archétypes transformés
                transformed_archetypes_df = simulation_results.get('transformed_archetypes')
                is_future_scenario = simulation_results.get('is_future_scenario', False)
                simulation_results = simulation_results['results']
            else:
                # Format standard (dictionnaire simple)
                is_future_scenario = config.future_scenarios.is_future_scenario(scenario)
            
            # Check if scenario is a future scenario
            if not is_future_scenario:
                # Regular property data loading
                if self.property_data is None or self.zone_weights is None:
                    self._load_property_data(year)
                    self.zone_weights = ZoneWeights(self.property_data, year)
                    
                    # Calculate representativity factors
                    if apply_representativity:
                        self.zone_weights.calculate_representativity_factors()
            else:
                # Load property data for future scenario
                self._load_property_data_for_scenario(year, scenario)
                
                # MODIFICATION: Récupérer les paramètres du scénario pour ajuster les facteurs de représentativité
                from ..scenario_manager import ScenarioManager
                scenario_manager = ScenarioManager()
                scenario_params = scenario_manager.get_scenario_params(scenario)
                
                # Calculer les facteurs de représentativité avec les paramètres du scénario
                if apply_representativity and 'growth' in scenario_params:
                    self.logger.info(f"Calcul des facteurs de représentativité pour le scénario futur {scenario}")
                    self.zone_weights.calculate_representativity_factors(scenario_params)
                else:
                    # Pas de paramètres de croissance, calcul standard
                    self.zone_weights.calculate_representativity_factors()
            
            # Create output directory
            if output_dir is None:
                output_dir = ensure_dir(
                    config.paths['output'].get_simulation_dir(year, scenario)
                )
            
            # NOUVEAU: Sauvegarder un résumé des données foncières avant l'agrégation
            if self.property_data is not None:
                property_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'scenario': scenario,
                    'year': year,
                    'building_counts': {}
                }
                
                # Ajouter les comptes par type de bâtiment
                if 'building_type' in self.property_data.columns:
                    for btype, group in self.property_data.groupby('building_type'):
                        property_summary['building_counts'][btype] = {
                            'count': int(len(group)),
                            'total_area_m2': float(group['aire_etages'].sum()),
                            'avg_area_m2': float(group['aire_etages'].mean())
                        }
                
                # Calculer totaux
                property_summary['total_buildings'] = int(len(self.property_data))
                property_summary['total_area_m2'] = float(self.property_data['aire_etages'].sum())
                property_summary['total_area_million_m2'] = float(property_summary['total_area_m2'] / 1_000_000)
                
                # Sauvegarder
                property_dir = ensure_dir(output_dir / 'property_data')
                with open(property_dir / 'summary.json', 'w') as f:
                    json.dump(property_summary, f, indent=2)
                
                # Log des statistiques
                self.logger.info(f"Résumé des données foncières pour {scenario}:")
                self.logger.info(f"  → Total: {property_summary['total_buildings']} bâtiments, "
                           f"{property_summary['total_area_million_m2']:.2f} millions m²")
                for btype, stats in property_summary.get('building_counts', {}).items():
                    self.logger.info(f"  → {btype}: {stats['count']} bâtiments, {stats['total_area_m2']/1_000_000:.2f} millions m²")
            
            # Aggregate by zone
            zonal_results = {}
            zone_summary = {}
            for zone in set(a.attrs.get('weather_zone') for a in simulation_results.values()):
                # MODIFICATION: Passer les archétypes transformés à _aggregate_zone
                results = self._aggregate_zone(
                    zone, 
                    simulation_results,
                    apply_representativity=apply_representativity,
                    transformed_archetypes_df=transformed_archetypes_df
                )
                if results is not None:
                    zonal_results[zone] = results
                    zone_summary[zone] = self._get_zone_summary(results)
                    
                    # Save zone results
                    zone_dir = ensure_dir(output_dir / 'zones' / f'zone_{int(zone)}')
                    save_results(results, zone_dir / 'hourly.csv')
                    with open(zone_dir / 'summary.json', 'w') as f:
                        json.dump(zone_summary[zone], f, indent=2)
            
            # Aggregate provincial results
            provincial_results = self._aggregate_provincial(zonal_results)
            if provincial_results is not None:
                prov_dir = ensure_dir(output_dir / 'provincial')
                save_results(provincial_results, prov_dir / 'hourly.csv')
                
                # Calculate provincial summary
                provincial_summary = self._get_zone_summary(provincial_results)
                
                # Add validation metrics if this is not a future scenario
                if not is_future_scenario:
                    validation = self.validate_results(provincial_results, year)
                    provincial_summary['validation'] = validation
                
                # NOUVEAU: Ajouter des comparaisons directes entre scénarios PV et UB si applicable
                if is_future_scenario and '_PV_' in scenario:
                    # Essayer de trouver le scénario UB correspondant
                    ub_scenario = scenario.replace('_PV_', '_UB_')
                    ub_dir = config.paths['output'].get_simulation_dir(year, ub_scenario)
                    ub_summary_file = ub_dir / 'provincial' / 'summary.json'
                    
                    if ub_summary_file.exists():
                        try:
                            with open(ub_summary_file, 'r') as f:
                                ub_summary = json.load(f)
                            
                            # Calculer la différence de consommation
                            pv_consumption = provincial_summary['annual_consumption'].get('total_twh', 0)
                            ub_consumption = ub_summary['annual_consumption'].get('total_twh', 0)
                            
                            # Ajouter la comparaison
                            provincial_summary['scenario_comparison'] = {
                                'comparison_with': ub_scenario,
                                'this_twh': float(pv_consumption),
                                'other_twh': float(ub_consumption),
                                'difference_twh': float(ub_consumption - pv_consumption),
                                'percentage_difference': float((ub_consumption - pv_consumption) / pv_consumption * 100) if pv_consumption > 0 else 0
                            }
                            
                            self.logger.info(f"Comparaison entre scénarios {scenario} et {ub_scenario}:")
                            self.logger.info(f"  → Différence de {provincial_summary['scenario_comparison']['difference_twh']:.2f} TWh "
                                      f"({provincial_summary['scenario_comparison']['percentage_difference']:.1f}%)")
                            
                        except Exception as e:
                            self.logger.warning(f"Impossible de comparer les scénarios: {str(e)}")
                    
                elif is_future_scenario and '_UB_' in scenario:
                    # Essayer de trouver le scénario PV correspondant
                    pv_scenario = scenario.replace('_UB_', '_PV_')
                    pv_dir = config.paths['output'].get_simulation_dir(year, pv_scenario)
                    pv_summary_file = pv_dir / 'provincial' / 'summary.json'
                    
                    if pv_summary_file.exists():
                        try:
                            with open(pv_summary_file, 'r') as f:
                                pv_summary = json.load(f)
                            
                            # Calculer la différence de consommation
                            ub_consumption = provincial_summary['annual_consumption'].get('total_twh', 0)
                            pv_consumption = pv_summary['annual_consumption'].get('total_twh', 0)
                            
                            # Ajouter la comparaison
                            provincial_summary['scenario_comparison'] = {
                                'comparison_with': pv_scenario,
                                'this_twh': float(ub_consumption),
                                'other_twh': float(pv_consumption),
                                'difference_twh': float(ub_consumption - pv_consumption),
                                'percentage_difference': float((ub_consumption - pv_consumption) / pv_consumption * 100) if pv_consumption > 0 else 0
                            }
                            
                            self.logger.info(f"Comparaison entre scénarios {scenario} et {pv_scenario}:")
                            self.logger.info(f"  → Différence de {provincial_summary['scenario_comparison']['difference_twh']:.2f} TWh "
                                      f"({provincial_summary['scenario_comparison']['percentage_difference']:.1f}%)")
                            
                        except Exception as e:
                            self.logger.warning(f"Impossible de comparer les scénarios: {str(e)}")
                
                with open(prov_dir / 'summary.json', 'w') as f:
                    json.dump(provincial_summary, f, indent=2)
            
            # MRC aggregation - skip if requested (for calibration speedup)
            mrc_results = {}
            if not skip_mrc_aggregation:
                self.logger.info(f"Performing MRC aggregation for {scenario}")
                mrc_results = self._aggregate_by_mrc(zonal_results)
                if mrc_results:
                    mrc_dir = ensure_dir(output_dir / 'mrc')
                    for mrc_id, results in mrc_results.items():
                        mrc_specific_dir = ensure_dir(mrc_dir / f'mrc_{mrc_id}')
                        save_results(results, mrc_specific_dir / 'hourly.csv')
                        
                        # Chercher le nom de la MRC dans le mapping
                        mrc_name = "Unknown"
                        for zone_mapping in self._load_mrc_mapping().values():
                            if mrc_id in zone_mapping['mrc_ids']:
                                idx = zone_mapping['mrc_ids'].index(mrc_id)
                                mrc_name = zone_mapping['mrc_names'][idx]
                                break
                        
                        # Sauvegarder le résumé MRC
                        mrc_summary = self._get_zone_summary(results)
                        mrc_summary['mrc_id'] = mrc_id
                        mrc_summary['mrc_name'] = mrc_name
                        
                        with open(mrc_specific_dir / 'summary.json', 'w') as f:
                            json.dump(mrc_summary, f, indent=2)
            else:
                self.logger.info(f"Skipping MRC aggregation for {scenario} (optimization)")
                    
            self.logger.info(f"Aggregation completed: {len(zonal_results)} zones, {len(mrc_results)} MRCs")
            
            return provincial_results, zonal_results, mrc_results
            
        except Exception as e:
            self.logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            return None, {}, {}
    
    def get_zone_statistics(self, year: int) -> pd.DataFrame:
        """
        Obtient des statistiques sur les zones depuis les données foncières.
        
        Args:
            year: Année à analyser
            
        Returns:
            DataFrame avec les statistiques par zone
        """
        if self.property_data is None or self.zone_weights is None:
            self._load_property_data(year)
            self.zone_weights = ZoneWeights(self.property_data, year)
            
        stats = []
        
        for zone in self.zone_weights.weights.keys():
            zone_data = self.property_data[
                self.property_data['weather_zone'] == zone
            ]
            
            # Statistiques par typologie
            type_stats = {}
            for building_type in ['Maisons unifamiliales', 'Maisons individuelles attenantes', 'Appartements']:
                # Utiliser la classification simplifiée
                type_mask = zone_data['building_type'] == building_type
                type_data = zone_data[type_mask]
                
                type_stats[building_type] = {
                    'total_area': type_data['aire_etages'].sum(),
                    'building_count': len(type_data),
                    'avg_area': type_data['aire_etages'].mean() if len(type_data) > 0 else 0,
                    'avg_year': type_data['annee_construction'].mean() if len(type_data) > 0 else 0
                }
            
            # Statistiques globales de la zone
            zone_total_area = sum(stats['total_area'] for stats in type_stats.values())
            zone_total_buildings = sum(stats['building_count'] for stats in type_stats.values())
            
            stats.append({
                'zone': zone,
                'building_count': zone_total_buildings,
                'total_floor_area': zone_total_area,
                'avg_floor_area': zone_total_area / zone_total_buildings if zone_total_buildings > 0 else 0,
                'avg_year_built': zone_data['annee_construction'].mean(),
                'building_types': type_stats
            })
        
        return pd.DataFrame(stats)

    def _get_zone_summary(self, results: pd.DataFrame) -> Dict:
        """
        Génère un résumé des résultats pour une zone.
        
        Args:
            results: DataFrame des résultats horaires pour la zone
            
        Returns:
            Dictionnaire contenant des statistiques résumées
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'annual_consumption': {},
            'building_data': {}  # Nouvelle section pour les données sur les bâtiments
        }
        
        # Consommation annuelle totale
        if 'Fuel Use: Electricity: Total' in results.columns:
            total_kwh = results['Fuel Use: Electricity: Total'].sum()
            summary['annual_consumption']['total_kwh'] = float(total_kwh)
            summary['annual_consumption']['total_mwh'] = float(total_kwh / 1000)
            summary['annual_consumption']['total_gwh'] = float(total_kwh / 1000000)
            summary['annual_consumption']['total_twh'] = float(total_kwh / 1000000000)  # Nouveau: TWh pour comparaison facile
        
        # Consommation par usage final
        end_uses = [col for col in results.columns if col.startswith('End Use: Electricity:')]
        for use in end_uses:
            # Extraire le nom de l'usage final
            use_name = use.replace('End Use: Electricity: ', '').lower().replace(' ', '_')
            annual_use = results[use].sum()
            summary['annual_consumption'][f'{use_name}_kwh'] = float(annual_use)
        
        # Statistiques de pic de demande
        if 'Fuel Use: Electricity: Total' in results.columns:
            peak_demand = results['Fuel Use: Electricity: Total'].max()
            peak_time = results['Fuel Use: Electricity: Total'].idxmax()
            
            summary['peak_demand'] = {
                'value_kw': float(peak_demand),
                'timestamp': peak_time.isoformat() if isinstance(peak_time, pd.Timestamp) else str(peak_time)
            }
        
        # NOUVEAU: Ajouter des informations sur les bâtiments si ZoneWeights est disponible
        if hasattr(self, 'zone_weights') and self.zone_weights is not None:
            # Extraire la zone à partir des attributs du résultat si disponible
            zone = None
            if hasattr(results, 'attrs') and 'aggregation_stats' in results.attrs:
                zone = results.attrs['aggregation_stats'].get('zone')
            
            if zone is not None and zone in self.zone_weights.weights:
                zone_weights = self.zone_weights.weights[zone]
                
                # Compteurs pour les totaux par typologie
                building_totals = {}
                for building_type in ['Maisons unifamiliales', 'Maisons individuelles attenantes', 'Appartements']:
                    if building_type in zone_weights:
                        type_data = zone_weights[building_type]
                        
                        # Calculer le total pour ce type
                        total_area = 0
                        total_buildings = 0
                        
                        for period in type_data:
                            for system, system_data in type_data[period].items():
                                total_area += system_data['total_area']
                                total_buildings += system_data['building_count']
                        
                        building_totals[building_type] = {
                            'total_area_m2': float(total_area),
                            'building_count': int(total_buildings),
                            'avg_area_m2': float(total_area / total_buildings) if total_buildings > 0 else 0
                        }
                
                # Calculer le total global
                total_area_all = sum(data['total_area_m2'] for data in building_totals.values())
                total_buildings_all = sum(data['building_count'] for data in building_totals.values())
                
                # Ajouter au résumé
                summary['building_data'] = {
                    'by_type': building_totals,
                    'total_area_m2': float(total_area_all),
                    'total_buildings': int(total_buildings_all),
                    'total_area_million_m2': float(total_area_all / 1_000_000)
                }
                
                # Calculer les intensités énergétiques par typologie si possible
                if 'Fuel Use: Electricity: Total' in results.columns:
                    # Estimation approximative basée sur la proportion de surface
                    total_consumption = results['Fuel Use: Electricity: Total'].sum()
                    intensities = {}
                    
                    for building_type, data in building_totals.items():
                        area_proportion = data['total_area_m2'] / total_area_all if total_area_all > 0 else 0
                        estimated_consumption = total_consumption * area_proportion
                        intensity = estimated_consumption / data['total_area_m2'] if data['total_area_m2'] > 0 else 0
                        
                        intensities[building_type] = {
                            'estimated_kwh': float(estimated_consumption),
                            'intensity_kwh_per_m2': float(intensity)
                        }
                    
                    summary['building_data']['energy_intensities'] = intensities
                    summary['building_data']['overall_intensity_kwh_per_m2'] = float(total_consumption / total_area_all) if total_area_all > 0 else 0
        
        return summary