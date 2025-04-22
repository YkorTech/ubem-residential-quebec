"""
Module for generating stochastic schedules for Quebec UBEM.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

class ScheduleGenerator:
    """Generates stochastic schedules for different building loads."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Init schedule generator.
        
        Args:
            seed: RNG seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.seed = seed or random.randint(1, 10000)
        self.rng = random.Random(self.seed)
        self.cache_dir = Path("data/inputs/schedules")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_archetype_seed(self, archetype_id: int, archetype_data: Dict) -> int:
        """
        Generate deterministic seed based on archetype characteristics.
        
        Args:
            archetype_id: Archetype ID
            archetype_data: Archetype data
            
        Returns:
            Seed value for PRNG
        """
        # Create unique string from key characteristics
        key_str = f"{archetype_id}_{archetype_data.get('weather_zone')}_{archetype_data.get('houseType')}_"
        key_str += f"{archetype_data.get('vintageExact')}_{archetype_data.get('spaceHeatingType')}"
        
        # Generate hash and convert to integer
        hash_val = int(hashlib.md5(key_str.encode()).hexdigest(), 16) 
        return hash_val % 1000000
    
    def _generate_daily_profile(self, profile_type: str, 
                               occupants: float, 
                               weekday: bool,
                               rng: random.Random) -> List[float]:
        """
        Generate daily profile (24 values) for schedule type.
        
        Args:
            profile_type: Profile type ('occupants', 'lighting', etc.)
            occupants: Number of occupants
            weekday: Weekday (True) or weekend (False)
            rng: Random number generator
            
        Returns:
            List of 24 values (0-1)
        """
        # Base profiles for different types
        base_profiles = {
            'occupants': {
                'weekday': [
                    0.85, 0.90, 0.90, 0.90, 0.90, 0.80,  # 0-5h: Night
                    0.70, 0.50, 0.30, 0.20, 0.20, 0.20,  # 6-11h: Morning
                    0.20, 0.20, 0.20, 0.30, 0.50,        # 12-16h: Afternoon
                    0.60, 0.70, 0.80, 0.85, 0.85, 0.85, 0.85  # 17-23h: Evening
                ],
                'weekend': [
                    0.85, 0.90, 0.90, 0.90, 0.90, 0.85,  # 0-5h: Night
                    0.80, 0.70, 0.60, 0.50, 0.50, 0.50,  # 6-11h: Morning
                    0.50, 0.50, 0.50, 0.60, 0.60,        # 12-16h: Afternoon
                    0.70, 0.70, 0.80, 0.85, 0.85, 0.85, 0.85  # 17-23h: Evening
                ]
            },
            'lighting_interior': {
                'weekday': [
                    0.10, 0.05, 0.05, 0.05, 0.05, 0.10,  # 0-5h: Night
                    0.30, 0.40, 0.30, 0.20, 0.20, 0.20,  # 6-11h: Morning
                    0.20, 0.20, 0.20, 0.30, 0.50,        # 12-16h: Afternoon
                    0.70, 0.80, 0.70, 0.50, 0.30, 0.20, 0.15  # 17-23h: Evening
                ],
                'weekend': [
                    0.10, 0.05, 0.05, 0.05, 0.05, 0.10,  # 0-5h: Night
                    0.20, 0.30, 0.40, 0.40, 0.30, 0.30,  # 6-11h: Morning
                    0.30, 0.30, 0.30, 0.40, 0.50,        # 12-16h: Afternoon
                    0.60, 0.70, 0.70, 0.50, 0.30, 0.20, 0.15  # 17-23h: Evening
                ]
            },
            'cooking_range': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.10, 0.20, 0.10, 0.05, 0.05, 0.10,  # 6-11h: Morning
                    0.20, 0.10, 0.05, 0.10, 0.20,        # 12-16h: Afternoon
                    0.40, 0.60, 0.30, 0.15, 0.05, 0.01, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.10, 0.20, 0.30, 0.20, 0.10, 0.20,  # 6-11h: Morning
                    0.40, 0.20, 0.10, 0.15, 0.25,        # 12-16h: Afternoon
                    0.30, 0.50, 0.30, 0.15, 0.05, 0.01, 0.01  # 17-23h: Evening
                ]
            },
            'dishwasher': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.01, 0.05, 0.10, 0.05, 0.05, 0.05,  # 6-11h: Morning
                    0.05, 0.05, 0.05, 0.05, 0.10,        # 12-16h: Afternoon
                    0.30, 0.60, 0.40, 0.20, 0.10, 0.05, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.01, 0.05, 0.20, 0.30, 0.20, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.10, 0.20,        # 12-16h: Afternoon
                    0.30, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01  # 17-23h: Evening
                ]
            },
            'clothes_washer': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.05, 0.10, 0.10, 0.10, 0.10, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.15, 0.20,        # 12-16h: Afternoon
                    0.30, 0.40, 0.20, 0.10, 0.05, 0.01, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.05, 0.20, 0.40, 0.40, 0.30, 0.20,  # 6-11h: Morning
                    0.20, 0.20, 0.20, 0.20, 0.30,        # 12-16h: Afternoon
                    0.30, 0.20, 0.10, 0.05, 0.01, 0.01, 0.01  # 17-23h: Evening
                ]
            },
            'hot_water_fixtures': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.20, 0.40, 0.30, 0.20, 0.15, 0.15,  # 6-11h: Morning
                    0.15, 0.15, 0.15, 0.20, 0.25,        # 12-16h: Afternoon
                    0.35, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.10, 0.30, 0.40, 0.40, 0.30, 0.25,  # 6-11h: Morning
                    0.25, 0.25, 0.25, 0.30, 0.30,        # 12-16h: Afternoon
                    0.30, 0.35, 0.25, 0.15, 0.10, 0.05, 0.01  # 17-23h: Evening
                ]
            },
            'plug_loads_other': {
                'weekday': [
                    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,  # 0-5h: Night
                    0.30, 0.40, 0.40, 0.40, 0.40, 0.40,  # 6-11h: Morning
                    0.40, 0.40, 0.40, 0.40, 0.50,        # 12-16h: Afternoon
                    0.60, 0.70, 0.70, 0.60, 0.50, 0.30, 0.20  # 17-23h: Evening
                ],
                'weekend': [
                    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,  # 0-5h: Night
                    0.30, 0.40, 0.50, 0.50, 0.50, 0.50,  # 6-11h: Morning
                    0.50, 0.50, 0.50, 0.60, 0.60,        # 12-16h: Afternoon
                    0.70, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20  # 17-23h: Evening
                ]
            },
            'ceiling_fan': {
                'weekday': [
                    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 0-5h: Night
                    0.20, 0.30, 0.20, 0.10, 0.10, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.20, 0.30,        # 12-16h: Afternoon
                    0.40, 0.50, 0.50, 0.40, 0.30, 0.20, 0.10  # 17-23h: Evening
                ],
                'weekend': [
                    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 0-5h: Night
                    0.20, 0.30, 0.40, 0.40, 0.40, 0.40,  # 6-11h: Morning
                    0.40, 0.40, 0.40, 0.40, 0.40,        # 12-16h: Afternoon
                    0.50, 0.50, 0.40, 0.30, 0.20, 0.10, 0.10  # 17-23h: Evening
                ]
            },
            'plug_loads_tv': {
                'weekday': [
                    0.05, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.10, 0.10, 0.05, 0.05, 0.05, 0.05,  # 6-11h: Morning
                    0.05, 0.05, 0.05, 0.10, 0.20,        # 12-16h: Afternoon
                    0.30, 0.60, 0.80, 0.70, 0.50, 0.30, 0.15  # 17-23h: Evening
                ],
                'weekend': [
                    0.05, 0.01, 0.01, 0.01, 0.01, 0.05,  # 0-5h: Night
                    0.10, 0.20, 0.30, 0.30, 0.30, 0.30,  # 6-11h: Morning
                    0.30, 0.30, 0.30, 0.40, 0.40,        # 12-16h: Afternoon
                    0.50, 0.70, 0.80, 0.70, 0.60, 0.40, 0.20  # 17-23h: Evening
                ]
            },

            # Additional profiles
            'clothes_dryer': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.01, 0.05, 0.10, 0.10, 0.10, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.15, 0.20,        # 12-16h: Afternoon
                    0.30, 0.40, 0.20, 0.10, 0.05, 0.01, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.05, 0.20, 0.40, 0.40, 0.30, 0.20,  # 6-11h: Morning
                    0.20, 0.20, 0.20, 0.20, 0.30,        # 12-16h: Afternoon
                    0.30, 0.20, 0.10, 0.05, 0.01, 0.01, 0.01  # 17-23h: Evening
                ]
            },
            'hot_water_dishwasher': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.01, 0.05, 0.10, 0.05, 0.05, 0.05,  # 6-11h: Morning
                    0.05, 0.05, 0.05, 0.05, 0.10,        # 12-16h: Afternoon
                    0.30, 0.60, 0.40, 0.20, 0.10, 0.05, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.01, 0.05, 0.20, 0.30, 0.20, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.10, 0.20,        # 12-16h: Afternoon
                    0.30, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01  # 17-23h: Evening
                ]
            },
            'hot_water_clothes_washer': {
                'weekday': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.05, 0.10, 0.10, 0.10, 0.10, 0.10,  # 6-11h: Morning
                    0.10, 0.10, 0.10, 0.15, 0.20,        # 12-16h: Afternoon
                    0.30, 0.40, 0.20, 0.10, 0.05, 0.01, 0.01  # 17-23h: Evening
                ],
                'weekend': [
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5h: Night
                    0.05, 0.20, 0.40, 0.40, 0.30, 0.20,  # 6-11h: Morning
                    0.20, 0.20, 0.20, 0.20, 0.30,        # 12-16h: Afternoon
                    0.30, 0.20, 0.10, 0.05, 0.01, 0.01, 0.01  # 17-23h: Evening
                ]
            }

        }
        
        # Fix profiles that don't have exactly 24 hours
        for profile_category in base_profiles.values():
            for day_type_key in list(profile_category.keys()):
                day_type = profile_category[day_type_key]
                if len(day_type) != 24:
                    # Create new list instead of modifying original
                    if len(day_type) > 24:
                        profile_category[day_type_key] = day_type[:24]
                    else:
                        new_profile = day_type.copy()
                        new_profile.extend([day_type[-1]] * (24 - len(day_type)))
                        profile_category[day_type_key] = new_profile
        
        # Select base profile by type and day
        day_type = 'weekend' if not weekday else 'weekday'
        if profile_type in base_profiles and day_type in base_profiles[profile_type]:
            base = base_profiles[profile_type][day_type].copy()
        else:
            # Default profile if undefined
            base = base_profiles['occupants'][day_type].copy()
        
        # VERIFICATION: Ensure profile has exactly 24 hours
        if len(base) != 24:
            # Handle any profiles that escaped earlier correction
            if len(base) > 24:
                base = base[:24]
            else:
                base.extend([base[-1]] * (24 - len(base)))
        
        # Introduce different behavioral styles
        behavior_types = ['early_riser', 'late_sleeper', 'standard', 'home_worker', 'night_owl']
        
        # CORRECTION: Manual weighted choice instead of rng.choice with p param
        # Define weights for behaviors
        weights = [0.2, 0.2, 0.4, 0.1, 0.1]  # Sum = 1.0
        
        # Manual weighted choice implementation
        r = rng.random()  # Random number between 0 and 1
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r < cumsum:
                behavior = behavior_types[i]
                break
        else:
            behavior = 'standard'  # Fallback
        
        # Apply modifications based on behavior type
        modified_base = base.copy()
        if behavior == 'early_riser':
            # Safe shift
            shifted = [base[-1]] + base[:-1]  # Shift 1h earlier (safer)
            
            # Loops with index check
            for i in range(5, min(9, len(shifted))):
                modified_base[i] = min(1.0, shifted[i] * 1.2)
            for i in range(19, min(24, len(shifted))):
                modified_base[i] = shifted[i] * 0.9
                
        elif behavior == 'late_sleeper':
            # Safe shift
            shifted = base[1:] + [base[0]]  # Shift 1h later (safer)
            
            # Loops with index check
            for i in range(5, min(9, len(shifted))):
                modified_base[i] = shifted[i] * 0.8
            for i in range(19, min(24, len(shifted))):
                modified_base[i] = min(1.0, shifted[i] * 1.2)
                
        elif behavior == 'home_worker':
            # Loops with index check
            for i in range(9, min(17, len(base))):
                modified_base[i] = min(1.0, base[i] * 1.5)
            for i in range(6, min(9, len(base))):
                modified_base[i] = base[i] * 0.8
            for i in range(17, min(20, len(base))):
                modified_base[i] = base[i] * 0.9
                
        elif behavior == 'night_owl':
            # Loops with index check
            for i in range(0, min(9, len(base))):
                modified_base[i] = base[i] * 0.7
            for i in range(21, min(24, len(base))):
                modified_base[i] = min(1.0, base[i] * 1.3)
        
        # Code remains identical to previous version
        # Customize based on occupant count
        if profile_type == 'occupants':
            multiplier = min(occupants / 3.0, 1.0)
            profile = [value * multiplier for value in modified_base]
        else:
            occupant_factor = min(1.0, 0.7 + 0.3 * (occupants / 3.0))
            profile = [value * occupant_factor for value in modified_base]
        
        # Add stochastic variation (±15% max) - reduced to avoid peaks
        profile = [value * (1 + rng.uniform(-0.15, 0.15)) for value in profile]
        
        # Smoothing to reduce abrupt variations between adjacent hours
        smoothed = profile.copy()
        for i in range(1, 23):
            # Moving average with 20%-60%-20% weights
            smoothed[i] = 0.2 * profile[i-1] + 0.6 * profile[i] + 0.2 * profile[i+1]
        
        # Ensure values remain between 0 and 1
        final_profile = [max(0.0, min(1.0, value)) for value in smoothed]
        
        return final_profile
    
    def generate_annual_schedule(self, 
                            profile_types: List[str], 
                            archetype_id: int,
                            archetype_data: Dict,
                            scale_factors: Optional[Dict[str, float]] = None,
                            temporal_diversity: float = 0.0) -> pd.DataFrame:
        """
        Generate annual schedule for an archetype.
        
        Args:
            profile_types: List of profile types to generate
            archetype_id: Archetype ID
            archetype_data: Archetype data
            scale_factors: Scale factors for different profile types
            temporal_diversity: Temporal diversity factor between archetypes [0.0-1.0]
                            0.0 = no shift, 1.0 = maximum shift
            
        Returns:
            DataFrame with hourly schedules (8760 rows)
        """
        # Init default scale factors if not provided
        if scale_factors is None:
            scale_factors = {'occupancy_scale': 0.0, 'lighting_scale': 0.0, 'appliance_scale': 0.0}
        
        # Ensure temporal_diversity is within [0.0, 1.0]
        temporal_diversity = max(0.0, min(1.0, temporal_diversity))
        
        # Get deterministic seed for this archetype
        seed = self._get_archetype_seed(archetype_id, archetype_data)
        rng = random.Random(seed)
        
        # Create index for complete year
        time_index = list(range(8760))
        
        # Init DataFrame
        schedules_df = pd.DataFrame(index=time_index)
        
        # Extract number of occupants (or use default)
        occupants = float(archetype_data.get('numAdults', 2.0)) + float(archetype_data.get('numChildren', 0.0))
        if occupants <= 0:
            occupants = 2.0  # Default value
        
        # Generate differentiated shifts by time period
        # Use float values for partial shifts
        hash_value = (archetype_id * seed) % 10000
        period_rng = random.Random(hash_value)
        
        # Specific shifts by time of day
        morning_shift = period_rng.uniform(-2.5, 2.5) * temporal_diversity  # 5h-10h period
        evening_shift = period_rng.uniform(-2.5, 2.5) * temporal_diversity  # 16h-21h period
        midday_shift = period_rng.uniform(-1.5, 1.5) * temporal_diversity   # 10h-16h period
        night_shift = period_rng.uniform(-1.0, 1.0) * temporal_diversity    # 21h-5h period
        
        if temporal_diversity > 0:
            self.logger.info(f"Archetype {archetype_id}: Temporal shifts - "
                            f"morning: {morning_shift:.2f}h, "
                            f"midday: {midday_shift:.2f}h, "
                            f"evening: {evening_shift:.2f}h, "
                            f"night: {night_shift:.2f}h")
        
        # Generate each profile
        for profile_type in profile_types:
            # Init empty array for the year
            annual_profile = [0.0] * 8760
            
            # For each day of the year
            for day in range(365):
                # Determine if it's a weekday (0=Monday, 6=Sunday)
                date = (day + 5) % 7  # Shift to start with Saturday
                is_weekday = date < 5  # Monday-Friday = 0-4
                
                # Generate daily profile
                daily_profile = self._generate_daily_profile(
                    profile_type, occupants, is_weekday, rng
                )
                
                # Apply differentiated shifts based on hour
                shifted_profile = []
                for hour in range(24):
                    # Determine applicable shift based on hour
                    if 5 <= hour < 10:  # Morning peak hours
                        shift = morning_shift
                    elif 16 <= hour < 21:  # Evening peak hours
                        shift = evening_shift
                    elif 10 <= hour < 16:  # Midday
                        shift = midday_shift
                    else:  # Night
                        shift = night_shift
                        
                    # Calculate source hour with float shift
                    source_hour = (hour - shift) % 24
                    hour_index = int(source_hour)
                    fraction = source_hour - hour_index

                    # Linear interpolation between two hours
                    next_index = (hour_index + 1) % 24
                    # ADDITION: Check index validity
                    if hour_index < len(daily_profile) and next_index < len(daily_profile):
                        value = daily_profile[hour_index] * (1-fraction) + daily_profile[next_index] * fraction
                    else:
                        # Use default value in case of problem
                        value = daily_profile[hour_index % len(daily_profile)]
                    shifted_profile.append(value)
                
                # Insert into annual profile
                for hour in range(24):
                    annual_profile[day*24 + hour] = shifted_profile[hour]
            
            # Add seasonal variations
            if profile_type in ['lighting_interior', 'lighting_garage']:
                # More lighting in winter
                for day in range(365):
                    month = (day // 30) + 1  # Approximation
                    if month > 12: month = 12
                        
                    if month in [11, 12, 1, 2]:  # Winter
                        factor = rng.uniform(1.1, 1.3)
                    elif month in [6, 7, 8]:  # Summer
                        factor = rng.uniform(0.7, 0.9)
                    else:
                        factor = 1.0
                        
                    for hour in range(24):
                        index = day*24 + hour
                        annual_profile[index] = min(1.0, annual_profile[index] * factor)
            
            # Apply scale factors
            if profile_type == 'occupants' and 'occupancy_scale' in scale_factors:
                scale = 1.0 + scale_factors['occupancy_scale']
                # Non-linear application attenuated at peak hours
                for i in range(len(annual_profile)):
                    hour = (i % 24)
                    if 6 <= hour < 9 or 17 <= hour < 20:  # Peak hours
                        # Apply attenuated factor at peak hours to reduce spikes
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * 0.7)
                    else:
                        # Normal application at other hours
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * annual_profile[i])
                        
            elif profile_type in ['lighting_interior', 'lighting_garage'] and 'lighting_scale' in scale_factors:
                scale = 1.0 + scale_factors['lighting_scale']
                # Similar application for lighting
                for i in range(len(annual_profile)):
                    hour = (i % 24)
                    if 6 <= hour < 9 or 17 <= hour < 20:  # Peak hours
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * 0.7)
                    else:
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * annual_profile[i])
                        
            elif profile_type in ['cooking_range', 'dishwasher', 'clothes_washer', 'plug_loads_other', 
                            'plug_loads_tv', 'ceiling_fan'] and 'appliance_scale' in scale_factors:
                scale = 1.0 + scale_factors['appliance_scale']
                # Similar application for appliances
                for i in range(len(annual_profile)):
                    hour = (i % 24)
                    if 6 <= hour < 9 or 17 <= hour < 20:  # Peak hours
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * 0.7)
                    else:
                        annual_profile[i] = annual_profile[i] * (1.0 + (scale - 1.0) * annual_profile[i])
            
            # Add profile to DataFrame
            schedules_df[profile_type] = annual_profile
        
        # Final normalization
        for column in schedules_df.columns:
            max_val = schedules_df[column].max()
            if max_val > 0:
                schedules_df[column] = schedules_df[column] / max_val
            else:
                schedules_df.loc[12, column] = 1.0
        
        return schedules_df
    
    def generate_and_save_schedule(self,
                                  profile_types: List[str],
                                  archetype_id: int,
                                  archetype_data: Dict,
                                  scale_factors: Optional[Dict[str, float]] = None,
                                  temporal_diversity: float = 0.0,
                                  output_dir: Optional[Path] = None) -> Path:
        """
        Generate and save schedule for an archetype.
        
        Args:
            profile_types: List of profile types to generate
            archetype_id: Archetype ID
            archetype_data: Archetype data
            scale_factors: Scale factors for different profile types
            temporal_diversity: Temporal diversity factor between archetypes [0.0-1.0]
            output_dir: Output directory (optional)
            
        Returns:
            Path to saved schedule file
        """
        try:
            # Generate schedule
            schedules_df = self.generate_annual_schedule(
                profile_types, archetype_id, archetype_data, scale_factors, temporal_diversity
            )
            
            # Determine output directory
            if output_dir is None:
                output_dir = self.cache_dir
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique filename based on archetype
            filename = f"schedule_archetype_{archetype_id}.csv"
            filepath = output_dir / filename
            
            # Save schedule
            schedules_df.to_csv(filepath, index=False)
            self.logger.debug(f"Schedule saved to {filepath}") 
            
            return filepath
        except Exception as e:
            self.logger.error(f"Error generating schedule for archetype {archetype_id}: {str(e)}")
            raise
            
    def calculate_schedule_statistics(self, schedules_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics on a schedule.
        
        Args:
            schedules_df: DataFrame containing schedules
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Occupancy statistics
        if 'occupants' in schedules_df.columns:
            occupants = schedules_df['occupants']
            stats['occupants_avg'] = occupants.mean()
            stats['occupants_max'] = occupants.max()
            
            # Peak hours
            daily_profiles = []
            for day in range(365):
                start_idx = day * 24
                end_idx = start_idx + 24
                if end_idx <= len(occupants):
                    daily = occupants[start_idx:end_idx].values
                    daily_profiles.append(daily)
            
            if daily_profiles:
                avg_daily = np.mean(daily_profiles, axis=0)
                morning_peak = max(avg_daily[6:10])  # 6h-10h
                evening_peak = max(avg_daily[17:22])  # 17h-22h
                
                stats['morning_peak'] = morning_peak
                stats['evening_peak'] = evening_peak
                stats['morning_peak_ratio'] = morning_peak / stats['occupants_avg'] if stats['occupants_avg'] > 0 else 1.0
                stats['evening_peak_ratio'] = evening_peak / stats['occupants_avg'] if stats['occupants_avg'] > 0 else 1.0
        
        # Statistics for other schedules
        for column in ['lighting_interior', 'cooking_range', 'plug_loads_other']:
            if column in schedules_df.columns:
                stats[f'{column}_avg'] = schedules_df[column].mean()
        
        return stats