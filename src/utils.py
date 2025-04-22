"""
Utility functions for the project.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from typing import Dict, List, Tuple, Optional, Union, Any

def get_available_cpus(usage_percent: float = 0.8) -> int:
    """
    Get number of CPUs to use based on usage percentage.
    
    Args:
        usage_percent: Percentage of CPUs to use (0.0 to 1.0)
        
    Returns:
        Number of CPUs to use
    """
    n_cpus = os.cpu_count()
    if n_cpus is None:
        return 1
    
    # Calculate CPUs to use without logging
    n_use = max(1, int(n_cpus * usage_percent))
    
    return n_use

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_run_id() -> str:
    """Generate a unique run ID."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def process_hydro_data(data: pd.DataFrame, year) -> pd.DataFrame:
    """
    Process Hydro-Quebec data from 15-min power to hourly energy.
    
    Args:
        data: DataFrame with columns ['Intervalle15Minutes', 'energie_sum_secteur']
        year: Year of the simulation
        
    Returns:
        DataFrame with hourly energy consumption in kWh
    """
    # Convert timestamp to datetime (localize to UTC)
    data['Intervalle15Minutes'] = pd.to_datetime(data['Intervalle15Minutes']).dt.tz_localize(None)
    data.set_index('Intervalle15Minutes', inplace=True)
    
    
    # Resample to hourly, summing the energy
    hourly = data.resample('h').sum()  # Using 'h' instead of deprecated 'H'
    
    # Set year to 2022 (simulation year) to match simulation timestamps
    hourly.index = hourly.index.map(lambda t: t.replace(year=int(year)))

    return hourly

def calculate_daily_profiles(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate average daily profiles by season and day type.
    
    Args:
        data: DataFrame with datetime index and consumption data
        column: Name of the column containing consumption data
        
    Returns:
        DataFrame with average profiles (24 hours x (season, day_type) combinations)
    """
    # Add season and day type indicators
    data = data.copy()
    data['hour'] = data.index.hour
    
    # Map months to seasons (handle December separately)
    def get_season(month):
        if month == 12:
            return 'winter'
        elif month <= 2:
            return 'winter'
        elif month <= 5:
            return 'spring'
        elif month <= 8:
            return 'summer'
        else:
            return 'fall'
    
    data['season'] = data.index.month.map(get_season)
    data['is_weekend'] = data.index.weekday >= 5
    
    # Calculate average profiles with ordered=False to allow duplicate labels
    profiles = data.groupby(['season', 'is_weekend', 'hour'], 
                          observed=True, 
                          sort=False)[column].mean().unstack(level='hour')
    
    return profiles

def calculate_peak_metrics(simulated: pd.DataFrame, measured: pd.DataFrame, 
                         aligned_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None) -> Dict[str, float]:
    """
    Calculate metrics focused on peak demand periods.
    
    Args:
        simulated: DataFrame with simulated hourly energy consumption
        measured: DataFrame with measured hourly energy consumption
        aligned_data: Optional tuple of pre-aligned data to avoid recalculation
        
    Returns:
        Dictionary of peak-related metrics
    """
    try:
        # Use pre-aligned data if provided, otherwise align
        if aligned_data:
            sim, meas = aligned_data
        else:
            # Align timestamps
            sim, meas = align_timestamps(simulated, measured)
        
        # Get consumption columns
        sim_values = sim['Fuel Use: Electricity: Total']
        meas_values = meas['energie_sum_secteur']
        
        # Convert to MWh
        sim_mwh = sim_values / 1000
        meas_mwh = meas_values / 1000
        
        # Add time indicators
        df = pd.DataFrame({
            'sim': sim_mwh,
            'meas': meas_mwh,
            'hour': sim.index.hour,
            'month': sim.index.month,
            'is_weekend': sim.index.weekday >= 5
        })
        
        # Define peak periods
        winter_peak_hours = [7, 8, 9, 16, 17, 18, 19]  # 7-9h et 16-19h
        summer_peak_hours = [11, 12, 13, 14, 15, 16]   # 11h-16h
        
        # Calculate winter peaks (Dec-Feb)
        winter_mask = df['month'].isin([12, 1, 2])
        winter_peak_mask = winter_mask & df['hour'].isin(winter_peak_hours)
        winter_peak_rmse = np.sqrt(((df.loc[winter_peak_mask, 'sim'] - 
                                   df.loc[winter_peak_mask, 'meas']) ** 2).mean())
        
        # Calculate summer peaks (Jun-Aug)
        summer_mask = df['month'].isin([6, 7, 8])
        summer_peak_mask = summer_mask & df['hour'].isin(summer_peak_hours)
        summer_peak_rmse = np.sqrt(((df.loc[summer_peak_mask, 'sim'] - 
                                   df.loc[summer_peak_mask, 'meas']) ** 2).mean())
        
        # Calculate peak timing error
        def get_peak_hour(group):
            if len(group) == 0:
                return np.nan
            max_idx = group.values.argmax()
            return group.index[max_idx].hour

        # Group by month and calculate peak hours
        sim_monthly = df.groupby('month')['sim']
        meas_monthly = df.groupby('month')['meas']
        
        sim_peak_hours = []
        meas_peak_hours = []
        
        for month in range(1, 13):
            if month in sim_monthly.groups:
                sim_group = sim_monthly.get_group(month)
                meas_group = meas_monthly.get_group(month)
                
                sim_peak_hours.append(get_peak_hour(sim_group))
                meas_peak_hours.append(get_peak_hour(meas_group))
        
        # Calculate mean absolute difference in peak hours
        peak_timing_error = np.nanmean(np.abs(
            np.array(sim_peak_hours) - np.array(meas_peak_hours)
        ))
        
        return {
            'winter_peak_rmse': float(winter_peak_rmse),
            'summer_peak_rmse': float(summer_peak_rmse),
            'peak_timing_error': float(peak_timing_error)
        }
        
    except Exception as e:
        logging.error(f"Error calculating peak metrics: {str(e)}")
        return {
            'winter_peak_rmse': float('nan'),
            'summer_peak_rmse': float('nan'),
            'peak_timing_error': float('nan')
        }

def calculate_metrics(simulated: pd.DataFrame, measured: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comparison metrics between simulated and measured data.
    
    Args:
        simulated: DataFrame with simulated hourly energy consumption
        measured: DataFrame with measured hourly energy consumption
        
    Returns:
        Dictionary of metric names and values
    """
    # Ensure both have same index
    common_index = simulated.index.intersection(measured.index)
    sim = simulated.loc[common_index]
    meas = measured.loc[common_index]
    
    # Calculate metrics with error handling
    try:
        logging.info(f"Simulated index range: {sim.index.min()} to {sim.index.max()}")
        logging.info(f"Measured index range: {meas.index.min()} to {meas.index.max()}")
        logging.info(f"Common index points: {len(common_index)}")
        
        # Ensure we're comparing the right columns
        sim_values = sim['Fuel Use: Electricity: Total']
        meas_values = meas['energie_sum_secteur']
        
        logging.info(f"Simulated values shape: {sim_values.shape}")
        logging.info(f"Measured values shape: {meas_values.shape}")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(sim_values) | np.isnan(meas_values))
        sim_clean = sim_values[valid_mask]
        meas_clean = meas_values[valid_mask]
        
        if len(sim_clean) == 0 or len(meas_clean) == 0:
            raise ValueError("No valid data points for comparison")
            
        # Convert units if needed (simulation is in kWh, Hydro data is in kW)
        sim_kw = sim_clean  # Already in kWh, which is what we want
        meas_kw = meas_clean  # Already converted to kWh in process_hydro_data
        
        # Scale values to similar magnitude (convert to MWh)
        sim_mwh = sim_kw / 1000
        meas_mwh = meas_kw / 1000
        
        # Log value ranges
        logging.info(f"Simulated MWh range: {sim_mwh.min():.2f} to {sim_mwh.max():.2f}")
        logging.info(f"Measured MWh range: {meas_mwh.min():.2f} to {meas_mwh.max():.2f}")
        
        # Calculate metrics excluding zero values for MAPE
        non_zero_mask = meas_mwh > 0
        rmse = np.sqrt(((sim_mwh - meas_mwh) ** 2).mean())
        mae = (sim_mwh - meas_mwh).abs().mean()
        # Calculate MAPE only on non-zero values
        mape = ((sim_mwh[non_zero_mask] - meas_mwh[non_zero_mask]).abs() / 
                meas_mwh[non_zero_mask]).mean() * 100
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        rmse = mae = mape = float('nan')
    
    # Calculate base metrics
    base_metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }
    
    try:
        # Calculate peak metrics
        peak_metrics = calculate_peak_metrics(simulated, measured, aligned_data=(sim, meas))
        
        # Calculate profile metrics
        sim_profiles = calculate_daily_profiles(simulated, 'Fuel Use: Electricity: Total')
        meas_profiles = calculate_daily_profiles(measured, 'energie_sum_secteur')
        
        # Profile RMSE by season
        profile_rmse = {}
        for season in sim_profiles.index.get_level_values('season').unique():
            for weekend in [True, False]:
                key = f'profile_rmse_{season}_{"weekend" if weekend else "weekday"}'
                sim_prof = sim_profiles.loc[(season, weekend)]
                meas_prof = meas_profiles.loc[(season, weekend)]
                profile_rmse[key] = float(np.sqrt(((sim_prof - meas_prof) ** 2).mean()))
        
        # Combine all metrics
        metrics = {**base_metrics, **peak_metrics, **profile_rmse}
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating additional metrics: {str(e)}")
        return base_metrics

def align_timestamps(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames to have the same timestamps, handling timezone differences.
    
    Args:
        df1: First DataFrame with datetime index
        df2: Second DataFrame with datetime index
        
    Returns:
        Tuple of (aligned_df1, aligned_df2)
    """
    # Convert both indexes to naive UTC if they have timezones
    if df1.index.tz is not None:
        df1.index = df1.index.tz_localize(None)
    if df2.index.tz is not None:
        df2.index = df2.index.tz_localize(None)
    
    # Get common timestamps
    common_index = df1.index.intersection(df2.index)
    
    # Align both DataFrames
    df1_aligned = df1.loc[common_index]
    df2_aligned = df2.loc[common_index]
    
    return df1_aligned, df2_aligned

def save_results(data: Any, path: Path, format: str = 'csv') -> None:
    """
    Save results to a file.
    
    Args:
        data: Data to save (DataFrame for CSV, dict for JSON)
        path: Path to save file to
        format: File format ('csv' or 'json')
    """
    # Create directory if needed
    ensure_dir(path.parent)
    
    try:
        # Save based on format
        if format.lower() == 'csv':
            data.to_csv(path)
        elif format.lower() == 'json':
            def json_serializer(obj):
                """Handle numpy types during JSON serialization."""
                if isinstance(obj, (np.integer)):
                    return int(obj)
                elif isinstance(obj, (np.floating)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=json_serializer)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logging.info(f"Successfully saved results to {path}")
        
    except Exception as e:
        logging.error(f"Error saving results to {path}: {str(e)}")
        if format.lower() == 'json':
            logging.error(f"Data type: {type(data)}")
            if isinstance(data, dict):
                for k, v in data.items():
                    logging.error(f"Key: {k}, Type: {type(v)}")
        raise
