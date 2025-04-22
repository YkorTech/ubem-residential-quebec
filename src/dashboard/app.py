"""
Main dashboard application for UBEM Quebec.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from pathlib import Path
import logging

from ..config import config
from .layouts import create_layout
from .callbacks import register_callbacks
from .schedule_module import register_schedule_callbacks

# Initialize logger
logger = logging.getLogger(__name__)

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="UBEM Quebec Dashboard",
    suppress_callback_exceptions=True  
)

# Create layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)
register_schedule_callbacks(app)

def run_dashboard(debug: bool = True, port: int = 8050):
    """
    Run the dashboard application.
    
    Args:
        debug: Enable debug mode
        port: Port to run the server on
    """
    try:
        logging.getLogger('src.utils').setLevel(logging.WARNING)
        logging.getLogger('src.managers.simulation_manager').setLevel(logging.WARNING)
        logging.getLogger('src.managers.aggregation_manager').setLevel(logging.WARNING)
        
        # Ensure MCMC and PSO loggers are at INFO level
        logging.getLogger('src.calibration.mcmc_calibrator').setLevel(logging.INFO)
        logging.getLogger('src.calibration.pso_optimizer').setLevel(logging.INFO)
        
        # Set root logger to INFO
        logging.getLogger().setLevel(logging.INFO)
        
        logger.info(f"Starting dashboard on port {port}")
        app.run_server(debug=debug, port=port)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

if __name__ == '__main__':
    run_dashboard()
