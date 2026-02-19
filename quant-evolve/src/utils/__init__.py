"""
Utilities module for QuantEvolve.

This module provides utility functions for data loading, logging, and visualization
that support the multi-agent evolutionary framework.
"""

from .data_loader import load_ohlcv_data, load_data_bundle, validate_data
from .logging import setup_logging, EvolutionLogger, get_logger
from .visualization import (
    plot_feature_map_evolution,
    plot_sharpe_vs_generation,
    plot_cumulative_returns,
    plot_category_distribution,
    plot_insight_timeline,
    plot_insight_evolution,
    plot_3d_feature_map,
    plot_dimension_pair_heatmap
)

__all__ = [
    # Data loading
    'load_ohlcv_data',
    'load_data_bundle',
    'validate_data',
    
    # Logging
    'setup_logging',
    'EvolutionLogger',
    'get_logger',
    
    # Visualization
    'plot_feature_map_evolution',
    'plot_sharpe_vs_generation',
    'plot_cumulative_returns',
    'plot_category_distribution',
    'plot_insight_timeline',
    'plot_insight_evolution',
    'plot_3d_feature_map',
    'plot_dimension_pair_heatmap',
]
