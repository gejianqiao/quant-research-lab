"""
Evolutionary Algorithm Module for QuantEvolve.

This module implements the core evolutionary optimization framework,
including the feature map for quality-diversity optimization,
island model for parallel evolution, and selection mechanisms.
"""

from .feature_map import FeatureMap, Strategy, bit_flip, perturb_feature_vector, create_feature_map_from_config, STRATEGY_CATEGORIES
from .selection import sample_parent, sample_cousins, tournament_selection, create_selection_pool, validate_selection_config
from .island_manager import Island, IslandManager, create_island_manager_from_config
from .algorithm import QuantEvolve, EvolutionaryDatabase, create_quant_evolve_from_config, run_evolution

__all__ = [
    # Feature Map
    'FeatureMap',
    'Strategy',
    'bit_flip',
    'perturb_feature_vector',
    'create_feature_map_from_config',
    'STRATEGY_CATEGORIES',
    
    # Selection
    'sample_parent',
    'sample_cousins',
    'tournament_selection',
    'create_selection_pool',
    'validate_selection_config',
    
    # Island Manager
    'Island',
    'IslandManager',
    'create_island_manager_from_config',
    
    # Algorithm
    'QuantEvolve',
    'EvolutionaryDatabase',
    'create_quant_evolve_from_config',
    'run_evolution',
]
