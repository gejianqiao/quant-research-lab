"""
Selection mechanisms for the QuantEvolve evolutionary algorithm.

This module implements:
- Parent Selection (Equation 1): Balances exploitation (elite) vs exploration (diverse)
- Cousin Selection (Equation 2): Generates diverse cousins via feature perturbation
- Bit flip operations for category dimension mutation
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from logging import getLogger

from .feature_map import FeatureMap, Strategy, bit_flip, perturb_feature_vector

logger = getLogger(__name__)


def sample_parent(
    island: List[Strategy],
    feature_map: FeatureMap,
    alpha: float = 0.5,
    rng: Optional[random.Random] = None
) -> Strategy:
    """
    Sample a parent strategy from an island using Equation 1.
    
    Formula: P(s_p = s) = α/|M_I| if s ∈ M_I (elite), (1-α)/|I| if s ∈ I (diverse)
    
    Args:
        island: List of strategies in the current island
        feature_map: Feature map containing elite strategies (archived)
        alpha: Balance parameter (0.0 = full exploration, 1.0 = full exploitation)
        rng: Optional random number generator for reproducibility
    
    Returns:
        Selected parent strategy
    
    Raises:
        ValueError: If island is empty
    """
    if rng is None:
        rng = random
    
    if not island:
        raise ValueError("Cannot sample parent from empty island")
    
    # Get elite strategies (those in the feature map archive)
    if hasattr(feature_map, 'get_archived_strategy_ids'):
        elite_strategy_ids = set(feature_map.get_archived_strategy_ids())
    else:
        elite_strategy_ids = set(getattr(feature_map, 'strategy_ids', []))
    elite_strategies = [s for s in island if s.id in elite_strategy_ids]
    
    # Decide whether to sample from elite or diverse pool
    if elite_strategies and rng.random() < alpha:
        # Exploitation: Sample from elite strategies
        parent = rng.choice(elite_strategies)
        logger.debug(f"Selected elite parent: {parent.id} (score={parent.score:.4f})")
    else:
        # Exploration: Sample from entire island
        parent = rng.choice(island)
        logger.debug(f"Selected diverse parent: {parent.id} (score={parent.score:.4f})")
    
    return parent


def sample_cousins(
    parent: Strategy,
    island: List[Strategy],
    feature_map: FeatureMap,
    config: Dict[str, Any],
    rng: Optional[random.Random] = None
) -> List[Strategy]:
    """
    Sample cousin strategies using Equation 2.
    
    Configuration: 2 best cousins + 3 diverse cousins + 2 random cousins = 7 total
    
    For diverse cousins (Eq 2):
    - Continuous dimension d: f_d^c = floor(N(f_d^p, σ_d²)) where σ_d = 1.0
    - Category dimension: f_d^c = BitFlip(f_d^p, k_bf) where k_bf = n/4
    
    Args:
        parent: Parent strategy to generate cousins from
        island: List of strategies in the current island
        feature_map: Feature map for looking up strategies by feature vector
        config: Configuration dictionary with cousin selection parameters
        rng: Optional random number generator for reproducibility
    
    Returns:
        List of up to 7 cousin strategies
    """
    if rng is None:
        rng = random
    
    cousins = []
    parent_id = parent.id
    
    # Get cousin selection parameters from config
    # Accept either the full config or a cousin_selection subsection.
    cousin_config = config.get('cousin_selection', config)
    num_best = cousin_config.get('num_best_cousins', 2)
    num_diverse = cousin_config.get('num_diverse_cousins', 3)
    num_random = cousin_config.get('num_random_cousins', 2)
    perturbation_sigma = cousin_config.get('perturbation_sigma', 1.0)
    bit_flip_ratio = cousin_config.get('bit_flip_ratio', 0.25)
    
    # Filter out parent from island for cousin selection
    island_without_parent = [s for s in island if s.id != parent_id]
    
    if not island_without_parent:
        logger.warning("Island has no strategies other than parent, returning empty cousins")
        return cousins
    
    # 1. Best cousins: Top performers excluding parent
    sorted_island = sorted(island_without_parent, key=lambda s: s.score, reverse=True)
    best_cousins = sorted_island[:num_best]
    cousins.extend(best_cousins)
    logger.debug(f"Selected {len(best_cousins)} best cousins")
    
    # 2. Diverse cousins: Generate via feature perturbation
    diverse_cousins = []
    for i in range(num_diverse):
        # Perturb parent's feature vector
        perturbed_vector = perturb_feature_vector(
            parent.feature_vector,
            sigma_d=perturbation_sigma,
            k_bf_ratio=bit_flip_ratio,
            category_bits=feature_map.category_bits,
        )
        
        # Try to find a strategy with the perturbed feature vector
        cousin = feature_map.get_strategy_with_feature(perturbed_vector)
        
        if cousin and cousin.id != parent_id:
            diverse_cousins.append(cousin)
        else:
            # Fallback: Find closest matching strategy in island
            closest = _find_closest_strategy_by_feature(
                perturbed_vector,
                island_without_parent,
                feature_map
            )
            if closest:
                diverse_cousins.append(closest)
    
    cousins.extend(diverse_cousins)
    logger.debug(f"Selected {len(diverse_cousins)} diverse cousins")
    
    # 3. Random cousins: Uniform sampling from island
    available_for_random = [s for s in island_without_parent if s not in cousins]
    if available_for_random:
        num_random_to_select = min(num_random, len(available_for_random))
        random_cousins = rng.sample(available_for_random, num_random_to_select)
        cousins.extend(random_cousins)
        logger.debug(f"Selected {len(random_cousins)} random cousins")
    
    # Limit to maximum number of cousins
    max_cousins = num_best + num_diverse + num_random
    if len(cousins) > max_cousins:
        cousins = cousins[:max_cousins]
    
    logger.info(f"Sampled {len(cousins)} total cousins for parent {parent.id}")
    return cousins


def _find_closest_strategy_by_feature(
    target_vector: List[Any],
    candidates: List[Strategy],
    feature_map: FeatureMap
) -> Optional[Strategy]:
    """
    Find the strategy with the closest feature vector to the target.
    
    Args:
        target_vector: Target feature vector to match
        candidates: List of candidate strategies
        feature_map: Feature map for distance calculation
    
    Returns:
        Closest matching strategy or None if no candidates
    """
    if not candidates:
        return None
    
    min_distance = float('inf')
    closest_strategy = None
    
    for candidate in candidates:
        distance = _compute_feature_distance(
            target_vector,
            candidate.feature_vector,
            feature_map
        )
        if distance < min_distance:
            min_distance = distance
            closest_strategy = candidate
    
    return closest_strategy


def _compute_feature_distance(
    vector1: List[Any],
    vector2: List[Any],
    feature_map: FeatureMap
) -> float:
    """
    Compute distance between two feature vectors.
    
    Handles mixed types: binary strings (Hamming distance) and continuous values (Euclidean).
    
    Args:
        vector1: First feature vector
        vector2: Second feature vector
        feature_map: Feature map for dimension metadata
    
    Returns:
        Distance value (lower = more similar)
    """
    if len(vector1) != len(vector2):
        logger.warning(f"Feature vector length mismatch: {len(vector1)} vs {len(vector2)}")
        return float('inf')
    
    distance = 0.0
    
    for i, (v1, v2) in enumerate(zip(vector1, vector2)):
        if i == 0:
            # Category dimension is an int index derived from bit-encoding.
            try:
                i1 = int(v1)
                i2 = int(v2)
                xor_bits = i1 ^ i2
                hamming = bin(xor_bits).count('1')
                distance += hamming / max(1, getattr(feature_map, 'category_bits', 8))
            except (ValueError, TypeError):
                distance += 1.0
        else:
            # Continuous: Normalized Euclidean distance
            try:
                diff = abs(float(v1) - float(v2))
                # Normalize by dimension range
                dim_name = (
                    feature_map.dimension_names[i]
                    if hasattr(feature_map, 'dimension_names') and i < len(feature_map.dimension_names)
                    else None
                )
                if (
                    dim_name
                    and hasattr(feature_map, 'max_values')
                    and hasattr(feature_map, 'min_values')
                    and dim_name in feature_map.max_values
                    and dim_name in feature_map.min_values
                ):
                    dim_range = feature_map.max_values[dim_name] - feature_map.min_values[dim_name]
                else:
                    dim_range = 15.0  # Default 16-bin span
                if dim_range > 0:
                    distance += (diff / dim_range) ** 2
                else:
                    distance += diff ** 2
            except (ValueError, TypeError):
                distance += 1.0
    
    return np.sqrt(distance)


def tournament_selection(
    island: List[Strategy],
    tournament_size: int = 3,
    rng: Optional[random.Random] = None
) -> Strategy:
    """
    Select a parent using tournament selection.
    
    Alternative to Equation 1 selection, useful for maintaining selection pressure.
    
    Args:
        island: List of strategies in the current island
        tournament_size: Number of strategies to compete in tournament
        rng: Optional random number generator
    
    Returns:
        Winner of the tournament (highest score among sampled)
    """
    if rng is None:
        rng = random
    
    if not island:
        raise ValueError("Cannot perform tournament selection on empty island")
    
    tournament_size = min(tournament_size, len(island))
    tournament = rng.sample(island, tournament_size)
    
    winner = max(tournament, key=lambda s: s.score)
    logger.debug(f"Tournament selection: {tournament_size} competitors, winner {winner.id}")
    
    return winner


def create_selection_pool(
    parent: Strategy,
    cousins: List[Strategy],
    selection_method: str = 'uniform'
) -> List[Strategy]:
    """
    Create a pool of strategies for hypothesis generation context.
    
    Args:
        parent: Selected parent strategy
        cousins: List of cousin strategies
        selection_method: Method for ordering the pool ('uniform', 'by_score')
    
    Returns:
        Ordered list of strategies for context
    """
    pool = [parent] + cousins
    
    if selection_method == 'by_score':
        pool = sorted(pool, key=lambda s: s.score, reverse=True)
    
    return pool


def validate_selection_config(config: Dict[str, Any]) -> bool:
    """
    Validate cousin selection configuration parameters.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if configuration is valid, False otherwise
    """
    cousin_config = config.get('cousin_selection', {})
    
    # Check required parameters
    required_params = ['num_best_cousins', 'num_diverse_cousins', 'num_random_cousins']
    for param in required_params:
        if param not in cousin_config:
            logger.warning(f"Missing cousin selection parameter: {param}")
            return False
    
    # Validate ranges
    total_cousins = (
        cousin_config.get('num_best_cousins', 0) +
        cousin_config.get('num_diverse_cousins', 0) +
        cousin_config.get('num_random_cousins', 0)
    )
    
    if total_cousins < 1:
        logger.warning("Total cousins must be at least 1")
        return False
    
    if total_cousins > 20:
        logger.warning(f"Large number of cousins ({total_cousins}) may slow down evolution")
    
    # Validate perturbation parameters
    sigma = cousin_config.get('perturbation_sigma', 1.0)
    if sigma <= 0:
        logger.warning("Perturbation sigma must be positive")
        return False
    
    bit_flip_ratio = cousin_config.get('bit_flip_ratio', 0.25)
    if not (0.0 <= bit_flip_ratio <= 1.0):
        logger.warning("Bit flip ratio must be between 0.0 and 1.0")
        return False
    
    return True
