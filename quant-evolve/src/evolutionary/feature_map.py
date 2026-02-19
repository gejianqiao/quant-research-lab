"""
Feature Map Data Structure for QuantEvolve.

Implements a multi-dimensional archive preserving strategy diversity across niches.
Uses binary encoding for strategy categories and equally-spaced bins for continuous metrics.

Dimensions (6 total):
1. Strategy Category (8-bit binary string)
2. Trading Frequency (continuous, binned)
3. Max Drawdown (continuous, binned)
4. Sharpe Ratio (continuous, binned)
5. Sortino Ratio (continuous, binned)
6. Total Return (continuous, binned)
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a trading strategy in the evolutionary database."""
    id: str
    hypothesis: Dict[str, Any]
    code: str
    metrics: Dict[str, float]
    analysis: Dict[str, Any]
    feature_vector: List[Any]
    generation: int
    island_id: int
    score: float
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        return {
            'id': self.id,
            'hypothesis': self.hypothesis,
            'code': self.code,
            'metrics': self.metrics,
            'analysis': self.analysis,
            'feature_vector': self.feature_vector,
            'generation': self.generation,
            'island_id': self.island_id,
            'score': self.score,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create Strategy from dictionary."""
        return cls(
            id=data['id'],
            hypothesis=data['hypothesis'],
            code=data['code'],
            metrics=data['metrics'],
            analysis=data['analysis'],
            feature_vector=data['feature_vector'],
            generation=data['generation'],
            island_id=data['island_id'],
            score=data['score'],
            timestamp=data.get('timestamp', '')
        )


class FeatureMap:
    """
    Multi-dimensional grid archive for preserving strategy diversity.
    
    Implements quality-diversity optimization by maintaining a grid of cells,
    where each cell corresponds to a unique combination of feature bins.
    When a new strategy arrives, it is placed in the corresponding cell.
    If the cell is occupied, the strategy with the higher score is kept.
    """
    
    # Strategy category names for binary encoding (8 families)
    STRATEGY_CATEGORIES = [
        'momentum',           # bit 0
        'mean_reversion',     # bit 1
        'trend_following',    # bit 2
        'breakout',           # bit 3
        'statistical_arb',    # bit 4
        'machine_learning',   # bit 5
        'sentiment_analysis', # bit 6
        'multi_factor'        # bit 7
    ]
    
    def __init__(
        self,
        dimensions: int = 6,
        bin_sizes: List[int] = None,
        category_bits: int = 8,
        min_values: Dict[str, float] = None,
        max_values: Dict[str, float] = None
    ):
        """
        Initialize the feature map.
        
        Args:
            dimensions: Number of feature dimensions (default 6)
            bin_sizes: Number of bins per dimension (default [256, 16, 16, 16, 16, 16])
                      First dimension is binary (2^category_bits), others are continuous
            category_bits: Number of bits for strategy category encoding (default 8)
            min_values: Minimum values for continuous dimensions
            max_values: Maximum values for continuous dimensions
        """
        self.dimensions = dimensions
        self.category_bits = category_bits
        
        # Default bin sizes: 256 for category (2^8), 16 for each continuous dimension
        if bin_sizes is None:
            self.bin_sizes = [2 ** category_bits] + [16] * (dimensions - 1)
        else:
            self.bin_sizes = bin_sizes
        
        # Default ranges for continuous dimensions
        # Format: {dimension_name: (min, max)}
        self.dimension_names = [
            'strategy_category',
            'trading_frequency',
            'max_drawdown',
            'sharpe_ratio',
            'sortino_ratio',
            'total_return'
        ]
        
        # Default min/max values for continuous dimensions
        self.min_values = min_values or {
            'trading_frequency': 0.0,
            'max_drawdown': -1.0,
            'sharpe_ratio': -5.0,
            'sortino_ratio': -5.0,
            'total_return': -1.0
        }
        
        self.max_values = max_values or {
            'trading_frequency': 1.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 10.0,
            'sortino_ratio': 10.0,
            'total_return': 10.0
        }
        
        # Main archive: maps feature vector tuple -> Strategy
        self.archive: Dict[Tuple, Strategy] = {}
        
        # Archive of all strategies ever added (including rejected ones)
        self.all_strategies: List[Strategy] = []
        
        # Statistics tracking
        self.total_updates = 0
        self.successful_additions = 0
        self.replacements = 0
        
        logger.info(f"Initialized FeatureMap with {dimensions} dimensions, "
                   f"bin sizes: {self.bin_sizes}")

    @property
    def total_cells(self) -> int:
        """Total number of cells in the feature map grid."""
        return int(np.prod(self.bin_sizes))

    def get_archived_strategy_ids(self) -> List[str]:
        """Return strategy IDs currently occupying feature-map cells."""
        return [strategy.id for strategy in self.archive.values()]
    
    def _category_to_index(self, category_bits: str) -> int:
        """
        Convert binary category string to integer index.
        
        Args:
            category_bits: 8-bit binary string (e.g., "00000001")
            
        Returns:
            Integer index (0-255)
        """
        return int(category_bits, 2)
    
    def _index_to_category(self, index: int) -> str:
        """
        Convert integer index to binary category string.
        
        Args:
            index: Integer index (0-255)
            
        Returns:
            8-bit binary string
        """
        return format(index, f'0{self.category_bits}b')
    
    def _bin_continuous_value(
        self,
        value: float,
        dimension_name: str,
        num_bins: int = 16
    ) -> int:
        """
        Bin a continuous value into one of num_bins equally-spaced bins.
        
        Args:
            value: Continuous value to bin
            dimension_name: Name of the dimension (for min/max lookup)
            num_bins: Number of bins (default 16)
            
        Returns:
            Bin index (0 to num_bins-1)
        """
        min_val = self.min_values.get(dimension_name, 0.0)
        max_val = self.max_values.get(dimension_name, 1.0)
        
        # Handle edge cases
        if max_val <= min_val:
            logger.warning(f"Invalid range for {dimension_name}: [{min_val}, {max_val}]")
            return 0
        
        # Clamp value to range
        value = max(min_val, min(max_val, value))
        
        # Linear interpolation to bin index
        normalized = (value - min_val) / (max_val - min_val)
        bin_index = int(normalized * num_bins)
        
        # Ensure index is within bounds
        bin_index = max(0, min(num_bins - 1, bin_index))
        
        return bin_index
    
    def _unbin_continuous_value(
        self,
        bin_index: int,
        dimension_name: str,
        num_bins: int = 16
    ) -> float:
        """
        Convert bin index back to representative continuous value.
        
        Args:
            bin_index: Bin index (0 to num_bins-1)
            dimension_name: Name of the dimension
            num_bins: Number of bins
            
        Returns:
            Representative value (center of bin)
        """
        min_val = self.min_values.get(dimension_name, 0.0)
        max_val = self.max_values.get(dimension_name, 1.0)
        
        if max_val <= min_val:
            return min_val
        
        # Return center of bin
        bin_width = (max_val - min_val) / num_bins
        return min_val + (bin_index + 0.5) * bin_width
    
    def compute_feature_vector(
        self,
        metrics: Dict[str, float],
        category_bits: str = None
    ) -> Tuple:
        """
        Compute feature vector from strategy metrics.
        
        Args:
            metrics: Dictionary containing strategy metrics
                    (sharpe_ratio, sortino_ratio, max_drawdown, total_return, etc.)
            category_bits: 8-bit binary string for strategy category
                          (if None, inferred from metrics or set to default)
        
        Returns:
            Tuple of bin indices: (category_idx, freq_idx, mdd_idx, sr_idx, sor_idx, ret_idx)
        """
        # Extract or compute category bits
        if category_bits is None:
            # Default to all zeros if not provided
            category_bits = '0' * self.category_bits
            logger.debug("No category bits provided, using default")
        
        # Bin each dimension
        feature_vector = []
        
        # 1. Strategy Category (binary encoding)
        category_idx = self._category_to_index(category_bits)
        feature_vector.append(category_idx)
        
        # 2. Trading Frequency
        trading_freq = metrics.get('trading_frequency', 0.5)
        freq_idx = self._bin_continuous_value(trading_freq, 'trading_frequency')
        feature_vector.append(freq_idx)
        
        # 3. Max Drawdown
        max_dd = metrics.get('max_drawdown', 0.0)
        mdd_idx = self._bin_continuous_value(max_dd, 'max_drawdown')
        feature_vector.append(mdd_idx)
        
        # 4. Sharpe Ratio
        sharpe = metrics.get('sharpe_ratio', 0.0)
        sr_idx = self._bin_continuous_value(sharpe, 'sharpe_ratio')
        feature_vector.append(sr_idx)
        
        # 5. Sortino Ratio
        sortino = metrics.get('sortino_ratio', 0.0)
        sor_idx = self._bin_continuous_value(sortino, 'sortino_ratio')
        feature_vector.append(sor_idx)
        
        # 6. Total Return
        total_ret = metrics.get('total_return', 0.0)
        ret_idx = self._bin_continuous_value(total_ret, 'total_return')
        feature_vector.append(ret_idx)
        
        return tuple(feature_vector)
    
    def update(
        self,
        strategy: Strategy,
        feature_vector: Tuple = None,
        island_id: int = 0
    ) -> bool:
        """
        Update the feature map with a new strategy.
        
        If the cell is empty, add the strategy.
        If the cell is occupied, compare scores and keep the better one.
        
        Args:
            strategy: Strategy object to add
            feature_vector: Pre-computed feature vector (if None, computed from metrics)
            island_id: ID of the island this strategy belongs to
            
        Returns:
            True if strategy was added/replaced, False if rejected
        """
        self.total_updates += 1
        
        # Compute feature vector if not provided
        if feature_vector is None:
            # Extract category bits from strategy analysis if available
            category_bits = strategy.analysis.get('category_bits', '0' * self.category_bits)
            feature_vector = self.compute_feature_vector(strategy.metrics, category_bits)
        
        # Update strategy's feature vector and island ID
        strategy.feature_vector = list(feature_vector)
        strategy.island_id = island_id
        
        # Check if cell is occupied
        existing_strategy = self.archive.get(feature_vector)
        
        if existing_strategy is None:
            # Cell is empty, add new strategy
            self.archive[feature_vector] = strategy
            self.all_strategies.append(strategy)
            self.successful_additions += 1
            logger.debug(f"Added new strategy {strategy.id} to cell {feature_vector}")
            return True
        else:
            # Cell is occupied, compare scores
            # Score = SR + IR + MDD (from Equation 3)
            new_score = strategy.score
            old_score = existing_strategy.score
            
            if new_score > old_score:
                # New strategy is better, replace
                self.archive[feature_vector] = strategy
                self.all_strategies.append(strategy)
                self.replacements += 1
                logger.debug(f"Replaced strategy {existing_strategy.id} (score={old_score:.3f}) "
                           f"with {strategy.id} (score={new_score:.3f}) in cell {feature_vector}")
                return True
            else:
                # Existing strategy is better, archive new one but don't replace
                self.all_strategies.append(strategy)
                logger.debug(f"Rejected strategy {strategy.id} (score={new_score:.3f}) "
                           f"in favor of {existing_strategy.id} (score={old_score:.3f})")
                return False
    
    def get_strategy_with_feature(self, feature_vector: Tuple) -> Optional[Strategy]:
        """
        Retrieve strategy from a specific feature cell.
        
        Args:
            feature_vector: Tuple of bin indices
            
        Returns:
            Strategy object if cell is occupied, None otherwise
        """
        return self.archive.get(feature_vector)
    
    def get_strategies_in_cell(
        self,
        feature_vector: Tuple
    ) -> List[Strategy]:
        """
        Get all strategies (current and historical) that mapped to a cell.
        
        Args:
            feature_vector: Tuple of bin indices
            
        Returns:
            List of Strategy objects
        """
        return [s for s in self.all_strategies 
                if tuple(s.feature_vector) == feature_vector]
    
    def get_occupied_cells(self) -> List[Tuple]:
        """
        Get list of all occupied cell indices.
        
        Returns:
            List of feature vector tuples
        """
        return list(self.archive.keys())
    
    def get_occupancy_rate(self) -> float:
        """
        Calculate the fraction of cells that are occupied.
        
        Returns:
            Occupancy rate (0.0 to 1.0)
        """
        occupied_cells = len(self.archive)
        return occupied_cells / self.total_cells if self.total_cells > 0 else 0.0
    
    def get_diversity_score(self) -> float:
        """
        Calculate diversity score based on occupied cells.
        
        Returns:
            Diversity score (higher = more diverse)
        """
        # Simple metric: number of unique category bins occupied
        occupied_categories = set()
        for feature_vector in self.archive.keys():
            occupied_categories.add(feature_vector[0])  # First dimension is category
        
        max_categories = self.bin_sizes[0]
        return len(occupied_categories) / max_categories if max_categories > 0 else 0.0
    
    def get_best_strategy(self) -> Optional[Strategy]:
        """
        Get the strategy with the highest score in the archive.
        
        Returns:
            Best Strategy object, or None if archive is empty
        """
        if not self.archive:
            return None
        
        return max(self.archive.values(), key=lambda s: s.score)
    
    def get_strategies_by_category(self, category_name: str) -> List[Strategy]:
        """
        Get all strategies belonging to a specific category.
        
        Args:
            category_name: Name of strategy category (e.g., 'momentum')
            
        Returns:
            List of Strategy objects in that category
        """
        if category_name not in self.STRATEGY_CATEGORIES:
            logger.warning(f"Unknown category: {category_name}")
            return []
        
        category_idx = self.STRATEGY_CATEGORIES.index(category_name)
        category_bit = 1 << category_idx
        
        strategies = []
        for strategy in self.archive.values():
            # Check if the corresponding bit is set
            category_int = strategy.feature_vector[0]
            if category_int & category_bit:
                strategies.append(strategy)
        
        return strategies
    
    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of strategies across categories.
        
        Returns:
            Dictionary mapping category names to counts
        """
        distribution = {cat: 0 for cat in self.STRATEGY_CATEGORIES}
        
        for strategy in self.archive.values():
            category_int = strategy.feature_vector[0]
            for i, category in enumerate(self.STRATEGY_CATEGORIES):
                if category_int & (1 << i):
                    distribution[category] += 1
        
        return distribution
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current feature map state.
        
        Returns:
            Dictionary containing archive state and statistics
        """
        return {
            'occupied_cells': len(self.archive),
            'total_cells': self.total_cells,
            'occupancy_rate': self.get_occupancy_rate(),
            'diversity_score': self.get_diversity_score(),
            'total_strategies': len(self.all_strategies),
            'total_updates': self.total_updates,
            'successful_additions': self.successful_additions,
            'replacements': self.replacements,
            'category_distribution': self.get_category_distribution(),
            'best_score': self.get_best_strategy().score if self.get_best_strategy() else None,
            'archive': {str(k): v.to_dict() for k, v in self.archive.items()}
        }
    
    def clear(self):
        """Clear the feature map archive."""
        self.archive.clear()
        self.all_strategies.clear()
        self.total_updates = 0
        self.successful_additions = 0
        self.replacements = 0
        logger.info("Feature map cleared")


def bit_flip(binary_string: str, k_bf: int) -> str:
    """
    Flip k_bf random bits in a binary string (with replacement).
    
    This allows bits to be flipped multiple times, potentially canceling out.
    Used for perturbing strategy category during cousin selection.
    
    Args:
        binary_string: Binary string (e.g., "00000001")
        k_bf: Number of bit flips to perform
        
    Returns:
        Modified binary string
    """
    if not binary_string:
        return binary_string
    
    bits = list(binary_string)
    n = len(bits)
    
    for _ in range(k_bf):
        # Select random bit position (with replacement)
        pos = random.randint(0, n - 1)
        # Flip the bit
        bits[pos] = '1' if bits[pos] == '0' else '0'
    
    return ''.join(bits)


def perturb_feature_vector(
    feature_vector: List[Any],
    sigma_d: float = 1.0,
    k_bf_ratio: float = 0.25,
    category_bits: int = 8
) -> Tuple:
    """
    Perturb a feature vector to generate a diverse cousin.
    
    For continuous dimensions: Add Gaussian noise N(0, sigma_d^2)
    For category dimension: Flip k_bf = n/4 bits (where n = bit length)
    
    Args:
        feature_vector: Original feature vector as list of bin indices
        sigma_d: Standard deviation for Gaussian perturbation (default 1.0)
        k_bf_ratio: Ratio of bits to flip for category (default 0.25 = n/4)
        category_bits: Number of bits in category encoding (default 8)
        
    Returns:
        Perturbed feature vector as tuple
    """
    if not feature_vector:
        return tuple(feature_vector)
    
    perturbed = []
    
    for i, value in enumerate(feature_vector):
        if i == 0:
            # First dimension is strategy category (binary encoding)
            # Convert to binary string, flip bits, convert back
            binary_str = format(int(value), f'0{category_bits}b')
            k_bf = max(1, int(category_bits * k_bf_ratio))
            flipped_binary = bit_flip(binary_str, k_bf)
            perturbed.append(int(flipped_binary, 2))
        else:
            # Continuous dimensions: add Gaussian noise
            noise = np.random.normal(0, sigma_d)
            new_value = int(np.floor(value + noise))
            # Clamp to valid range (assuming 16 bins for continuous dimensions)
            new_value = max(0, min(15, new_value))
            perturbed.append(new_value)
    
    return tuple(perturbed)


# Convenience function for creating feature map from config
def create_feature_map_from_config(config: Dict[str, Any]) -> FeatureMap:
    """
    Create a FeatureMap instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'feature_map' section
        
    Returns:
        Initialized FeatureMap object
    """
    fm_config = config.get('feature_map', {})
    
    dimensions = fm_config.get('dimensions', 6)
    bins_per_dimension = fm_config.get('bins_per_dimension', 16)
    category_bits = fm_config.get('category_bits', 8)
    
    # Create bin sizes list
    bin_sizes = [2 ** category_bits] + [bins_per_dimension] * (dimensions - 1)
    
    # Get custom min/max values if specified
    min_values = fm_config.get('min_values', None)
    max_values = fm_config.get('max_values', None)
    
    return FeatureMap(
        dimensions=dimensions,
        bin_sizes=bin_sizes,
        category_bits=category_bits,
        min_values=min_values,
        max_values=max_values
    )


# Backward-compatible module-level export used by package imports.
STRATEGY_CATEGORIES = FeatureMap.STRATEGY_CATEGORIES
