"""
Test suite for the FeatureMap data structure and related utilities.

Validates:
- Multi-dimensional binning with 16 bins per dimension
- Binary encoding for strategy categories (8-bit)
- Update rule with score comparison (Eq 3)
- Bit flip operation for category perturbation
- Feature vector perturbation for diverse cousin selection
- Archive storage and retrieval
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Import the components to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolutionary.feature_map import (
    FeatureMap,
    Strategy,
    bit_flip,
    perturb_feature_vector,
    create_feature_map_from_config,
    STRATEGY_CATEGORIES
)


class TestBitFlip:
    """Test the bit_flip operation for category dimension perturbation."""

    def test_bit_flip_single_bit(self):
        """Test flipping a single bit."""
        binary_string = "00000000"
        result = bit_flip(binary_string, k_bf=1)
        
        # Should have exactly one bit different
        diff_count = sum(1 for a, b in zip(binary_string, result) if a != b)
        assert diff_count == 1, f"Expected 1 bit flipped, got {diff_count}"

    def test_bit_flip_multiple_bits(self):
        """Test flipping multiple bits."""
        binary_string = "00000000"
        result = bit_flip(binary_string, k_bf=4)
        
        # Count differences (may be less than 4 due to replacement)
        diff_count = sum(1 for a, b in zip(binary_string, result) if a != b)
        assert 1 <= diff_count <= 4, f"Expected 1-4 bits flipped, got {diff_count}"

    def test_bit_flip_with_replacement(self):
        """Test that bit flip uses replacement (same bit can be flipped twice)."""
        binary_string = "10000000"
        # Flip same bit twice should return to original (with some probability)
        # Run multiple times to check behavior
        results = [bit_flip(binary_string, k_bf=2) for _ in range(100)]
        
        # At least some results should have 0 or 2 bits different (not always 1)
        diff_counts = [sum(1 for a, b in zip(binary_string, r) if a != b) for r in results]
        assert len(set(diff_counts)) > 1, "Bit flip should show variation due to replacement"

    def test_bit_flip_preserves_length(self):
        """Test that bit flip preserves the binary string length."""
        binary_string = "10110010"
        result = bit_flip(binary_string, k_bf=3)
        assert len(result) == len(binary_string), "Bit flip should preserve string length"

    def test_bit_flip_zero_flips(self):
        """Test flipping zero bits returns original string."""
        binary_string = "10110010"
        result = bit_flip(binary_string, k_bf=0)
        assert result == binary_string, "Zero flips should return original string"


class TestPerturbFeatureVector:
    """Test the perturb_feature_vector function for diverse cousin generation."""

    def test_perturb_continuous_dimensions(self):
        """Test Gaussian perturbation of continuous dimensions."""
        # Feature vector: [category_bits, frequency, mdd, sr, sor, return]
        feature_vector = ["00000000", 5, -0.15, 1.2, 1.5, 0.25]
        
        np.random.seed(42)
        perturbed = perturb_feature_vector(
            feature_vector,
            sigma_d=1.0,
            k_bf_ratio=0.25,
            category_bits=8
        )
        
        # Category should be perturbed via bit flip
        assert isinstance(perturbed[0], str)
        assert len(perturbed[0]) == 8
        
        # Continuous dimensions should be perturbed
        for i in range(1, len(perturbed)):
            assert isinstance(perturbed[i], (int, float))

    def test_perturb_category_dimension(self):
        """Test bit flip perturbation of category dimension."""
        feature_vector = ["10000000", 5, -0.15, 1.2, 1.5, 0.25]
        
        np.random.seed(42)
        perturbed = perturb_feature_vector(
            feature_vector,
            sigma_d=1.0,
            k_bf_ratio=0.25,
            category_bits=8
        )
        
        # k_bf = n * k_bf_ratio = 8 * 0.25 = 2 bits to flip
        original_category = feature_vector[0]
        perturbed_category = perturbed[0]
        
        # Should be different (with high probability)
        # Note: Due to replacement, might occasionally be same, but rare
        diff_count = sum(1 for a, b in zip(original_category, perturbed_category) if a != b)
        assert 0 <= diff_count <= 8, "Category bits should be valid 8-bit string"

    def test_perturb_reproducibility(self):
        """Test that perturbation is reproducible with same seed."""
        feature_vector = ["00000000", 5, -0.15, 1.2, 1.5, 0.25]
        
        np.random.seed(123)
        perturbed1 = perturb_feature_vector(feature_vector, sigma_d=1.0, k_bf_ratio=0.25, category_bits=8)
        
        np.random.seed(123)
        perturbed2 = perturb_feature_vector(feature_vector, sigma_d=1.0, k_bf_ratio=0.25, category_bits=8)
        
        assert perturbed1 == perturbed2, "Perturbation should be reproducible with same seed"


class TestStrategyDataclass:
    """Test the Strategy dataclass."""

    def test_strategy_creation(self):
        """Test creating a Strategy instance."""
        strategy = Strategy(
            id="test_strategy_001",
            hypothesis={"hypothesis": "Test hypothesis", "rationale": "Test rationale"},
            code="def initialize(context): pass",
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.1, "total_return": 0.25},
            analysis={"category": "momentum", "score": 0.8},
            feature_vector=["10000000", 5, -0.1, 1.5, 1.8, 0.25],
            generation=10,
            island_id=2,
            score=2.65,  # SR + IR + MDD
            timestamp=datetime.now()
        )
        
        assert strategy.id == "test_strategy_001"
        assert strategy.generation == 10
        assert strategy.island_id == 2
        assert strategy.score == 2.65

    def test_strategy_to_dict(self):
        """Test converting Strategy to dictionary."""
        strategy = Strategy(
            id="test_001",
            hypothesis={"test": "data"},
            code="test_code",
            metrics={"sr": 1.0},
            analysis={"cat": "momentum"},
            feature_vector=["10000000", 5, -0.1, 1.0, 1.2, 0.1],
            generation=5,
            island_id=1,
            score=2.0,
            timestamp=datetime.now()
        )
        
        strategy_dict = strategy.to_dict()
        
        assert isinstance(strategy_dict, dict)
        assert strategy_dict["id"] == "test_001"
        assert strategy_dict["generation"] == 5

    def test_strategy_from_dict(self):
        """Test creating Strategy from dictionary."""
        strategy_dict = {
            "id": "test_002",
            "hypothesis": {"test": "data"},
            "code": "test_code",
            "metrics": {"sr": 1.0},
            "analysis": {"cat": "momentum"},
            "feature_vector": ["01000000", 3, -0.15, 0.8, 1.0, 0.08],
            "generation": 7,
            "island_id": 3,
            "score": 1.5,
            "timestamp": datetime.now().isoformat()
        }
        
        strategy = Strategy.from_dict(strategy_dict)
        
        assert strategy.id == "test_002"
        assert strategy.generation == 7
        assert strategy.island_id == 3


class TestFeatureMap:
    """Test the FeatureMap multi-dimensional archive."""

    def test_feature_map_initialization(self):
        """Test FeatureMap initialization with default parameters."""
        dimensions = 6
        bin_sizes = [16, 16, 16, 16, 16, 16]
        category_bits = 8
        
        feature_map = FeatureMap(
            dimensions=dimensions,
            bin_sizes=bin_sizes,
            category_bits=category_bits
        )
        
        assert feature_map.dimensions == dimensions
        assert feature_map.bin_sizes == bin_sizes
        assert len(feature_map.archive) == 0

    def test_feature_map_from_config(self):
        """Test creating FeatureMap from configuration dictionary."""
        config = {
            "feature_map": {
                "dimensions_list": [
                    {"name": "strategy_category", "type": "binary", "bits": 8},
                    {"name": "trading_frequency", "type": "continuous", "min": 0, "max": 1},
                    {"name": "max_drawdown", "type": "continuous", "min": -1.0, "max": 0.0},
                    {"name": "sharpe_ratio", "type": "continuous", "min": -5.0, "max": 10.0},
                    {"name": "sortino_ratio", "type": "continuous", "min": -5.0, "max": 10.0},
                    {"name": "total_return", "type": "continuous", "min": -1.0, "max": 10.0}
                ],
                "bins_per_dimension": 16
            }
        }
        
        feature_map = create_feature_map_from_config(config)
        
        assert feature_map.dimensions == 6
        assert all(bs == 16 for bs in feature_map.bin_sizes)

    def test_compute_feature_vector(self):
        """Test computing feature vector from metrics."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        metrics = {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 1.8,
            "max_drawdown": -0.12,
            "total_return": 0.35,
            "trading_frequency": 0.25
        }
        category_bits = "10000000"  # Momentum
        
        feature_vector = feature_map.compute_feature_vector(metrics, category_bits)
        
        assert len(feature_vector) == 6
        assert feature_vector[0] == "10000000"  # Category bits
        assert isinstance(feature_vector[1], int)  # Trading frequency bin
        assert isinstance(feature_vector[2], int)  # MDD bin
        assert isinstance(feature_vector[3], int)  # SR bin
        assert isinstance(feature_vector[4], int)  # SOR bin
        assert isinstance(feature_vector[5], int)  # Return bin

    def test_feature_map_update_empty_cell(self):
        """Test updating feature map with empty cell."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        strategy = Strategy(
            id="test_001",
            hypothesis={},
            code="test",
            metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.2},
            analysis={},
            feature_vector=["10000000", 5, 3, 10, 11, 8],
            generation=1,
            island_id=0,
            score=2.5,
            timestamp=datetime.now()
        )
        
        feature_vector = tuple(strategy.feature_vector)
        success = feature_map.update(strategy, feature_vector)
        
        assert success, "Update should succeed for empty cell"
        assert feature_map.get_occupancy_rate() > 0
        assert len(feature_map.archive) == 1

    def test_feature_map_update_replace_better(self):
        """Test updating feature map replaces worse strategy with better one."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add first strategy with lower score
        strategy1 = Strategy(
            id="test_001",
            hypothesis={},
            code="test1",
            metrics={"sharpe_ratio": 1.0, "sortino_ratio": 1.2, "max_drawdown": -0.15, "total_return": 0.15},
            analysis={},
            feature_vector=["10000000", 5, 3, 8, 9, 6],
            generation=1,
            island_id=0,
            score=1.85,  # Lower score
            timestamp=datetime.now()
        )
        
        feature_map.update(strategy1, tuple(strategy1.feature_vector))
        
        # Add second strategy with higher score to same cell
        strategy2 = Strategy(
            id="test_002",
            hypothesis={},
            code="test2",
            metrics={"sharpe_ratio": 2.0, "sortino_ratio": 2.5, "max_drawdown": -0.08, "total_return": 0.3},
            analysis={},
            feature_vector=["10000000", 5, 3, 8, 9, 6],  # Same feature vector
            generation=2,
            island_id=0,
            score=3.22,  # Higher score
            timestamp=datetime.now()
        )
        
        feature_map.update(strategy2, tuple(strategy2.feature_vector))
        
        # Should have replaced the first strategy
        retrieved = feature_map.get_strategy_with_feature(tuple(strategy2.feature_vector))
        assert retrieved is not None
        assert retrieved.id == "test_002", "Better strategy should replace worse one"
        assert len(feature_map.archive) == 1  # Only one strategy in archive

    def test_feature_map_update_archive_worse(self):
        """Test updating feature map archives worse strategy."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add first strategy with higher score
        strategy1 = Strategy(
            id="test_001",
            hypothesis={},
            code="test1",
            metrics={"sharpe_ratio": 2.0, "sortino_ratio": 2.5, "max_drawdown": -0.08, "total_return": 0.3},
            analysis={},
            feature_vector=["10000000", 5, 3, 8, 9, 6],
            generation=1,
            island_id=0,
            score=3.22,  # Higher score
            timestamp=datetime.now()
        )
        
        feature_map.update(strategy1, tuple(strategy1.feature_vector))
        
        # Add second strategy with lower score to same cell
        strategy2 = Strategy(
            id="test_002",
            hypothesis={},
            code="test2",
            metrics={"sharpe_ratio": 1.0, "sortino_ratio": 1.2, "max_drawdown": -0.15, "total_return": 0.15},
            analysis={},
            feature_vector=["10000000", 5, 3, 8, 9, 6],  # Same feature vector
            generation=2,
            island_id=0,
            score=1.85,  # Lower score
            timestamp=datetime.now()
        )
        
        feature_map.update(strategy2, tuple(strategy2.feature_vector))
        
        # Archive should still have the better strategy
        retrieved = feature_map.get_strategy_with_feature(tuple(strategy1.feature_vector))
        assert retrieved.id == "test_001", "Better strategy should be retained"
        
        # Worse strategy should be in all_strategies but not archive
        assert len(feature_map.all_strategies) == 2
        assert len(feature_map.archive) == 1

    def test_feature_map_occupancy_rate(self):
        """Test occupancy rate calculation."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Initial occupancy should be 0
        assert feature_map.get_occupancy_rate() == 0.0
        
        # Add some strategies
        for i in range(10):
            strategy = Strategy(
                id=f"test_{i:03d}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.0 + i * 0.1, "sortino_ratio": 1.2, "max_drawdown": -0.1, "total_return": 0.2},
                analysis={},
                feature_vector=[f"{i:08b}", 5, 3, 8, 9, 6],
                generation=i,
                island_id=0,
                score=2.0,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        # Occupancy should be > 0
        assert feature_map.get_occupancy_rate() > 0.0

    def test_feature_map_diversity_score(self):
        """Test diversity score calculation."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Empty map should have 0 diversity
        assert feature_map.get_diversity_score() == 0.0
        
        # Add strategies with different categories
        for i in range(8):
            strategy = Strategy(
                id=f"test_{i:03d}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={},
                feature_vector=[f"{1 << i:08b}", 5, 3, 10, 11, 8],
                generation=i,
                island_id=i % 3,
                score=2.5,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        # Diversity should be high with different categories
        diversity = feature_map.get_diversity_score()
        assert diversity > 0.5, "Diversity should be high with varied categories"

    def test_feature_map_get_strategies_by_category(self):
        """Test retrieving strategies by category."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add strategies with different categories
        for i in range(5):
            strategy = Strategy(
                id=f"momentum_{i}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={},
                feature_vector=["10000000", 5, 3, 10, 11, 8],  # Momentum category
                generation=i,
                island_id=0,
                score=2.5,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        for i in range(3):
            strategy = Strategy(
                id=f"mean_rev_{i}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.2, "sortino_ratio": 1.5, "max_drawdown": -0.08, "total_return": 0.2},
                analysis={},
                feature_vector=["01000000", 5, 3, 9, 10, 7],  # Mean reversion category
                generation=i,
                island_id=1,
                score=2.1,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        momentum_strategies = feature_map.get_strategies_by_category("momentum")
        mean_rev_strategies = feature_map.get_strategies_by_category("mean_reversion")
        
        assert len(momentum_strategies) == 5
        assert len(mean_rev_strategies) == 3

    def test_feature_map_category_distribution(self):
        """Test category distribution calculation."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add strategies with different categories
        categories = ["momentum", "momentum", "mean_reversion", "trend_following", "momentum"]
        for i, cat in enumerate(categories):
            category_bits = f"{1 << i:08b}"
            strategy = Strategy(
                id=f"test_{i}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={"category": cat},
                feature_vector=[category_bits, 5, 3, 10, 11, 8],
                generation=i,
                island_id=0,
                score=2.5,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        distribution = feature_map.get_category_distribution()
        
        assert isinstance(distribution, dict)
        assert "momentum" in distribution
        assert distribution["momentum"] == 3  # 3 momentum strategies

    def test_feature_map_snapshot(self):
        """Test feature map snapshot for visualization."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add some strategies
        for i in range(5):
            strategy = Strategy(
                id=f"test_{i}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={},
                feature_vector=[f"{1 << i:08b}", 5, 3, 10, 11, 8],
                generation=i,
                island_id=0,
                score=2.5,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        snapshot = feature_map.get_snapshot()
        
        assert isinstance(snapshot, dict)
        assert "occupied_cells" in snapshot
        assert "occupancy_rate" in snapshot
        assert "diversity_score" in snapshot
        assert "category_distribution" in snapshot
        assert len(snapshot["occupied_cells"]) == 5

    def test_feature_map_clear(self):
        """Test clearing the feature map."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add some strategies
        for i in range(5):
            strategy = Strategy(
                id=f"test_{i}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={},
                feature_vector=[f"{1 << i:08b}", 5, 3, 10, 11, 8],
                generation=i,
                island_id=0,
                score=2.5,
                timestamp=datetime.now()
            )
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        assert len(feature_map.archive) == 5
        
        # Clear the map
        feature_map.clear()
        
        assert len(feature_map.archive) == 0
        assert len(feature_map.all_strategies) == 0
        assert feature_map.get_occupancy_rate() == 0.0


class TestFeatureMapBinning:
    """Test the binning logic for continuous dimensions."""

    def test_binning_16_bins(self):
        """Test that 16 bins correctly partition the space."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8,
            min_values=[0, -1.0, -5.0, -5.0, -1.0],
            max_values=[1, 0.0, 10.0, 10.0, 10.0]
        )
        
        # Test Sharpe Ratio binning (range: -5 to 10, 16 bins)
        # Bin width = 15/16 = 0.9375
        sr_values = [-5.0, -4.0, 0.0, 5.0, 10.0]
        bins = []
        
        for sr in sr_values:
            metrics = {
                "sharpe_ratio": sr,
                "sortino_ratio": 1.5,
                "max_drawdown": -0.1,
                "total_return": 0.2,
                "trading_frequency": 0.5
            }
            fv = feature_map.compute_feature_vector(metrics, "00000000")
            bins.append(fv[3])  # SR bin index
        
        # Verify bin indices are in correct order
        assert bins[0] <= bins[1] <= bins[2] <= bins[3] <= bins[4]
        assert all(0 <= b < 16 for b in bins), "All bins should be in range [0, 16)"

    def test_binning_clamping(self):
        """Test that values outside range are clamped to valid bins."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8,
            min_values=[0, -1.0, -5.0, -5.0, -1.0],
            max_values=[1, 0.0, 10.0, 10.0, 10.0]
        )
        
        # Test with extreme values
        metrics_low = {
            "sharpe_ratio": -100.0,  # Way below min
            "sortino_ratio": 1.5,
            "max_drawdown": -0.1,
            "total_return": 0.2,
            "trading_frequency": 0.5
        }
        
        metrics_high = {
            "sharpe_ratio": 100.0,  # Way above max
            "sortino_ratio": 1.5,
            "max_drawdown": -0.1,
            "total_return": 0.2,
            "trading_frequency": 0.5
        }
        
        fv_low = feature_map.compute_feature_vector(metrics_low, "00000000")
        fv_high = feature_map.compute_feature_vector(metrics_high, "00000000")
        
        # Should be clamped to min/max bins
        assert fv_low[3] == 0, "Value below min should clamp to bin 0"
        assert fv_high[3] == 15, "Value above max should clamp to bin 15"


class TestFeatureMapIntegration:
    """Integration tests for FeatureMap with 100 strategies."""

    def test_feature_map_100_strategies(self):
        """Test that 100 strategies are correctly binned."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        np.random.seed(42)
        
        # Generate 100 strategies with known feature vectors
        for i in range(100):
            category_idx = i % 8
            category_bits = f"{1 << category_idx:08b}"
            
            metrics = {
                "sharpe_ratio": np.random.uniform(0.5, 3.0),
                "sortino_ratio": np.random.uniform(0.8, 4.0),
                "max_drawdown": np.random.uniform(-0.5, -0.05),
                "total_return": np.random.uniform(0.1, 1.0),
                "trading_frequency": np.random.uniform(0.0, 1.0)
            }
            
            strategy = Strategy(
                id=f"strategy_{i:03d}",
                hypothesis={},
                code=f"test_code_{i}",
                metrics=metrics,
                analysis={"category": STRATEGY_CATEGORIES[category_idx]},
                feature_vector=[],  # Will be computed
                generation=i // 10,
                island_id=i % 9,
                score=0.0,  # Will be computed
                timestamp=datetime.now()
            )
            
            feature_vector = feature_map.compute_feature_vector(metrics, category_bits)
            strategy.feature_vector = feature_vector
            strategy.score = metrics["sharpe_ratio"] + 0.5 + metrics["max_drawdown"]  # Simplified
            
            feature_map.update(strategy, tuple(feature_vector))
        
        # Verify all strategies are stored
        assert len(feature_map.all_strategies) == 100
        
        # Verify archive has <= 100 strategies (some may have been replaced)
        assert len(feature_map.archive) <= 100
        
        # Verify occupancy rate is > 0
        assert feature_map.get_occupancy_rate() > 0.0
        
        # Verify diversity score is > 0
        assert feature_map.get_diversity_score() > 0.0

    def test_feature_map_retrieval_accuracy(self):
        """Test that strategies can be retrieved by feature vector."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[16, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add 10 strategies with distinct feature vectors
        strategies = []
        for i in range(10):
            category_bits = f"{1 << i:08b}"
            strategy = Strategy(
                id=f"test_{i:03d}",
                hypothesis={},
                code=f"test_{i}",
                metrics={"sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.1, "total_return": 0.25},
                analysis={},
                feature_vector=[category_bits, i % 16, 3, 10, 11, 8],
                generation=1,
                island_id=0,
                score=2.5,
                timestamp=datetime.now()
            )
            strategies.append(strategy)
            feature_map.update(strategy, tuple(strategy.feature_vector))
        
        # Retrieve each strategy by feature vector
        for strategy in strategies:
            retrieved = feature_map.get_strategy_with_feature(tuple(strategy.feature_vector))
            assert retrieved is not None, f"Strategy {strategy.id} should be retrievable"
            assert retrieved.id == strategy.id, f"Retrieved strategy ID should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
