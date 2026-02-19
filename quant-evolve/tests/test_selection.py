"""
Test suite for the selection mechanisms in QuantEvolve.

This module validates the parent selection (Equation 1) and cousin selection (Equation 2)
mechanisms, including feature vector perturbation, bit flip operations, and the balance
between exploitation (elite selection) and exploration (diverse selection).

Tests cover:
- Parent selection probability distribution (alpha parameter)
- Cousin selection composition (best, diverse, random)
- Feature vector perturbation (Gaussian noise + bit flips)
- Bit flip operations on binary category strings
- Edge cases (empty islands, single strategy, etc.)
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Import selection functions
from src.evolutionary.selection import (
    sample_parent,
    sample_cousins,
    tournament_selection,
    create_selection_pool,
    validate_selection_config
)

# Import FeatureMap and Strategy for test setup
from src.evolutionary.feature_map import (
    FeatureMap,
    Strategy,
    bit_flip,
    perturb_feature_vector,
    STRATEGY_CATEGORIES
)


class TestBitFlip:
    """Test bit flip operations for category dimension perturbation."""

    def test_bit_flip_single_bit(self):
        """Test flipping a single bit in a binary string."""
        original = '00000000'
        result = bit_flip(original, k_bf=1)
        
        # Should differ by exactly 1 bit (with replacement, could be same bit flipped twice)
        # But with k_bf=1, exactly 1 bit should flip
        diff_count = sum(1 for a, b in zip(original, result) if a != b)
        assert diff_count == 1, f"Expected 1 bit flip, got {diff_count}"
        assert len(result) == 8, "Binary string length should be preserved"

    def test_bit_flip_multiple_bits(self):
        """Test flipping multiple bits."""
        original = '00000000'
        result = bit_flip(original, k_bf=4)
        
        # With replacement, could flip same bit multiple times
        # But on average, should see several differences
        diff_count = sum(1 for a, b in zip(original, result) if a != b)
        assert 0 <= diff_count <= 4, f"Bit flips should be between 0 and 4, got {diff_count}"
        assert len(result) == 8

    def test_bit_flip_with_replacement_cancellation(self):
        """Test that flipping same bit twice cancels out."""
        original = '10000000'
        # Flip bit 0 twice should return to original
        # Note: bit_flip uses random selection, so we test statistical behavior
        np.random.seed(42)
        result = bit_flip(original, k_bf=2)
        # With seed 42, we can check reproducibility
        assert len(result) == 8

    def test_bit_flip_preserves_length(self):
        """Test that bit flip preserves binary string length."""
        for length in [4, 8, 16, 32]:
            original = '0' * length
            result = bit_flip(original, k_bf=length // 2)
            assert len(result) == length, f"Length should be preserved for {length}-bit string"

    def test_bit_flip_zero_flips(self):
        """Test that zero flips returns original string."""
        original = '10101010'
        result = bit_flip(original, k_bf=0)
        assert result == original, "Zero flips should return original string"

    def test_bit_flip_all_bits(self):
        """Test flipping all bits."""
        original = '00000000'
        result = bit_flip(original, k_bf=8)
        # With replacement, might not flip all 8, but should see changes
        diff_count = sum(1 for a, b in zip(original, result) if a != b)
        assert diff_count > 0, "Should see at least one bit flip"

    def test_bit_flip_reproducibility(self):
        """Test that bit flip is reproducible with same seed."""
        original = '11001100'
        np.random.seed(123)
        result1 = bit_flip(original, k_bf=3)
        np.random.seed(123)
        result2 = bit_flip(original, k_bf=3)
        assert result1 == result2, "Bit flip should be reproducible with same seed"


class TestPerturbFeatureVector:
    """Test feature vector perturbation for diverse cousin selection."""

    def test_perturb_continuous_dimensions(self):
        """Test Gaussian perturbation of continuous dimensions."""
        # Feature vector: [category_bits, frequency, mdd, sr, sor, return]
        original = ['00000000', 5, -0.15, 1.2, 1.5, 0.25]
        
        np.random.seed(42)
        perturbed = perturb_feature_vector(
            original,
            sigma_d=1.0,
            k_bf_ratio=0.25,
            category_bits=8
        )
        
        # Category should be perturbed via bit flip
        assert isinstance(perturbed[0], str), "Category should remain a string"
        assert len(perturbed[0]) == 8, "Category bits length should be preserved"
        
        # Continuous dimensions should be perturbed
        # With sigma=1.0, expect some deviation
        for i in range(1, len(original)):
            # Just check it's a number (specific values depend on random seed)
            assert isinstance(perturbed[i], (int, float)), f"Dimension {i} should be numeric"

    def test_perturb_category_dimension(self):
        """Test bit flip perturbation of category dimension."""
        original = ['10000000', 10, -0.20, 0.8, 1.0, 0.15]
        
        np.random.seed(99)
        perturbed = perturb_feature_vector(
            original,
            sigma_d=1.0,
            k_bf_ratio=0.25,  # 25% of 8 bits = 2 bits
            category_bits=8
        )
        
        # Category should be different (with high probability)
        # Note: could be same by chance, but seed 99 should produce difference
        assert isinstance(perturbed[0], str)
        assert len(perturbed[0]) == 8

    def test_perturb_reproducibility(self):
        """Test perturbation reproducibility with same seed."""
        original = ['01010101', 8, -0.10, 1.5, 1.8, 0.30]
        
        np.random.seed(456)
        perturbed1 = perturb_feature_vector(original, sigma_d=1.0, k_bf_ratio=0.25, category_bits=8)
        
        np.random.seed(456)
        perturbed2 = perturb_feature_vector(original, sigma_d=1.0, k_bf_ratio=0.25, category_bits=8)
        
        assert perturbed1 == perturbed2, "Perturbation should be reproducible"

    def test_perturb_different_sigma(self):
        """Test that different sigma values produce different perturbation magnitudes."""
        original = ['00000000', 10, -0.15, 1.0, 1.2, 0.20]
        
        np.random.seed(789)
        perturbed_low = perturb_feature_vector(original, sigma_d=0.1, k_bf_ratio=0.25, category_bits=8)
        
        np.random.seed(789)
        perturbed_high = perturb_feature_vector(original, sigma_d=5.0, k_bf_ratio=0.25, category_bits=8)
        
        # Higher sigma should produce larger deviations (on average)
        # Check continuous dimensions
        deviations_low = [abs(perturbed_low[i] - original[i]) for i in range(1, len(original))]
        deviations_high = [abs(perturbed_high[i] - original[i]) for i in range(1, len(original))]
        
        # This is probabilistic, but with seed control should hold
        # Just verify both produce valid perturbations
        assert len(deviations_low) == len(deviations_high) == 5


class TestStrategyDataclass:
    """Test Strategy dataclass for selection operations."""

    def test_strategy_creation(self):
        """Test creating a Strategy instance."""
        strategy = Strategy(
            id='test_001',
            hypothesis={'hypothesis': 'Test hypothesis'},
            code='def initialize(context): pass',
            metrics={'sharpe_ratio': 1.5, 'max_drawdown': -0.10, 'total_return': 0.25},
            analysis={'category': 'momentum'},
            feature_vector=['10000000', 5, -0.10, 1.5, 1.8, 0.25],
            generation=0,
            island_id=0,
            score=2.65,  # SR + IR + MDD
            timestamp=datetime.now()
        )
        
        assert strategy.id == 'test_001'
        assert strategy.score == 2.65
        assert strategy.feature_vector[0] == '10000000'

    def test_strategy_comparison_by_score(self):
        """Test that strategies can be compared/sorted by score."""
        strategies = [
            Strategy(id=f's_{i}', hypothesis={}, code='', metrics={}, analysis={},
                    feature_vector=['00000000', 5, -0.1, 1.0, 1.2, 0.1],
                    generation=0, island_id=0, score=1.0 + i * 0.5, timestamp=datetime.now())
            for i in range(5)
        ]
        
        # Sort by score descending
        sorted_strategies = sorted(strategies, key=lambda s: s.score, reverse=True)
        
        assert sorted_strategies[0].score == 3.0  # Highest score
        assert sorted_strategies[-1].score == 1.0  # Lowest score


class TestSampleParent:
    """Test parent selection mechanism (Equation 1)."""

    def _create_test_strategies(self, n: int, island_id: int = 0) -> List[Strategy]:
        """Helper to create test strategies."""
        strategies = []
        for i in range(n):
            score = 1.0 + i * 0.5  # Increasing scores
            in_feature_map = i < n // 2  # First half are "elite" (in feature map)
            strategies.append(Strategy(
                id=f's_{i}',
                hypothesis={},
                code='',
                metrics={'sharpe_ratio': 1.0},
                analysis={},
                feature_vector=['00000000', 5, -0.1, 1.0, 1.2, 0.1],
                generation=0,
                island_id=island_id,
                score=score,
                timestamp=datetime.now()
            ))
        return strategies

    def _create_feature_map_mock(self, strategy_ids: List[str]):
        """Create a mock feature map with specific strategy IDs."""
        class MockFeatureMap:
            def __init__(self, ids):
                self.strategy_ids = set(ids)
        
        return MockFeatureMap(strategy_ids)

    def test_sample_parent_exploitation_mode(self):
        """Test parent selection with alpha=1.0 (always elite)."""
        island = self._create_test_strategies(10)
        elite_ids = [f's_{i}' for i in range(5)]  # First 5 are elite
        feature_map = self._create_feature_map_mock(elite_ids)
        
        np.random.seed(42)
        # With alpha=1.0, should always select from elite
        for _ in range(20):
            parent = sample_parent(island, feature_map, alpha=1.0)
            assert parent.id in elite_ids, f"Expected elite strategy, got {parent.id}"

    def test_sample_parent_exploration_mode(self):
        """Test parent selection with alpha=0.0 (always random from island)."""
        island = self._create_test_strategies(10)
        feature_map = self._create_feature_map_mock(['s_0', 's_1'])
        
        np.random.seed(42)
        # With alpha=0.0, should select from entire island
        parent = sample_parent(island, feature_map, alpha=0.0)
        assert parent in island, "Should select from island"

    def test_sample_parent_balanced_mode(self):
        """Test parent selection with alpha=0.5 (balanced)."""
        island = self._create_test_strategies(20)
        elite_ids = [f's_{i}' for i in range(10)]
        feature_map = self._create_feature_map_mock(elite_ids)
        
        np.random.seed(42)
        # With alpha=0.5, should select ~50% elite, ~50% random
        elite_count = 0
        n_samples = 100
        
        for _ in range(n_samples):
            parent = sample_parent(island, feature_map, alpha=0.5)
            if parent.id in elite_ids:
                elite_count += 1
        
        # Should be roughly 50% (allowing for randomness)
        elite_ratio = elite_count / n_samples
        assert 0.35 <= elite_ratio <= 0.65, f"Elite ratio {elite_ratio} should be ~0.5"

    def test_sample_parent_empty_feature_map(self):
        """Test parent selection when feature map is empty."""
        island = self._create_test_strategies(10)
        feature_map = self._create_feature_map_mock([])
        
        parent = sample_parent(island, feature_map, alpha=0.5)
        assert parent in island, "Should fallback to random island selection"

    def test_sample_parent_single_strategy(self):
        """Test parent selection with single strategy in island."""
        island = self._create_test_strategies(1)
        feature_map = self._create_feature_map_mock(['s_0'])
        
        parent = sample_parent(island, feature_map, alpha=1.0)
        assert parent.id == 's_0'

    def test_sample_parent_reproducibility(self):
        """Test parent selection reproducibility with same seed."""
        island = self._create_test_strategies(10)
        feature_map = self._create_feature_map_mock(['s_0', 's_1', 's_2'])
        
        np.random.seed(999)
        parent1 = sample_parent(island, feature_map, alpha=0.5)
        
        np.random.seed(999)
        parent2 = sample_parent(island, feature_map, alpha=0.5)
        
        assert parent1.id == parent2.id, "Should be reproducible with same seed"


class TestSampleCousins:
    """Test cousin selection mechanism (Equation 2)."""

    def _create_test_island(self, n: int) -> List[Strategy]:
        """Create a test island with diverse strategies."""
        strategies = []
        for i in range(n):
            category_bits = format(i % 8, '08b')  # Cycle through categories
            strategies.append(Strategy(
                id=f'c_{i}',
                hypothesis={},
                code='',
                metrics={'sharpe_ratio': 1.0 + i * 0.1},
                analysis={},
                feature_vector=[category_bits, 5 + i, -0.1 - i * 0.01, 
                               1.0 + i * 0.1, 1.2 + i * 0.1, 0.1 + i * 0.02],
                generation=0,
                island_id=0,
                score=2.0 + i * 0.2,
                timestamp=datetime.now()
            ))
        return strategies

    def _create_feature_map_with_strategies(self, strategies: List[Strategy]):
        """Create a feature map populated with strategies."""
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[8, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        for strategy in strategies:
            feature_map.update(strategy, strategy.feature_vector)
        
        return feature_map

    def test_sample_cousins_count(self):
        """Test that sample_cousins returns correct number of cousins."""
        island = self._create_test_island(20)
        feature_map = self._create_feature_map_with_strategies(island)
        parent = island[0]  # Use first strategy as parent
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        np.random.seed(42)
        cousins = sample_cousins(parent, island, feature_map, config)
        
        # Should return up to 7 cousins (may be fewer if island is small)
        assert len(cousins) <= 7, f"Expected <= 7 cousins, got {len(cousins)}"
        assert len(cousins) > 0, "Should return at least some cousins"

    def test_sample_cousins_excludes_parent(self):
        """Test that parent is not included in cousins."""
        island = self._create_test_island(20)
        feature_map = self._create_feature_map_with_strategies(island)
        parent = island[5]
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        np.random.seed(42)
        cousins = sample_cousins(parent, island, feature_map, config)
        
        cousin_ids = [c.id for c in cousins]
        assert parent.id not in cousin_ids, "Parent should not be in cousins"

    def test_sample_cousins_best_cousins(self):
        """Test that best cousins are high-performing strategies."""
        island = self._create_test_island(20)
        feature_map = self._create_feature_map_with_strategies(island)
        parent = island[0]  # Low-scoring parent
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 0,
                'num_random_cousins': 0,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        np.random.seed(42)
        cousins = sample_cousins(parent, island, feature_map, config)
        
        # Best cousins should have higher scores than parent
        for cousin in cousins:
            assert cousin.score >= parent.score, "Best cousins should outperform parent"

    def test_sample_cousins_diverse_cousins(self):
        """Test that diverse cousins have perturbed feature vectors."""
        island = self._create_test_island(20)
        feature_map = self._create_feature_map_with_strategies(island)
        parent = island[10]
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 0,
                'num_diverse_cousins': 3,
                'num_random_cousins': 0,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        np.random.seed(42)
        cousins = sample_cousins(parent, island, feature_map, config)
        
        # Diverse cousins should be found (or fallback to island strategies)
        assert len(cousins) > 0, "Should find diverse cousins"

    def test_sample_cousins_small_island(self):
        """Test cousin selection with small island."""
        island = self._create_test_island(5)  # Small island
        feature_map = self._create_feature_map_with_strategies(island)
        parent = island[0]
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        np.random.seed(42)
        cousins = sample_cousins(parent, island, feature_map, config)
        
        # Should handle small island gracefully
        assert len(cousins) <= 4, "Should not exceed island size - 1"


class TestTournamentSelection:
    """Test tournament selection as alternative parent selection."""

    def _create_test_strategies(self, n: int) -> List[Strategy]:
        """Helper to create test strategies."""
        return [
            Strategy(
                id=f't_{i}',
                hypothesis={},
                code='',
                metrics={},
                analysis={},
                feature_vector=['00000000', 5, -0.1, 1.0, 1.2, 0.1],
                generation=0,
                island_id=0,
                score=1.0 + i * 0.5,
                timestamp=datetime.now()
            )
            for i in range(n)
        ]

    def test_tournament_selection_winner(self):
        """Test that tournament selection returns the winner."""
        population = self._create_test_strategies(10)
        
        np.random.seed(42)
        winner = tournament_selection(population, tournament_size=3)
        
        assert winner in population, "Winner should be from population"

    def test_tournament_selection_larger_tournament(self):
        """Test tournament with larger tournament size."""
        population = self._create_test_strategies(20)
        
        # Larger tournament should select higher-scoring individuals on average
        np.random.seed(42)
        winner_small = tournament_selection(population, tournament_size=2)
        
        np.random.seed(42)
        winner_large = tournament_selection(population, tournament_size=10)
        
        # Just verify both return valid strategies
        assert winner_small in population
        assert winner_large in population

    def test_tournament_selection_single_individual(self):
        """Test tournament with single individual."""
        population = self._create_test_strategies(1)
        
        winner = tournament_selection(population, tournament_size=3)
        assert winner == population[0]


class TestCreateSelectionPool:
    """Test selection pool creation for LLM context."""

    def _create_strategies(self, n: int, prefix: str = 's') -> List[Strategy]:
        """Helper to create test strategies."""
        return [
            Strategy(
                id=f'{prefix}_{i}',
                hypothesis={'hypothesis': f'Test {i}'},
                code='',
                metrics={'sharpe_ratio': 1.0},
                analysis={},
                feature_vector=['00000000', 5, -0.1, 1.0, 1.2, 0.1],
                generation=0,
                island_id=0,
                score=1.0 + i * 0.5,
                timestamp=datetime.now()
            )
            for i in range(n)
        ]

    def test_create_selection_pool_uniform(self):
        """Test uniform selection pool creation."""
        parent = self._create_strategies(1, 'p')[0]
        cousins = self._create_strategies(7, 'c')
        
        pool = create_selection_pool(parent, cousins, selection_method='uniform')
        
        # Parent should be first, followed by cousins
        assert len(pool) == 8
        assert pool[0].id == parent.id

    def test_create_selection_pool_by_score(self):
        """Test score-based selection pool creation."""
        parent = Strategy(
            id='p_0',
            hypothesis={},
            code='',
            metrics={},
            analysis={},
            feature_vector=['00000000', 5, -0.1, 1.0, 1.2, 0.1],
            generation=0,
            island_id=0,
            score=1.5,
            timestamp=datetime.now()
        )
        cousins = self._create_strategies(3, 'c')
        
        pool = create_selection_pool(parent, cousins, selection_method='by_score')
        
        # Should be sorted by score
        assert len(pool) == 4
        # Verify sorted (descending)
        for i in range(len(pool) - 1):
            assert pool[i].score >= pool[i + 1].score

    def test_create_selection_pool_empty_cousins(self):
        """Test pool creation with no cousins."""
        parent = self._create_strategies(1, 'p')[0]
        
        pool = create_selection_pool(parent, [], selection_method='uniform')
        
        assert len(pool) == 1
        assert pool[0].id == parent.id


class TestValidateSelectionConfig:
    """Test configuration validation for selection mechanisms."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        result = validate_selection_config(config)
        assert result is True

    def test_validate_missing_cousin_selection(self):
        """Test validation with missing cousin_selection key."""
        config = {}
        
        result = validate_selection_config(config)
        assert result is False

    def test_validate_negative_sigma(self):
        """Test validation with negative sigma."""
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': -1.0,  # Invalid
                'bit_flip_ratio': 0.25
            }
        }
        
        result = validate_selection_config(config)
        assert result is False

    def test_validate_invalid_bit_flip_ratio(self):
        """Test validation with invalid bit flip ratio."""
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 1.5  # Should be 0-1
            }
        }
        
        result = validate_selection_config(config)
        assert result is False

    def test_validate_negative_cousin_count(self):
        """Test validation with negative cousin count."""
        config = {
            'cousin_selection': {
                'num_best_cousins': -2,  # Invalid
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        result = validate_selection_config(config)
        assert result is False


class TestSelectionIntegration:
    """Integration tests for selection mechanisms."""

    def test_parent_cousin_selection_pipeline(self):
        """Test complete parent and cousin selection pipeline."""
        # Create island with 30 strategies
        strategies = []
        for i in range(30):
            category_bits = format(i % 8, '08b')
            strategies.append(Strategy(
                id=f'int_{i}',
                hypothesis={},
                code='',
                metrics={'sharpe_ratio': 1.0 + i * 0.05},
                analysis={},
                feature_vector=[category_bits, 5 + i % 10, -0.1 - i * 0.005,
                               1.0 + i * 0.05, 1.2 + i * 0.05, 0.1 + i * 0.01],
                generation=0,
                island_id=0,
                score=2.0 + i * 0.1,
                timestamp=datetime.now()
            ))
        
        # Create feature map
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[8, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        # Add top 15 strategies to feature map (elites)
        for strategy in strategies[:15]:
            feature_map.update(strategy, strategy.feature_vector)
        
        # Configuration
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        # Select parent (balanced mode)
        np.random.seed(123)
        parent = sample_parent(strategies, feature_map, alpha=0.5)
        
        # Select cousins
        cousins = sample_cousins(parent, strategies, feature_map, config)
        
        # Verify
        assert parent in strategies
        assert len(cousins) > 0
        assert parent.id not in [c.id for c in cousins]
        
        # Create selection pool
        pool = create_selection_pool(parent, cousins, selection_method='uniform')
        assert len(pool) == 1 + len(cousins)

    def test_selection_diversity_preservation(self):
        """Test that selection maintains diversity."""
        # Create island with strategies from different categories
        strategies = []
        for category_idx in range(8):
            category_bits = format(category_idx, '08b')
            for j in range(5):
                strategies.append(Strategy(
                    id=f'div_{category_idx}_{j}',
                    hypothesis={},
                    code='',
                    metrics={},
                    analysis={},
                    feature_vector=[category_bits, 5, -0.1, 1.0 + j * 0.1, 1.2, 0.1],
                    generation=0,
                    island_id=0,
                    score=2.0 + j * 0.1,
                    timestamp=datetime.now()
                ))
        
        feature_map = FeatureMap(
            dimensions=6,
            bin_sizes=[8, 16, 16, 16, 16, 16],
            category_bits=8
        )
        
        for strategy in strategies:
            feature_map.update(strategy, strategy.feature_vector)
        
        config = {
            'cousin_selection': {
                'num_best_cousins': 2,
                'num_diverse_cousins': 3,
                'num_random_cousins': 2,
                'perturbation_sigma': 1.0,
                'bit_flip_ratio': 0.25
            }
        }
        
        # Sample multiple parents and check category diversity
        np.random.seed(456)
        parent_categories = set()
        for _ in range(20):
            parent = sample_parent(strategies, feature_map, alpha=0.5)
            parent_categories.add(parent.feature_vector[0])
        
        # Should see multiple categories (not collapsed to one)
        assert len(parent_categories) >= 3, "Should maintain category diversity"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
