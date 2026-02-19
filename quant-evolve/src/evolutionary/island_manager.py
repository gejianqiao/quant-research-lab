"""
Island Manager for QuantEvolve.

Implements the island model for parallel evolution with migration between islands.
Each island maintains a subpopulation of strategies, with periodic migration
of top performers to neighboring islands (ring topology).

References:
    - Paper Section 4.3: Island Model and Migration
    - Plan Component 5: Island Manager
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from .feature_map import Strategy, FeatureMap

logger = logging.getLogger(__name__)


@dataclass
class Island:
    """Represents a single island in the island model.
    
    Attributes:
        island_id: Unique identifier for this island
        strategies: List of strategies in this island's population
        category_focus: Strategy category this island specializes in (optional)
        migration_history: List of strategies that migrated to/from this island
    """
    island_id: int
    strategies: List[Strategy] = field(default_factory=list)
    category_focus: Optional[str] = None
    migration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to this island's population."""
        self.strategies.append(strategy)
        logger.debug(f"Added strategy {strategy.id} to island {self.island_id}")
    
    def remove_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Remove a strategy from this island by ID."""
        for i, strategy in enumerate(self.strategies):
            if strategy.id == strategy_id:
                removed = self.strategies.pop(i)
                logger.debug(f"Removed strategy {strategy_id} from island {self.island_id}")
                return removed
        return None
    
    def get_top_strategies(self, n: int) -> List[Strategy]:
        """Get the top N strategies by score from this island."""
        if not self.strategies:
            return []
        sorted_strategies = sorted(self.strategies, key=lambda s: s.score, reverse=True)
        return sorted_strategies[:n]
    
    def get_best_strategy(self) -> Optional[Strategy]:
        """Get the single best strategy from this island."""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.score)
    
    def get_average_score(self) -> float:
        """Get the average score of strategies in this island."""
        if not self.strategies:
            return 0.0
        return sum(s.score for s in self.strategies) / len(self.strategies)
    
    def size(self) -> int:
        """Get the number of strategies in this island."""
        return len(self.strategies)
    
    def clear(self) -> None:
        """Clear all strategies from this island."""
        self.strategies.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert island to dictionary representation."""
        return {
            'island_id': self.island_id,
            'category_focus': self.category_focus,
            'size': self.size(),
            'average_score': self.get_average_score(),
            'best_score': self.get_best_strategy().score if self.get_best_strategy() else None,
            'strategy_ids': [s.id for s in self.strategies]
        }


class IslandManager:
    """Manages multiple islands and migration between them.
    
    Implements a ring topology where each island can migrate strategies
    to its left and right neighbors. Migration occurs every M generations,
    exchanging the top 10% of strategies from each island.
    
    Attributes:
        islands: List of Island objects
        num_islands: Total number of islands
        migration_rate: Fraction of population to migrate (default 0.1)
        migration_interval: Number of generations between migrations
        topology: Migration topology ('ring', 'complete', etc.)
    """
    
    def __init__(
        self,
        num_islands: int = 9,
        migration_rate: float = 0.1,
        migration_interval: int = 10,
        topology: str = 'ring',
        category_list: Optional[List[str]] = None
    ):
        """Initialize the Island Manager.
        
        Args:
            num_islands: Number of islands to create (default 9: 8 categories + 1 B&H)
            migration_rate: Fraction of top strategies to migrate (default 0.1 = 10%)
            migration_interval: Generations between migration events (default 10)
            topology: Migration topology ('ring' or 'complete')
            category_list: List of strategy categories for island specialization
        """
        self.num_islands = num_islands
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.topology = topology
        self.islands: List[Island] = []
        self.current_generation = 0
        self.total_migrations = 0
        self.migration_log: List[Dict[str, Any]] = []
        
        # Initialize islands
        self._initialize_islands(category_list)
        
        logger.info(f"Initialized IslandManager with {num_islands} islands, "
                   f"migration_rate={migration_rate}, interval={migration_interval}")
    
    def _initialize_islands(self, category_list: Optional[List[str]] = None) -> None:
        """Create and initialize all islands.
        
        Args:
            category_list: Optional list of category names to assign to islands
        """
        self.islands = []
        
        for i in range(self.num_islands):
            category_focus = None
            if category_list and i < len(category_list):
                category_focus = category_list[i]
            
            island = Island(
                island_id=i,
                strategies=[],
                category_focus=category_focus
            )
            self.islands.append(island)
        
        logger.debug(f"Created {len(self.islands)} islands")
    
    def seed_island(self, island_id: int, strategy: Strategy) -> None:
        """Add a seed strategy to a specific island.
        
        Args:
            island_id: ID of the island to seed
            strategy: Strategy to add as seed
        """
        if 0 <= island_id < len(self.islands):
            self.islands[island_id].add_strategy(strategy)
            logger.debug(f"Seeded island {island_id} with strategy {strategy.id}")
        else:
            logger.error(f"Invalid island_id {island_id} for seeding")
    
    def seed_all_islands(self, seed_strategies: List[Strategy]) -> None:
        """Seed all islands with initial strategies.
        
        Args:
            seed_strategies: List of strategies, one per island (or more for distribution)
        """
        if len(seed_strategies) < self.num_islands:
            logger.warning(f"Only {len(seed_strategies)} seed strategies for {self.num_islands} islands")
        
        # Distribute strategies across islands
        for i, strategy in enumerate(seed_strategies):
            island_id = i % self.num_islands
            self.seed_island(island_id, strategy)
        
        logger.info(f"Seeded all islands with {len(seed_strategies)} strategies")
    
    def add_strategy_to_island(self, island_id: int, strategy: Strategy) -> None:
        """Add a newly generated strategy to its parent island.
        
        Args:
            island_id: ID of the island where strategy was generated
            strategy: New strategy to add
        """
        if 0 <= island_id < len(self.islands):
            self.islands[island_id].add_strategy(strategy)
        else:
            logger.error(f"Cannot add strategy to invalid island {island_id}")
    
    def get_island(self, island_id: int) -> Optional[Island]:
        """Get an island by its ID.
        
        Args:
            island_id: ID of the island to retrieve
            
        Returns:
            Island object or None if not found
        """
        if 0 <= island_id < len(self.islands):
            return self.islands[island_id]
        return None
    
    def get_all_strategies(self) -> List[Strategy]:
        """Get all strategies from all islands.
        
        Returns:
            Flat list of all strategies across all islands
        """
        all_strategies = []
        for island in self.islands:
            all_strategies.extend(island.strategies)
        return all_strategies
    
    def get_neighbor_indices(self, island_id: int) -> List[int]:
        """Get the indices of neighboring islands for migration.
        
        Args:
            island_id: ID of the island to find neighbors for
            
        Returns:
            List of neighbor island indices
        """
        if self.topology == 'ring':
            # Ring topology: left and right neighbors
            left_neighbor = (island_id - 1) % self.num_islands
            right_neighbor = (island_id + 1) % self.num_islands
            return [left_neighbor, right_neighbor]
        elif self.topology == 'complete':
            # Complete topology: all other islands are neighbors
            return [i for i in range(self.num_islands) if i != island_id]
        else:
            logger.warning(f"Unknown topology '{self.topology}', defaulting to ring")
            return [(island_id - 1) % self.num_islands, (island_id + 1) % self.num_islands]
    
    def migrate_strategies(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Perform migration of top strategies between neighboring islands.
        
        Implements the migration process where top 10% of strategies from each
        island are copied to neighboring islands (ring topology by default).
        
        Args:
            generation: Current generation number (optional, for logging)
            
        Returns:
            Dictionary with migration statistics
        """
        if generation is not None:
            self.current_generation = generation
        
        migration_stats = {
            'generation': generation,
            'migrations_per_island': [],
            'total_migrated': 0
        }
        
        # Create a copy of strategies to migrate (to avoid modifying during iteration)
        migration_plan: List[Dict[str, Any]] = []
        
        for island in self.islands:
            if island.size() == 0:
                continue
            
            # Calculate number of strategies to migrate (top 10%)
            num_to_migrate = max(1, int(island.size() * self.migration_rate))
            top_strategies = island.get_top_strategies(num_to_migrate)
            
            # Get neighbors based on topology
            neighbor_indices = self.get_neighbor_indices(island.island_id)
            
            # Plan migration to neighbors
            for neighbor_idx in neighbor_indices:
                for strategy in top_strategies:
                    migration_plan.append({
                        'from_island': island.island_id,
                        'to_island': neighbor_idx,
                        'strategy_id': strategy.id,
                        'strategy_score': strategy.score
                    })
        
        # Execute migration plan (copy strategies, don't remove from source)
        for migration in migration_plan:
            from_island = self.get_island(migration['from_island'])
            to_island = self.get_island(migration['to_island'])
            
            if from_island and to_island:
                # Find the strategy in source island
                strategy = None
                for s in from_island.strategies:
                    if s.id == migration['strategy_id']:
                        strategy = s
                        break
                
                if strategy:
                    # Create a copy of the strategy for the destination island
                    strategy_copy = deepcopy(strategy)
                    strategy_copy.island_id = to_island.island_id
                    to_island.add_strategy(strategy_copy)
                    
                    # Record migration in history
                    to_island.migration_history.append({
                        'generation': generation,
                        'strategy_id': strategy.id,
                        'from_island': migration['from_island'],
                        'score': strategy.score
                    })
        
        # Update statistics
        self.total_migrations += len(migration_plan)
        migration_stats['total_migrated'] = len(migration_plan)
        migration_stats['migrations_per_island'] = migration_plan
        
        # Log migration
        self.migration_log.append({
            'generation': generation,
            'total_migrated': len(migration_plan),
            'details': migration_plan
        })
        
        logger.info(f"Migration at generation {generation}: {len(migration_plan)} strategies migrated")
        
        return migration_stats
    
    def should_migrate(self, generation: int) -> bool:
        """Check if migration should occur at this generation.
        
        Args:
            generation: Current generation number
            
        Returns:
            True if migration should occur, False otherwise
        """
        return (generation > 0 and generation % self.migration_interval == 0)
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population across all islands.
        
        Returns:
            Dictionary with population statistics
        """
        stats = {
            'total_strategies': 0,
            'islands': [],
            'average_score_all': 0.0,
            'best_score_all': float('-inf'),
            'best_strategy_id': None
        }
        
        all_scores = []
        
        for island in self.islands:
            island_stats = island.to_dict()
            stats['islands'].append(island_stats)
            stats['total_strategies'] += island.size()
            
            for strategy in island.strategies:
                all_scores.append(strategy.score)
                if strategy.score > stats['best_score_all']:
                    stats['best_score_all'] = strategy.score
                    stats['best_strategy_id'] = strategy.id
        
        if all_scores:
            stats['average_score_all'] = sum(all_scores) / len(all_scores)
        
        return stats
    
    def get_diversity_metrics(self) -> Dict[str, Any]:
        """Calculate diversity metrics across islands.
        
        Returns:
            Dictionary with diversity metrics
        """
        metrics = {
            'num_islands': self.num_islands,
            'island_sizes': [island.size() for island in self.islands],
            'size_variance': 0.0,
            'category_distribution': {},
            'migration_count': self.total_migrations
        }
        
        # Calculate size variance
        sizes = metrics['island_sizes']
        if sizes:
            mean_size = sum(sizes) / len(sizes)
            variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            metrics['size_variance'] = variance
        
        # Count category distribution
        category_counts: Dict[str, int] = {}
        for island in self.islands:
            for strategy in island.strategies:
                if strategy.analysis and 'category_bits' in strategy.analysis:
                    cat_bits = strategy.analysis['category_bits']
                    category_counts[cat_bits] = category_counts.get(cat_bits, 0) + 1
        
        metrics['category_distribution'] = category_counts
        
        return metrics
    
    def prune_island(self, island_id: int, max_size: int) -> int:
        """Remove lowest-performing strategies from an island if it exceeds max size.
        
        Args:
            island_id: ID of the island to prune
            max_size: Maximum number of strategies to keep
            
        Returns:
            Number of strategies removed
        """
        island = self.get_island(island_id)
        if not island:
            return 0
        
        if island.size() <= max_size:
            return 0
        
        # Sort by score and keep top performers
        num_to_remove = island.size() - max_size
        sorted_strategies = sorted(island.strategies, key=lambda s: s.score)
        
        # Remove lowest performers
        for i in range(num_to_remove):
            island.strategies.pop(0)
        
        logger.debug(f"Pruned {num_to_remove} strategies from island {island_id}")
        return num_to_remove
    
    def prune_all_islands(self, max_size_per_island: int) -> int:
        """Prune all islands to maintain maximum population size.
        
        Args:
            max_size_per_island: Maximum strategies per island
            
        Returns:
            Total number of strategies removed
        """
        total_removed = 0
        for island in self.islands:
            removed = self.prune_island(island.island_id, max_size_per_island)
            total_removed += removed
        return total_removed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert island manager state to dictionary.
        
        Returns:
            Dictionary representation of island manager state
        """
        return {
            'num_islands': self.num_islands,
            'migration_rate': self.migration_rate,
            'migration_interval': self.migration_interval,
            'topology': self.topology,
            'current_generation': self.current_generation,
            'total_migrations': self.total_migrations,
            'islands': [island.to_dict() for island in self.islands],
            'population_stats': self.get_population_statistics()
        }


def create_island_manager_from_config(config: Dict[str, Any]) -> IslandManager:
    """Create an IslandManager instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'islands' section
        
    Returns:
        Configured IslandManager instance
    """
    islands_config = config.get('islands', {})
    
    num_islands = islands_config.get('num_islands', 9)
    migration_rate = islands_config.get(
        'migration_rate',
        islands_config.get('migration_percentage', 0.1)
    )
    migration_interval = islands_config.get('migration_interval', 10)
    topology = islands_config.get('topology', 'ring')
    
    # Get category list from feature map config if available
    category_list = None
    feature_map_config = config.get('feature_map', {})
    if 'category_bits' in feature_map_config:
        from .feature_map import STRATEGY_CATEGORIES
        category_list = STRATEGY_CATEGORIES[:num_islands-1] + ['buy_and_hold']
    
    return IslandManager(
        num_islands=num_islands,
        migration_rate=migration_rate,
        migration_interval=migration_interval,
        topology=topology,
        category_list=category_list
    )
