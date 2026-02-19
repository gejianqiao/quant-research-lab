"""
Logging utilities for the QuantEvolve framework.

This module provides comprehensive logging infrastructure including:
- Standard logging setup with file and console handlers
- Custom EvolutionLogger class for tracking evolutionary progress
- Structured logging for generations, strategies, metrics, and insights
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class GenerationLog:
    """Data class for logging generation-level information."""
    generation: int
    island_id: int
    parent_id: str
    parent_score: float
    cousin_ids: List[str]
    hypothesis_summary: str
    code_generated: bool
    backtest_success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    category_bits: str = ""
    feature_vector: List[Any] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class StrategyLog:
    """Data class for logging individual strategy information."""
    strategy_id: str
    generation: int
    island_id: int
    hypothesis: Dict[str, Any]
    code_hash: str
    metrics: Dict[str, float]
    combined_score: float
    feature_vector: List[Any]
    category_bits: str
    archived: bool
    replaced_strategy_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MigrationLog:
    """Data class for logging migration events."""
    generation: int
    source_island: int
    target_island: int
    migrated_strategy_ids: List[str]
    migrated_scores: List[float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class InsightLog:
    """Data class for logging extracted insights."""
    insight_id: str
    generation: int
    category: str
    content: str
    source_strategy_id: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EvolutionLogger:
    """
    Custom logger for tracking evolutionary algorithm progress.
    
    Provides structured logging for:
    - Generation-level events (parent selection, cousin sampling, hypothesis generation)
    - Strategy evaluation and archiving
    - Migration events between islands
    - Insight extraction and curation
    - Performance metrics tracking
    """
    
    def __init__(self, log_dir: str, log_level: int = logging.INFO, **kwargs):
        """
        Initialize the evolution logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (default: INFO)
            **kwargs: Extra compatibility arguments (e.g., save_interval)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = log_level
        
        # Main logger
        self.logger = logging.getLogger('QuantEvolve')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        general_log_file = self.log_dir / 'evolution.log'
        file_handler = logging.FileHandler(general_log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON log files for structured data
        self.generation_log_file = self.log_dir / 'generations.jsonl'
        self.strategy_log_file = self.log_dir / 'strategies.jsonl'
        self.migration_log_file = self.log_dir / 'migrations.jsonl'
        self.insight_log_file = self.log_dir / 'insights.jsonl'
        self.metrics_log_file = self.log_dir / 'metrics.jsonl'
        
        # Initialize counters
        self.total_generations = 0
        self.total_strategies = 0
        self.total_migrations = 0
        self.total_insights = 0
        
        # Performance tracking
        self.generation_metrics = []
        self.best_scores_per_generation = []
        
        self.logger.info(f"EvolutionLogger initialized at {self.log_dir}")
    
    def log_generation_start(self, generation: int, num_islands: int):
        """Log the start of a new generation."""
        self.logger.info(f"Starting Generation {generation} with {num_islands} islands")
        self.total_generations = generation
    
    def log_generation_event(self, generation_log: GenerationLog):
        """
        Log a generation-level event (parent/cousin selection, hypothesis, backtest).
        
        Args:
            generation_log: GenerationLog dataclass instance
        """
        # Log to file as JSON line
        with open(self.generation_log_file, 'a') as f:
            f.write(json.dumps(generation_log.to_dict()) + '\n')
        
        # Log summary to main logger
        status = "✓" if generation_log.backtest_success else "✗"
        self.logger.debug(
            f"Gen {generation_log.generation} Island {generation_log.island_id}: "
            f"Parent={generation_log.parent_id} (score={generation_log.parent_score:.3f}) "
            f"| Backtest {status} | SR={generation_log.metrics.get('sharpe_ratio', 0):.3f}"
        )
    
    def log_strategy_created(self, strategy_log: StrategyLog):
        """
        Log a newly created strategy.
        
        Args:
            strategy_log: StrategyLog dataclass instance
        """
        # Log to file
        with open(self.strategy_log_file, 'a') as f:
            f.write(json.dumps(strategy_log.to_dict()) + '\n')
        
        self.total_strategies += 1
        
        # Log summary
        archive_status = "ARCHIVED" if strategy_log.archived else "REJECTED"
        self.logger.info(
            f"Strategy {strategy_log.strategy_id} (Gen {strategy_log.generation}): "
            f"Score={strategy_log.combined_score:.3f} | "
            f"SR={strategy_log.metrics.get('sharpe_ratio', 0):.3f} | "
            f"MDD={strategy_log.metrics.get('max_drawdown', 0):.3f} | "
            f"{archive_status}"
        )
    
    def log_migration(self, migration_log: MigrationLog):
        """
        Log a migration event between islands.
        
        Args:
            migration_log: MigrationLog dataclass instance
        """
        # Log to file
        with open(self.migration_log_file, 'a') as f:
            f.write(json.dumps(migration_log.to_dict()) + '\n')
        
        self.total_migrations += 1
        
        # Log summary
        self.logger.info(
            f"Migration (Gen {migration_log.generation}): "
            f"Island {migration_log.source_island} → Island {migration_log.target_island} | "
            f"{len(migration_log.migrated_strategy_ids)} strategies migrated | "
            f"Best score={max(migration_log.migrated_scores):.3f}"
        )
    
    def log_insight(self, insight_log: InsightLog):
        """
        Log an extracted insight.
        
        Args:
            insight_log: InsightLog dataclass instance
        """
        # Log to file
        with open(self.insight_log_file, 'a') as f:
            f.write(json.dumps(insight_log.to_dict()) + '\n')
        
        self.total_insights += 1
        
        # Log summary
        self.logger.debug(
            f"Insight #{self.total_insights} (Gen {insight_log.generation}): "
            f"[{insight_log.category}] {insight_log.content[:100]}..."
        )
    
    def log_metrics(self, generation: int, metrics: Dict[str, Any]):
        """
        Log generation-level metrics summary.
        
        Args:
            generation: Generation number
            metrics: Dictionary of metrics (avg_score, best_score, diversity, etc.)
        """
        metrics_entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Log to file
        with open(self.metrics_log_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
        
        # Track for performance monitoring
        self.generation_metrics.append(metrics_entry)
        if 'best_score' in metrics:
            self.best_scores_per_generation.append(metrics['best_score'])
        
        # Log summary
        self.logger.info(
            f"Generation {generation} Summary: "
            f"Best Score={metrics.get('best_score', 0):.3f} | "
            f"Avg Score={metrics.get('average_score', 0):.3f} | "
            f"Diversity={metrics.get('diversity_score', 0):.3f} | "
            f"Occupancy={metrics.get('occupancy_rate', 0):.2%}"
        )
    
    def log_checkpoint(self, generation: int, checkpoint_path: str):
        """
        Log a checkpoint save event.
        
        Args:
            generation: Generation number
            checkpoint_path: Path to the checkpoint file
        """
        self.logger.info(f"Checkpoint saved at Generation {generation}: {checkpoint_path}")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error with context.
        
        Args:
            error: Exception instance
            context: Additional context about where the error occurred
        """
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str, context: str = ""):
        """
        Log a warning with context.
        
        Args:
            message: Warning message
            context: Additional context
        """
        self.logger.warning(f"Warning in {context}: {message}")
    
    def get_generation_summary(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve summary for a specific generation from logs.
        
        Args:
            generation: Generation number to retrieve
            
        Returns:
            Dictionary with generation summary or None if not found
        """
        if not self.generation_log_file.exists():
            return None
        
        gen_events = []
        with open(self.generation_log_file, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                if event.get('generation') == generation:
                    gen_events.append(event)
        
        if not gen_events:
            return None
        
        # Aggregate statistics
        successful_backtests = sum(1 for e in gen_events if e.get('backtest_success', False))
        avg_sharpe = sum(e.get('metrics', {}).get('sharpe_ratio', 0) for e in gen_events) / len(gen_events)
        
        return {
            'generation': generation,
            'total_events': len(gen_events),
            'successful_backtests': successful_backtests,
            'success_rate': successful_backtests / len(gen_events),
            'average_sharpe': avg_sharpe,
            'islands_active': len(set(e.get('island_id') for e in gen_events))
        }
    
    def get_best_strategies(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the top N strategies by combined score from logs.
        
        Args:
            top_n: Number of top strategies to retrieve
            
        Returns:
            List of strategy dictionaries sorted by score
        """
        if not self.strategy_log_file.exists():
            return []
        
        strategies = []
        with open(self.strategy_log_file, 'r') as f:
            for line in f:
                strategy = json.loads(line.strip())
                if strategy.get('archived', False):
                    strategies.append(strategy)
        
        # Sort by combined score descending
        strategies.sort(key=lambda s: s.get('combined_score', 0), reverse=True)
        
        return strategies[:top_n]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the entire evolution run.
        
        Returns:
            Dictionary with overall statistics
        """
        return {
            'total_generations': self.total_generations,
            'total_strategies': self.total_strategies,
            'total_migrations': self.total_migrations,
            'total_insights': self.total_insights,
            'best_score_overall': max(self.best_scores_per_generation) if self.best_scores_per_generation else 0,
            'final_best_score': self.best_scores_per_generation[-1] if self.best_scores_per_generation else 0,
            'score_improvement': (
                self.best_scores_per_generation[-1] - self.best_scores_per_generation[0]
                if len(self.best_scores_per_generation) > 1 else 0
            )
        }


def setup_logging(
    log_dir: str = "results/logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    level: Optional[Any] = None,
    directory: Optional[str] = None,
) -> logging.Logger:
    """
    Set up standard logging configuration for the QuantEvolve framework.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Compatibility aliases.
    if directory is not None:
        log_dir = directory
    if level is not None:
        log_level = level

    # Convert string/int level to int
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    if isinstance(log_level, int):
        log_level_int = log_level
    else:
        log_level_int = level_map.get(str(log_level).upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('QuantEvolve')
    logger.setLevel(log_level_int)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # General log file
        general_log_file = log_path / 'quant_evolve.log'
        file_handler = logging.FileHandler(general_log_file, mode='a')
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error log file (separate file for errors only)
        error_log_file = log_path / 'errors.log'
        error_handler = logging.FileHandler(error_log_file, mode='a')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized at level {log_level} to {log_dir}")
    
    return logger


def get_logger(name: str = 'QuantEvolve') -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Logger name (default: 'QuantEvolve')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
