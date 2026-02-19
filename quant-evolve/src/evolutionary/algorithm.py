"""
QuantEvolve: Main Evolutionary Algorithm Implementation

This module implements Algorithm 1 from the paper: the main evolutionary loop
that orchestrates the multi-agent system, feature map updates, island migration,
and insight curation.

Author: QuantEvolve Team
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .feature_map import FeatureMap, Strategy, create_feature_map_from_config
from .island_manager import IslandManager, create_island_manager_from_config
from .selection import sample_parent, sample_cousins, create_selection_pool
from ..agents.data_agent import DataAgent
from ..agents.research_agent import ResearchAgent
from ..agents.coding_team import CodingTeam
from ..agents.evaluation_team import EvaluationTeam
from ..backtesting.engine import BacktestingEngine
from ..backtesting.metrics import compute_all_metrics, compute_combined_score
from ..utils.data_loader import load_ohlcv_data
from ..utils.logging import setup_logging, EvolutionLogger

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryDatabase:
    """
    Central repository containing feature map + archive of all strategies.
    
    Attributes:
        feature_map: Multi-dimensional archive for quality-diversity optimization
        archive: List of all strategies ever generated (including rejected ones)
        generation_counter: Current generation number
        total_strategies_evaluated: Count of all backtests run
    """
    feature_map: FeatureMap
    archive: List[Strategy] = field(default_factory=list)
    generation_counter: int = 0
    total_strategies_evaluated: int = 0
    insights_repository: List[Dict[str, Any]] = field(default_factory=list)
    feature_map_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    generation_metrics: List[Dict[str, Any]] = field(default_factory=list)
    insights_timeline: List[Dict[str, Any]] = field(default_factory=list)
    strategy_returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    
    def update(self, strategy: Strategy, feature_vector: List[Any], island_id: int) -> bool:
        """
        Update database with new strategy.
        
        Args:
            strategy: Strategy object with metrics and code
            feature_vector: Computed feature vector for binning
            island_id: ID of the island this strategy came from
            
        Returns:
            bool: True if strategy was added to feature map (new cell or replacement)
        """
        strategy.feature_vector = feature_vector
        strategy.island_id = island_id
        strategy.generation = self.generation_counter
        
        # Update feature map (handles replacement logic)
        added_to_map = self.feature_map.update(strategy, feature_vector)
        
        # Always add to archive
        self.archive.append(strategy)
        self.total_strategies_evaluated += 1
        
        return added_to_map
    
    def get_all_strategies(self) -> List[Strategy]:
        """Return all strategies in archive."""
        return self.archive
    
    def get_feature_map_snapshot(self) -> Dict[str, Any]:
        """Get current state of feature map for visualization."""
        return self.feature_map.get_snapshot()
    
    def add_insight(self, insight: Dict[str, Any]):
        """Add insight to repository."""
        self.insights_repository.append(insight)
        self.insights_timeline.append({
            'generation': insight.get('generation', self.generation_counter),
            'strategy_id': insight.get('strategy_id', ''),
            'island_id': insight.get('island_id', -1),
            'timestamp': datetime.now().isoformat(),
        })
    
    def curate_insights(self, max_insights: int = 500):
        """Curate insights repository to prevent memory bloat."""
        if len(self.insights_repository) > max_insights:
            # Keep most recent unique insights
            seen = set()
            curated = []
            for insight in reversed(self.insights_repository):
                key = json.dumps(insight, sort_keys=True)
                if key not in seen:
                    curated.insert(0, insight)
                    seen.add(key)
                if len(curated) >= max_insights:
                    break
            self.insights_repository = curated
            logger.info(f"Curated insights repository to {len(curated)} unique insights")
    
    def get_top_strategies(self, n: int = 10) -> List[Strategy]:
        """Get top N strategies by combined score."""
        sorted_archive = sorted(self.archive, key=lambda s: s.score, reverse=True)
        return sorted_archive[:n]

    def record_generation_state(self, generation: int, island_stats: Dict[str, Any]) -> None:
        """Record per-generation snapshots used by visualization and reporting."""
        snapshot = self.get_feature_map_snapshot()
        best_score = float(island_stats.get('best_score_all', 0.0))
        avg_score = float(island_stats.get('average_score_all', 0.0))
        best_sharpe = 0.0
        top_strategies = self.get_top_strategies(1)
        if top_strategies:
            best_sharpe = float(top_strategies[0].metrics.get('sharpe_ratio', 0.0))
        if not np.isfinite(best_score):
            best_score = 0.0
        if not np.isfinite(avg_score):
            avg_score = 0.0
        if not np.isfinite(best_sharpe):
            best_sharpe = 0.0
        self.feature_map_snapshots.append(snapshot)
        self.generation_metrics.append({
            'generation': generation,
            'best_score': best_score,
            'average_score': avg_score,
            'best_sharpe_ratio': best_sharpe,
            'occupancy_rate': float(snapshot.get('occupancy_rate', 0.0)),
            'diversity_score': float(snapshot.get('diversity_score', 0.0)),
            'occupied_cells': int(snapshot.get('occupied_cells', 0)),
            'total_strategies': int(island_stats.get('total_strategies', 0)),
        })
    
    def save_checkpoint(self, filepath: str):
        """Save database state to file."""
        checkpoint_data = {
            'generation_counter': self.generation_counter,
            'total_strategies_evaluated': self.total_strategies_evaluated,
            'feature_map_snapshot': self.get_feature_map_snapshot(),
            'top_strategies': [s.to_dict() for s in self.get_top_strategies(20)],
            'insights_count': len(self.insights_repository),
            'archive_size': len(self.archive)
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load database state from checkpoint file."""
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.generation_counter = checkpoint_data.get('generation_counter', 0)
            self.total_strategies_evaluated = checkpoint_data.get('total_strategies_evaluated', 0)
            
            logger.info(f"Loaded checkpoint from generation {self.generation_counter}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False


class QuantEvolve:
    """
    Main QuantEvolve evolutionary framework.
    
    Implements Algorithm 1 from the paper: multi-agent evolutionary optimization
    with quality-diversity preservation via feature maps.
    
    Attributes:
        config: Configuration dictionary
        database: EvolutionaryDatabase instance
        island_manager: IslandManager instance
        agents: Dictionary of agent instances
        backtesting_engine: BacktestingEngine instance
        evolution_logger: Specialized logger for evolution tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize QuantEvolve framework.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = self._normalize_config(config)
        self.start_time = None
        self.checkpoint_dir = None
        self.log_dir = self.config.get('logging', {}).get('directory', 'results/logs')
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_logging(level=log_level, log_dir=self.log_dir)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("QuantEvolve framework initialized successfully")
        logger.info(f"Configuration: {self.config.get('market', {}).get('type', 'unknown')} market")
        logger.info(f"Generations: {self.config.get('evolution', {}).get('generations', 150)}")
        logger.info(f"Islands: {self.config.get('islands', {}).get('num_islands', 9)}")
        logger.info(f"Feature Map: {self.config.get('feature_map', {}).get('bins_per_dimension', 16)} bins/dimension")

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize legacy/new config keys so runtime modules share one schema.
        """
        cfg = dict(config or {})

        logging_cfg = dict(cfg.get('logging', {}) or {})
        if 'directory' not in logging_cfg and 'log_directory' in logging_cfg:
            logging_cfg['directory'] = logging_cfg['log_directory']
        cfg['logging'] = logging_cfg

        checkpoints_cfg = dict(cfg.get('checkpoints', {}) or {})
        legacy_checkpoint = dict(cfg.get('checkpoint', {}) or {})
        if not checkpoints_cfg:
            checkpoints_cfg = legacy_checkpoint
        if 'interval' not in checkpoints_cfg:
            checkpoints_cfg['interval'] = (
                checkpoints_cfg.get('save_interval')
                or legacy_checkpoint.get('save_interval')
                or cfg.get('evolution', {}).get('checkpoint_interval', 10)
            )
        if 'directory' not in checkpoints_cfg:
            output_results_dir = cfg.get('output', {}).get('results_dir')
            checkpoints_cfg['directory'] = (
                checkpoints_cfg.get('log_directory')
                or (os.path.join(output_results_dir, 'checkpoints') if output_results_dir else None)
                or 'results/checkpoints'
            )
        cfg['checkpoints'] = checkpoints_cfg

        islands_cfg = dict(cfg.get('islands', {}) or {})
        if 'migration_rate' not in islands_cfg and 'migration_percentage' in islands_cfg:
            islands_cfg['migration_rate'] = islands_cfg['migration_percentage']
        if 'migration_interval' not in islands_cfg:
            islands_cfg['migration_interval'] = cfg.get('evolution', {}).get('migration_interval', 10)
        cfg['islands'] = islands_cfg

        bt_cfg = dict(cfg.get('backtesting', {}) or {})
        commission_cfg = dict(bt_cfg.get('commission', {}) or {})
        slippage_cfg = dict(bt_cfg.get('slippage', {}) or {})
        if 'commission_per_share' not in bt_cfg and 'cost' in commission_cfg:
            bt_cfg['commission_per_share'] = commission_cfg['cost']
        if 'min_trade_cost' not in bt_cfg:
            bt_cfg['min_trade_cost'] = commission_cfg.get('min_trade_cost', bt_cfg.get('min_commission', 1.0))
        if 'slippage_volume_limit' not in bt_cfg and 'volume_limit' in slippage_cfg:
            bt_cfg['slippage_volume_limit'] = slippage_cfg['volume_limit']
        if 'slippage_price_impact' not in bt_cfg and 'price_impact' in slippage_cfg:
            bt_cfg['slippage_price_impact'] = slippage_cfg['price_impact']
        cfg['backtesting'] = bt_cfg

        return cfg
    
    def _initialize_components(self):
        """Initialize all system components from configuration."""
        # 1. Feature Map
        self.feature_map = create_feature_map_from_config(self.config)
        logger.info(f"Feature map initialized with {len(self.feature_map.archive)} dimensions")
        
        # 2. Island Manager
        self.island_manager = create_island_manager_from_config(self.config)
        logger.info(f"Island manager initialized with {self.island_manager.num_islands} islands")
        
        # 3. Evolutionary Database
        self.database = EvolutionaryDatabase(feature_map=self.feature_map)
        
        # 4. Agents
        llm_config = self.config.get('llm', {})
        
        self.agents = {
            'data': DataAgent(
                model_name=llm_config.get('fast_model', 'Qwen3-30B-A3B-Instruct-2507'),
                api_key=os.getenv('LLM_API_KEY', ''),
                api_base=os.getenv('LLM_API_BASE', 'http://localhost:8000/v1'),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.9)
            ),
            'research': ResearchAgent(
                model_name=llm_config.get('reasoning_model', 'Qwen3-Next-80B-A3B-Instruct'),
                api_key=os.getenv('LLM_API_KEY', ''),
                api_base=os.getenv('LLM_API_BASE', 'http://localhost:8000/v1'),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.9)
            ),
            'coding': CodingTeam(
                model_name=llm_config.get('fast_model', 'Qwen3-30B-A3B-Instruct-2507'),
                api_key=os.getenv('LLM_API_KEY', ''),
                api_base=os.getenv('LLM_API_BASE', 'http://localhost:8000/v1'),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.9),
                max_iterations=3
            ),
            'evaluation': EvaluationTeam(
                model_name=llm_config.get('reasoning_model', 'Qwen3-Next-80B-A3B-Instruct'),
                api_key=os.getenv('LLM_API_KEY', ''),
                api_base=os.getenv('LLM_API_BASE', 'http://localhost:8000/v1'),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.9)
            )
        }
        
        # 5. Backtesting Engine
        backtest_config = self.config.get('backtesting', {})
        self.backtesting_engine = BacktestingEngine(
            config=backtest_config,
            capital_base=backtest_config.get('capital_base', 100000),
            commission_per_share=backtest_config.get('commission_per_share', 0.0075),
            min_trade_cost=backtest_config.get('min_trade_cost', 1.0),
            slippage_volume_limit=backtest_config.get('slippage_volume_limit', 0.025),
            frequency=backtest_config.get('frequency', 'daily')
        )
        
        # 6. Evolution Logger
        self.evolution_logger = EvolutionLogger(log_dir=self.log_dir)
    
    def run(
        self,
        data: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None,
        resume_from: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> EvolutionaryDatabase:
        """
        Execute the main evolutionary loop (Algorithm 1).
        
        Args:
            data: Optional OHLCV data dictionary. If None, will load from config.
            resume_from_checkpoint: Optional path to checkpoint file to resume from.
            resume_from: Alias of resume_from_checkpoint (backward compatibility).
            output_dir: Optional output directory (accepted for compatibility).
            
        Returns:
            EvolutionaryDatabase: Final state of evolutionary database
        """
        self.start_time = time.time()

        # Backward-compatible argument aliases.
        if resume_from_checkpoint is None and resume_from is not None:
            resume_from_checkpoint = resume_from
        
        # Load data if not provided
        if data is None:
            data = self._load_market_data()
        
        # Setup checkpoint directory
        checkpoints_cfg = self._get_checkpoints_config()
        self.checkpoint_dir = checkpoints_cfg.get('directory', 'results/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        start_generation = 0
        if resume_from_checkpoint:
            if self.database.load_checkpoint(resume_from_checkpoint):
                start_generation = self.database.generation_counter
                logger.info(f"Resuming from generation {start_generation}")
        
        # Get evolution parameters
        evolution_config = self.config.get('evolution', {})
        total_generations = evolution_config.get('generations', 150)
        migration_interval = evolution_config.get('migration_interval', 10)
        insight_curation_interval = evolution_config.get('insight_curation_interval', 50)
        
        # Get market context
        assets_config = self._get_assets_config()
        train_cfg = self.config.get('data_splits', {}).get('train', {})
        train_start = train_cfg.get('start') or train_cfg.get('start_date') or '2015-08-01'
        train_end = train_cfg.get('end') or train_cfg.get('end_date') or '2020-07-31'
        market_context = {
            'market_type': self.config.get('market', {}).get('type', 'equity'),
            'assets': assets_config.get('symbols', []),
            'benchmark': assets_config.get('benchmark', 'SPY'),
            'train_start': train_start,
            'train_end': train_end,
            'frequency': self.config.get('backtesting', {}).get('frequency', 'daily'),
        }
        
        # Get backtesting parameters
        assets = market_context['assets']
        start_date = market_context['train_start']
        end_date = market_context['train_end']
        
        logger.info(f"Starting evolution: {total_generations} generations")
        logger.info(f"Assets: {assets}")
        logger.info(f"Period: {start_date} to {end_date}")

        # Initialize N=C+1 seed strategies via DataAgent (Section 5.1).
        self._initialize_population_with_data_agent(data=data)
        
        # Main evolutionary loop
        for generation in range(start_generation, total_generations):
            self.database.generation_counter = generation
            gen_start_time = time.time()
            self.evolution_logger.log_generation_start(generation, self.island_manager.num_islands)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {generation}/{total_generations}")
            logger.info(f"{'='*60}")
            
            # Process each island
            for island_id in range(self.island_manager.num_islands):
                island = self.island_manager.get_island(island_id)
                
                if island.size() == 0:
                    logger.warning(f"Island {island_id} is empty, skipping")
                    continue
                
                # Step 1: Sample parent (Equation 1)
                alpha = evolution_config.get('alpha', 0.5)
                parent = sample_parent(island.strategies, self.feature_map, alpha=alpha)
                logger.debug(f"Island {island_id}: Selected parent {parent.id} (score={parent.score:.3f})")
                
                # Step 2: Sample cousins (Equation 2)
                cousin_config = self.config.get('cousin_selection', {})
                cousins = sample_cousins(parent, island.strategies, self.feature_map, cousin_config)
                logger.debug(f"Island {island_id}: Selected {len(cousins)} cousins")
                
                # Step 3: Create selection pool for context
                selection_pool = create_selection_pool(parent, cousins)
                
                # Step 4: Research Agent generates hypothesis
                try:
                    parent_context = parent.to_dict()
                    parent_context['category'] = parent.analysis.get('category', 'unknown')
                    cousin_contexts = []
                    for cousin in cousins:
                        cctx = cousin.to_dict()
                        cctx['category'] = cousin.analysis.get('category', 'unknown')
                        cousin_contexts.append(cctx)

                    hypothesis = self.agents['research'].execute(
                        parent_strategy=parent_context,
                        cousin_strategies=cousin_contexts,
                        generation=generation,
                        market_context=market_context
                    )
                    logger.debug(f"Island {island_id}: Generated hypothesis")
                except Exception as e:
                    logger.error(f"Island {island_id}: Research agent failed: {e}")
                    continue
                
                # Step 5: Coding Team generates and tests code
                try:
                    parent_code = parent.code if parent else ""
                    cousin_codes = [c.code for c in cousins if c.code]
                    
                    coding_result = self.agents['coding'].execute(
                        hypothesis=hypothesis,
                        market_context=market_context,
                        assets=assets,
                        start_date=start_date,
                        end_date=end_date,
                        parent_code=parent_code,
                        cousin_codes=cousin_codes
                    )
                    
                    strategy_code = coding_result.get('code', '')
                    metrics = coding_result.get('metrics', {})
                    success = coding_result.get('success', False)
                    error_message = coding_result.get('error', '')
                    
                    if not success:
                        logger.warning(f"Island {island_id}: Backtest failed: {error_message}")
                        continue

                    # Fallback to shared backtesting engine if agent metrics are empty.
                    if not metrics:
                        bt_success, bt_metrics, bt_error = self.backtesting_engine.run(
                            strategy_code=strategy_code,
                            assets=assets,
                            start_date=start_date,
                            end_date=end_date,
                            benchmark=market_context.get('benchmark'),
                        )
                        if bt_success:
                            metrics = bt_metrics
                        else:
                            logger.warning(f"Island {island_id}: Engine backtest fallback failed: {bt_error}")
                    
                    if coding_result.get('returns_series'):
                        self.database.strategy_returns = list(coding_result.get('returns_series', []))
                    if coding_result.get('benchmark_returns'):
                        self.database.benchmark_returns = list(coding_result.get('benchmark_returns', []))

                    logger.debug(f"Island {island_id}: Backtest successful")
                    
                except Exception as e:
                    logger.error(f"Island {island_id}: Coding team failed: {e}")
                    continue
                
                # Step 6: Evaluation Team analyzes results
                try:
                    evaluation_result = self.agents['evaluation'].execute(
                        hypothesis=hypothesis,
                        code=strategy_code,
                        metrics=metrics,
                        returns_series=coding_result.get('returns_series', []),
                        benchmark_returns=coding_result.get('benchmark_returns', []),
                        generation=generation,
                        market_context=market_context,
                        constraints=self.config.get('risk_constraints', {})
                    )
                    
                    categorization = evaluation_result.get('categorization', {})
                    category_bits = categorization.get('category_bits', '00000000')
                    insights = evaluation_result.get('insights', {})
                    analysis = {
                        'category': categorization.get('primary_category', 'unknown'),
                        'category_bits': category_bits,
                        'evaluation_feedback': evaluation_result.get('feedback', {}),
                        'evaluation_scores': {
                            'combined_score': evaluation_result.get('combined_score', 0.0),
                            'hypothesis_score': evaluation_result.get('hypothesis_evaluation', {}).get('hypothesis_evaluation_score', 0.0),
                            'code_score': evaluation_result.get('code_evaluation', {}).get('program_alignment_evaluation_score', 0.0),
                            'results_score': evaluation_result.get('results_analysis', {}).get('results_analysis_score', 0.0),
                        }
                    }
                    
                    logger.debug(f"Island {island_id}: Evaluation complete")
                    
                except Exception as e:
                    logger.error(f"Island {island_id}: Evaluation team failed: {e}")
                    continue
                
                # Step 7: Compute combined score (Equation 3)
                sr = metrics.get('sharpe_ratio', 0.0)
                ir = metrics.get('information_ratio', 0.0)
                mdd = metrics.get('max_drawdown', 0.0)
                combined_score = compute_combined_score(sr, ir, mdd)
                
                # Step 8: Create strategy object
                strategy = Strategy(
                    id=f"gen{generation}_isl{island_id}_s{self.database.total_strategies_evaluated}",
                    hypothesis=hypothesis,
                    code=strategy_code,
                    metrics=metrics,
                    analysis=analysis,
                    feature_vector=[],  # Will be set by database.update()
                    generation=generation,
                    island_id=island_id,
                    score=combined_score,
                    timestamp=datetime.now().isoformat()
                )
                
                # Step 9: Compute feature vector
                feature_vector = self.feature_map.compute_feature_vector(metrics, category_bits)
                
                # Step 10: Update database (handles feature map insertion)
                added_to_map = self.database.update(strategy, feature_vector, island_id)
                
                if added_to_map:
                    logger.info(f"Island {island_id}: Strategy {strategy.id} added to feature map (score={combined_score:.3f})")
                else:
                    logger.debug(f"Island {island_id}: Strategy {strategy.id} archived (not better than existing)")

                # Keep island populations updated across generations.
                self.island_manager.add_strategy_to_island(island_id, strategy)
                
                # Step 11: Add insights to repository
                if insights:
                    insight_payload = {
                        'generation': generation,
                        'island_id': island_id,
                        'strategy_id': strategy.id,
                        'insight': insights
                    }
                    self.database.add_insight(insight_payload)
                    # Feed the research agent with online learnings for next hypotheses.
                    if isinstance(insights, dict):
                        self.agents['research'].add_insight({
                            'strategy_id': strategy.id,
                            'category': analysis.get('category', 'unknown'),
                            'key_finding': insights.get('key_finding', ''),
                            'metric_impact': {
                                'combined_score': combined_score,
                                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                                'max_drawdown': metrics.get('max_drawdown', 0.0),
                            },
                            'suggestion': insights.get('actionability', ''),
                        })
                
                # Log generation progress
                gen_elapsed = time.time() - gen_start_time
                logger.debug(f"Island {island_id} processed in {gen_elapsed:.2f}s")
            
            # Post-generation operations
            
            # Migration (every M generations)
            if migration_interval > 0 and (generation + 1) % migration_interval == 0:
                logger.info(f"Executing migration at generation {generation + 1}")
                self.island_manager.migrate_strategies()
                
                # Log migration statistics
                mig_stats = self.island_manager.get_population_statistics()
                logger.info(f"Migration complete: {mig_stats}")
            
            # Insight curation (every K generations)
            if insight_curation_interval > 0 and (generation + 1) % insight_curation_interval == 0:
                logger.info(f"Curating insights at generation {generation + 1}")
                max_insights = (
                    self.config.get('evolution', {}).get('max_insights')
                    or self.config.get('insights', {}).get('max_size')
                    or 500
                )
                self.database.curate_insights(max_insights=max_insights)
                
                # Update research/evaluation agent insight memory.
                self.agents['research'].curate_insights(max_insights=max_insights)
                self.agents['evaluation'].curate_insights(max_insights=max_insights)

                # Feedback loop: use curated insights to refresh DataAgent's market view.
                if self.database.insights_repository:
                    recent_insights = self.database.insights_repository[-20:]
                    feedback = self._refresh_data_agent_feedback(data, recent_insights)
                    if feedback:
                        market_context['data_agent_feedback'] = feedback

            # Keep each island population bounded when configured.
            max_size_per_island = (
                evolution_config.get('population_size_per_island')
                or evolution_config.get('population_size')
            )
            if max_size_per_island:
                removed = self.island_manager.prune_all_islands(int(max_size_per_island))
                if removed:
                    logger.debug(f"Pruned {removed} strategies across islands")
            
            # Save checkpoint
            checkpoint_interval = self._get_checkpoints_config().get('interval', 10)
            if checkpoint_interval > 0 and (generation + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen{generation + 1}.json")
                self.database.save_checkpoint(checkpoint_path)
                
                # Save feature map visualization
                self._save_generation_summary(generation + 1)
            
            # Log generation summary
            total_elapsed = time.time() - self.start_time
            avg_gen_time = total_elapsed / (generation - start_generation + 1)
            eta_minutes = (total_generations - generation - 1) * avg_gen_time / 60
            
            logger.info(f"Generation {generation + 1} complete: "
                       f"{self.database.total_strategies_evaluated} strategies evaluated, "
                       f"{len(self.feature_map.archive)} cells occupied, "
                       f"ETA: {eta_minutes:.1f} minutes")
            self.database.record_generation_state(
                generation=generation + 1,
                island_stats=self.island_manager.get_population_statistics(),
            )
            if self.database.generation_metrics:
                self.evolution_logger.log_metrics(
                    generation + 1,
                    self.database.generation_metrics[-1]
                )
        
        # Final summary
        total_time = time.time() - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"EVOLUTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Total generations: {total_generations}")
        logger.info(f"Total strategies evaluated: {self.database.total_strategies_evaluated}")
        logger.info(f"Feature map occupancy: {len(self.feature_map.archive)} / {self.feature_map.total_cells} cells")
        top_one = self.database.get_top_strategies(1)
        if top_one:
            logger.info(f"Best strategy score: {top_one[0].score:.3f}")
        else:
            logger.info("Best strategy score: N/A (no valid strategies)")
        
        # Save final results
        self._save_final_results()
        
        return self.database
    
    def _get_assets_config(self) -> Dict[str, Any]:
        """Get asset configuration from either top-level or nested market section."""
        assets_config = self.config.get('assets', {})
        if isinstance(assets_config, dict) and assets_config:
            return assets_config

        market_assets = self.config.get('market', {}).get('assets', {})
        if isinstance(market_assets, dict):
            return market_assets
        return {}

    def _get_checkpoints_config(self) -> Dict[str, Any]:
        """Resolve checkpoint config across legacy/new key styles."""
        checkpoints = self.config.get('checkpoints', {})
        if isinstance(checkpoints, dict) and checkpoints:
            return checkpoints
        legacy = self.config.get('checkpoint', {})
        return legacy if isinstance(legacy, dict) else {}

    def _refresh_data_agent_feedback(
        self,
        data: Dict[str, Any],
        recent_insights: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Re-run lightweight DataAgent analysis conditioned on recent insights.

        This keeps a soft feedback loop without resetting islands.
        """
        insight_lines: List[str] = []
        for item in recent_insights:
            insight = item.get('insight', {})
            if isinstance(insight, dict):
                key_finding = str(insight.get('key_finding', '')).strip()
                actionability = str(insight.get('actionability', '')).strip()
                text = " | ".join([p for p in [key_finding, actionability] if p])
                if text:
                    insight_lines.append(text)

        if not insight_lines:
            return {}

        data_description = (
            "Recent evolution insights:\n- "
            + "\n- ".join(insight_lines[:12])
        )

        use_llm = bool(self.config.get('llm', {}).get('use_data_agent_llm', False))
        schema_result = self.agents['data'].analyze_data_schema(
            market_data=data,
            data_description=data_description,
            use_llm=use_llm,
        )
        category_result = self.agents['data'].identify_strategy_categories(
            schema_result=schema_result,
            num_categories=max(3, min(8, self.island_manager.num_islands - 1)),
            use_llm=use_llm,
        )

        return {
            'updated_at': datetime.now().isoformat(),
            'insight_count': len(insight_lines),
            'top_recommendations': [
                c.get('category_name')
                for c in category_result.get('recommended_categories', [])[:4]
            ],
        }

    def _initialize_population_with_data_agent(self, data: Dict[str, Any]) -> None:
        """
        Initialize seed strategies and seed islands using DataAgent output.

        Each seed strategy is inserted into both island populations and
        the evolutionary database to create valid starting points.
        """
        if any(island.size() > 0 for island in self.island_manager.islands):
            return

        num_categories = max(3, min(8, self.island_manager.num_islands - 1))
        use_llm = bool(self.config.get('llm', {}).get('use_data_agent_llm', False))
        seed_payload = self.agents['data'].execute(
            market_data=data,
            num_categories=num_categories,
            use_llm=use_llm,
        )
        seed_specs = seed_payload.get('seed_strategies', {}).get('seed_strategies', [])

        if not seed_specs:
            logger.warning("DataAgent returned no seeds; islands remain uninitialized.")
            return

        seeded_count = 0
        for i, spec in enumerate(seed_specs):
            if not isinstance(spec, dict):
                continue

            island_id = i % self.island_manager.num_islands
            category_bits = self._seed_category_bits(spec)
            metrics = self._seed_metrics(spec)
            score = compute_combined_score(
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('information_ratio', 0.0),
                metrics.get('max_drawdown', 0.0),
            )
            feature_vector = self.feature_map.compute_feature_vector(metrics, category_bits)

            strategy = Strategy(
                id=spec.get('strategy_id', f'seed_{i}'),
                hypothesis={
                    'hypothesis': spec.get('description', 'Seed strategy from DataAgent initialization.'),
                    'rationale': spec.get('pseudocode', ''),
                    'is_seed': True,
                },
                code=self._seed_code_stub(spec),
                metrics=metrics,
                analysis={
                    'category': spec.get('category', 'unknown'),
                    'category_bits': category_bits,
                    'is_seed': True,
                },
                feature_vector=list(feature_vector),
                generation=0,
                island_id=island_id,
                score=score,
                timestamp=datetime.now().isoformat(),
            )

            self.island_manager.seed_island(island_id, strategy)
            self.database.update(strategy, feature_vector, island_id)
            seeded_count += 1

        logger.info(f"Initialized {seeded_count} seed strategies across {self.island_manager.num_islands} islands")

    def _seed_category_bits(self, seed_spec: Dict[str, Any]) -> str:
        """Convert seed category_bit_index to fixed-length binary string."""
        n_bits = self.feature_map.category_bits
        if seed_spec.get('category') == 'benchmark':
            return '0' * n_bits

        # Prefer explicit category-name mapping so heterogeneous agent taxonomies
        # still map into the feature-map's canonical bit space.
        category = str(seed_spec.get('category', '')).strip().lower()
        canonical_aliases = {
            'momentum': 'momentum',
            'mean_reversion': 'mean_reversion',
            'trend_following': 'trend_following',
            'breakout': 'breakout',
            'volatility': 'statistical_arb',
            'market_microstructure': 'sentiment_analysis',
            'fundamental_value': 'multi_factor',
            'fundamental': 'multi_factor',
            'volume': 'sentiment_analysis',
            'hybrid': 'multi_factor',
            'machine_learning': 'machine_learning',
        }
        canonical_name = canonical_aliases.get(category, category)
        if canonical_name in self.feature_map.STRATEGY_CATEGORIES:
            idx = self.feature_map.STRATEGY_CATEGORIES.index(canonical_name)
            bits = ['0'] * n_bits
            bits[idx] = '1'
            return ''.join(bits)

        idx = seed_spec.get('category_bit_index', None)
        if isinstance(idx, int) and 0 <= idx < n_bits:
            bits = ['0'] * n_bits
            bits[idx] = '1'
            return ''.join(bits)

        return '0' * n_bits

    def _seed_metrics(self, seed_spec: Dict[str, Any]) -> Dict[str, float]:
        """Create lightweight synthetic metrics for initial seed placement."""
        expected = seed_spec.get('expected_characteristics', {}) if isinstance(seed_spec, dict) else {}
        freq_text = str(expected.get('trading_frequency', '')).lower()
        trading_frequency = 0.2
        if 'minimal' in freq_text:
            trading_frequency = 0.01
        elif 'low' in freq_text:
            trading_frequency = 0.1
        elif 'high' in freq_text:
            trading_frequency = 0.6

        if seed_spec.get('category') == 'benchmark':
            trading_frequency = 0.0

        return {
            'sharpe_ratio': 0.1,
            'sortino_ratio': 0.15,
            'information_ratio': 0.0,
            'max_drawdown': -0.1,
            'total_return': 0.02,
            'trading_frequency': trading_frequency,
        }

    def _seed_code_stub(self, seed_spec: Dict[str, Any]) -> str:
        """Build a minimal code stub stored with seed strategies for lineage context."""
        name = seed_spec.get('name', 'Seed Strategy')
        pseudocode = seed_spec.get('pseudocode', '')
        return (
            f"# {name}\n"
            "# Auto-generated seed placeholder from DataAgent.\n"
            f"# Pseudocode:\n# {str(pseudocode).replace(chr(10), chr(10) + '# ')}\n"
        )

    def _load_market_data(self) -> Dict[str, Any]:
        """Load OHLCV data from configured source."""
        data_config = self._get_assets_config()
        data_path = data_config.get('data_path') or data_config.get('data_source') or 'data/equities'
        symbols = data_config.get('symbols', [])
        
        logger.info(f"Loading market data from {data_path}")
        
        data = {}
        for symbol in symbols:
            symbol_data = load_ohlcv_data(symbol=symbol, data_path=data_path)
            if symbol_data is not None:
                data[symbol] = symbol_data
                logger.debug(f"Loaded {len(symbol_data)} rows for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")
        
        if not data:
            raise ValueError("No market data loaded. Check data paths and symbols.")
        
        return data
    
    def _save_generation_summary(self, generation: int):
        """Save summary statistics for a generation."""
        summary = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'feature_map_occupancy': len(self.feature_map.archive),
            'feature_map_occupancy_rate': self.feature_map.get_occupancy_rate(),
            'diversity_score': self.feature_map.get_diversity_score(),
            'total_strategies': self.database.total_strategies_evaluated,
            'insights_count': len(self.database.insights_repository),
            'top_strategies': [
                {
                    'id': s.id,
                    'score': s.score,
                    'sharpe_ratio': s.metrics.get('sharpe_ratio', 0.0),
                    'category_bits': s.feature_vector[0] if s.feature_vector else 'N/A'
                }
                for s in self.database.get_top_strategies(5)
            ],
            'island_statistics': self.island_manager.get_population_statistics()
        }
        
        summary_path = os.path.join(self.checkpoint_dir, f"summary_gen{generation}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _save_final_results(self):
        """Save final evolution results."""
        results_dir = self._get_checkpoints_config().get('directory', 'results/checkpoints')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save final database state
        final_checkpoint = os.path.join(results_dir, "final_checkpoint.json")
        self.database.save_checkpoint(final_checkpoint)
        
        # Save top strategies
        top_strategies = self.database.get_top_strategies(20)
        strategies_path = os.path.join(results_dir, "top_strategies.json")
        with open(strategies_path, 'w') as f:
            json.dump([s.to_dict() for s in top_strategies], f, indent=2, default=str)
        
        # Save feature map snapshot
        feature_map_snapshot = self.database.get_feature_map_snapshot()
        feature_map_path = os.path.join(results_dir, "feature_map_final.json")
        with open(feature_map_path, 'w') as f:
            json.dump(feature_map_snapshot, f, indent=2, default=str)
        
        # Save evolution log
        log_summary = {
            'total_generations': self.database.generation_counter,
            'total_strategies_evaluated': self.database.total_strategies_evaluated,
            'final_feature_map_occupancy': len(self.feature_map.archive),
            'total_insights': len(self.database.insights_repository),
            'best_strategy': self.database.get_top_strategies(1)[0].to_dict() if top_strategies else None,
            'runtime_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
        log_path = os.path.join(results_dir, "evolution_summary.json")
        with open(log_path, 'w') as f:
            json.dump(log_summary, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_dir}")


def create_quant_evolve_from_config(config_path: str) -> QuantEvolve:
    """
    Factory function to create QuantEvolve instance from YAML config.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        QuantEvolve: Initialized framework instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return QuantEvolve(config)


def run_evolution(config_path: str, resume_from: Optional[str] = None) -> EvolutionaryDatabase:
    """
    Convenience function to run complete evolution from config file.
    
    Args:
        config_path: Path to YAML configuration file
        resume_from: Optional checkpoint path to resume from
        
    Returns:
        EvolutionaryDatabase: Final database state
    """
    quant_evolve = create_quant_evolve_from_config(config_path)
    return quant_evolve.run(resume_from_checkpoint=resume_from)
