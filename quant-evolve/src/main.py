#!/usr/bin/env python3
"""
QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework

Main entry point for running the QuantEvolve system.

Usage:
    python src/main.py --config equity --generations 150
    python src/main.py --config futures --generations 100
    python src/main.py --config equity --resume results/checkpoints/generation_50.pkl
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# Add src directory to path for script-mode imports.
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Package mode: python -m src.main
    from .evolutionary.algorithm import QuantEvolve, EvolutionaryDatabase, run_evolution
    from .utils.logging import setup_logging, EvolutionLogger
    from .utils.visualization import (
        plot_feature_map_evolution,
        plot_sharpe_vs_generation,
        plot_cumulative_returns,
        plot_category_distribution,
        plot_insight_timeline
    )
except ImportError:
    # Script mode: python src/main.py
    from evolutionary.algorithm import QuantEvolve, EvolutionaryDatabase, run_evolution
    from utils.logging import setup_logging, EvolutionLogger
    from utils.visualization import (
        plot_feature_map_evolution,
        plot_sharpe_vs_generation,
        plot_cumulative_returns,
        plot_category_distribution,
        plot_insight_timeline
    )

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try default config locations
        possible_paths = [
            Path(__file__).parent.parent / "config" / f"{config_path}_config.yaml",
            Path(__file__).parent.parent / "config" / f"{config_path}.yaml",
            Path("config") / f"{config_path}_config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                config_file = path
                break
        else:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}. "
                f"Tried: {[str(p) for p in possible_paths]}"
            )
    
    logger.info(f"Loading configuration from: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle YAML defaults directive if present
    if 'defaults' in config:
        default_config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                default_config = yaml.safe_load(f)
            
            # Merge configs (config overrides default)
            config = _deep_merge(default_config, config)
            del config['defaults']
    
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="QuantEvolve: Multi-Agent Evolutionary Framework for Trading Strategy Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run equity market evolution (150 generations)
  python src/main.py --config equity
  
  # Run futures market evolution (100 generations)
  python src/main.py --config futures --generations 100
  
  # Resume from checkpoint
  python src/main.py --config equity --resume results/checkpoints/generation_50.pkl
  
  # Custom generation count
  python src/main.py --config equity --generations 200
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="equity",
        help="Configuration file name (without .yaml extension). Options: equity, futures, default (default: equity)"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Full path to configuration file (overrides --config)"
    )
    
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=None,
        help="Number of generations to run (overrides config file setting)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization generation"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing backtests (for testing configuration)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel backtesting workers (default: 1)"
    )
    
    return parser.parse_args()


def setup_output_directories(output_dir: str) -> Dict[str, Path]:
    """
    Create output directory structure for results.
    
    Args:
        output_dir: Base output directory path
        
    Returns:
        Dictionary of directory paths
    """
    base_path = Path(output_dir)
    
    directories = {
        "base": base_path,
        "feature_maps": base_path / "feature_maps",
        "strategies": base_path / "strategies",
        "metrics": base_path / "metrics",
        "insights": base_path / "insights",
        "checkpoints": base_path / "checkpoints",
        "visualizations": base_path / "visualizations",
        "logs": base_path / "logs"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories created at: {base_path.absolute()}")
    
    return directories


def run_quant_evolve(args: argparse.Namespace) -> Optional[EvolutionaryDatabase]:
    """
    Main execution function for QuantEvolve.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        EvolutionaryDatabase with results, or None if failed
    """
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, log_dir=args.output_dir)
    
    logger.info("=" * 80)
    logger.info("QuantEvolve: Multi-Agent Evolutionary Framework")
    logger.info("=" * 80)
    
    # Load configuration
    try:
        if args.config_path:
            config = load_config(args.config_path)
        else:
            config = load_config(args.config)
        
        logger.info(f"Configuration loaded: {args.config}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None
    
    # Override config with command-line arguments
    if args.generations is not None:
        config['evolution']['generations'] = args.generations
        logger.info(f"Overriding generations: {args.generations}")
    
    # Setup output directories
    directories = setup_output_directories(args.output_dir)
    
    # Initialize evolution logger
    evolution_logger = EvolutionLogger(
        log_dir=directories["logs"],
        save_interval=10
    )
    
    try:
        # Initialize QuantEvolve framework
        logger.info("Initializing QuantEvolve framework...")
        
        quant_evolve = QuantEvolve(config=config)
        
        # Set output paths
        quant_evolve.output_dir = directories["base"]
        quant_evolve.checkpoint_dir = directories["checkpoints"]
        quant_evolve.evolution_logger = evolution_logger
        
        # Run evolution
        logger.info(f"Starting evolution for {config['evolution']['generations']} generations...")
        logger.info(f"Market: {config.get('market', {}).get('type', 'unknown')}")
        logger.info(f"Assets: {config.get('assets', {}).get('symbols', [])}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: Skipping actual backtesting")
            # In dry run mode, we would just validate configuration
            logger.info("Configuration validated successfully")
            return None
        
        # Execute evolutionary loop
        database = quant_evolve.run(
            resume_from=args.resume,
            output_dir=directories["base"]
        )
        
        logger.info("Evolution completed successfully!")
        
        # Generate visualizations
        if not args.no_visualization:
            logger.info("Generating visualizations...")
            generate_visualizations(database, directories["visualizations"], config)
        
        # Save final summary
        save_final_summary(database, directories["metrics"], config)
        
        logger.info(f"All results saved to: {directories['base'].absolute()}")
        
        return database
        
    except KeyboardInterrupt:
        logger.warning("Evolution interrupted by user")
        return None
        
    except Exception as e:
        logger.error(f"Evolution failed with error: {e}", exc_info=True)
        return None


def generate_visualizations(database: EvolutionaryDatabase, output_dir: Path, config: Dict[str, Any]):
    """
    Generate all visualization plots from evolution results.
    
    Args:
        database: EvolutionaryDatabase with results
        output_dir: Output directory for visualizations
        config: Configuration dictionary
    """
    try:
        # Feature map evolution
        plot_feature_map_evolution(
            database.feature_map_snapshots,
            output_dir / "feature_map_evolution.png"
        )
        
        # Sharpe ratio vs generation
        plot_sharpe_vs_generation(
            database.generation_metrics,
            output_dir / "sharpe_vs_generation.png"
        )
        
        # Cumulative returns
        if hasattr(database, 'strategy_returns'):
            plot_cumulative_returns(
                database.strategy_returns,
                database.benchmark_returns,
                output_dir / "cumulative_returns.png"
            )
        
        # Category distribution
        category_dist = database.feature_map.get_category_distribution()
        plot_category_distribution(
            category_dist,
            output_dir / "category_distribution.png"
        )
        
        # Insight timeline
        if hasattr(database, 'insights_timeline'):
            plot_insight_timeline(
                database.insights_timeline,
                output_dir / "insight_timeline.png"
            )
        
        logger.info(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to generate some visualizations: {e}")


def save_final_summary(database: EvolutionaryDatabase, output_dir: Path, config: Dict[str, Any]):
    """
    Save final summary of evolution results.
    
    Args:
        database: EvolutionaryDatabase with results
        output_dir: Output directory for metrics
        config: Configuration dictionary
    """
    import json
    from datetime import datetime
    
    # Get top strategies
    top_strategies = database.get_top_strategies(n=10)
    
    # Compile summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "market": config.get('market', {}).get('type', 'unknown'),
            "generations": config['evolution']['generations'],
            "islands": config['islands']['num_islands'],
            "alpha": config['evolution']['alpha'],
            "bins": config['feature_map']['bins_per_dimension']
        },
        "final_statistics": {
            "total_strategies_evaluated": database.total_strategies_evaluated,
            "feature_map_occupancy": database.feature_map.get_occupancy_rate(),
            "diversity_score": database.feature_map.get_diversity_score(),
            "insights_collected": len(database.insights_repository)
        },
        "top_strategies": [
            {
                "id": s.id,
                "score": s.score,
                "sharpe_ratio": s.metrics.get('sharpe_ratio', 0),
                "total_return": s.metrics.get('total_return', 0),
                "max_drawdown": s.metrics.get('max_drawdown', 0),
                "category": s.analysis.get('category', 'unknown'),
                "generation": s.generation
            }
            for s in top_strategies
        ]
    }
    
    # Save summary
    summary_path = output_dir / "evolution_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Final summary saved to: {summary_path}")
    
    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("EVOLUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Strategies Evaluated: {summary['final_statistics']['total_strategies_evaluated']}")
    logger.info(f"Feature Map Occupancy: {summary['final_statistics']['feature_map_occupancy']:.2%}")
    logger.info(f"Diversity Score: {summary['final_statistics']['diversity_score']:.3f}")
    logger.info(f"Insights Collected: {summary['final_statistics']['insights_collected']}")
    logger.info("\nTop 3 Strategies:")
    for i, s in enumerate(summary['top_strategies'][:3], 1):
        logger.info(f"  {i}. ID={s['id']}, Score={s['score']:.3f}, SR={s['sharpe_ratio']:.3f}, Return={s['total_return']:.2%}")
    logger.info("=" * 80)


def main():
    """
    Main entry point.
    """
    args = parse_arguments()
    
    # Run QuantEvolve
    database = run_quant_evolve(args)
    
    # Exit with appropriate code
    if database is not None or args.dry_run:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
