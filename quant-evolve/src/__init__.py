"""
QuantEvolve: Multi-Agent Evolutionary Framework for Quantitative Strategy Discovery

This package implements the core QuantEvolve system, combining quality-diversity
optimization with hypothesis-driven LLM agents to automatically generate diverse,
high-performing trading strategies.

Main Components:
- evolutionary: Feature map, island manager, selection mechanisms, evolutionary algorithm
- agents: Multi-agent system (Data, Research, Coding, Evaluation agents)
- backtesting: Zipline wrapper, metrics computation, slippage models
- prompts: LLM prompt templates for all agents
- utils: Logging, visualization, data loading utilities

Usage:
    from src.evolutionary.algorithm import QuantEvolve
    from src.main import run_evolution

Author: QuantEvolve Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "QuantEvolve Team"

# Package structure
__all__ = [
    "evolutionary",
    "agents",
    "backtesting",
    "prompts",
    "utils",
]
