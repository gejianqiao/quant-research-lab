"""
Multi-Agent System for QuantEvolve.

This package implements the multi-agent architecture that drives strategy discovery:
- DataAgent: Analyzes market data and generates seed strategies
- ResearchAgent: Generates hypothesis in XML format
- CodingTeam: Translates hypotheses to executable trading code
- EvaluationTeam: Analyzes results and extracts insights

Agents use LLMs (Qwen3-30B/80B) to perform their specialized tasks.
"""

from .base_agent import BaseAgent
from .data_agent import DataAgent
from .research_agent import ResearchAgent
from .coding_team import CodingTeam
from .evaluation_team import EvaluationTeam

__all__ = [
    'BaseAgent',
    'DataAgent',
    'ResearchAgent',
    'CodingTeam',
    'EvaluationTeam',
]
