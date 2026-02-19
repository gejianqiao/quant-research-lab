"""
Prompts module for QuantEvolve.

This module contains all prompt templates used by the multi-agent system,
including prompts for data analysis, hypothesis generation, code creation,
and evaluation.
"""

from .data_agent_prompts import (
    DATA_SCHEMA_PROMPT,
    CATEGORY_IDENTIFICATION_PROMPT,
    SEED_STRATEGY_PROMPT,
)

from .research_agent_prompts import (
    HYPOTHESIS_GENERATION_PROMPT,
    XML_STRUCTURE_TEMPLATE,
)

from .coding_team_prompts import (
    CODE_GENERATION_PROMPT,
    CODE_DEBUGGING_PROMPT,
    ZIPLINE_STRUCTURE_TEMPLATE,
)

from .evaluation_team_prompts import (
    HYPOTHESIS_EVALUATION_PROMPT,
    CODE_EVALUATION_PROMPT,
    RESULTS_ANALYSIS_PROMPT,
    INSIGHT_EXTRACTION_PROMPT,
    STRATEGY_CATEGORIZATION_PROMPT,
)

__all__ = [
    # Data Agent Prompts
    'DATA_SCHEMA_PROMPT',
    'CATEGORY_IDENTIFICATION_PROMPT',
    'SEED_STRATEGY_PROMPT',
    
    # Research Agent Prompts
    'HYPOTHESIS_GENERATION_PROMPT',
    'XML_STRUCTURE_TEMPLATE',
    
    # Coding Team Prompts
    'CODE_GENERATION_PROMPT',
    'CODE_DEBUGGING_PROMPT',
    'ZIPLINE_STRUCTURE_TEMPLATE',
    
    # Evaluation Team Prompts
    'HYPOTHESIS_EVALUATION_PROMPT',
    'CODE_EVALUATION_PROMPT',
    'RESULTS_ANALYSIS_PROMPT',
    'INSIGHT_EXTRACTION_PROMPT',
    'STRATEGY_CATEGORIZATION_PROMPT',
]
