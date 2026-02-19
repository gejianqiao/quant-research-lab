"""
Backtesting module for QuantEvolve.

Provides Zipline-based backtesting engine with commission/slippage models,
performance metrics calculation, and strategy validation.
"""

from .engine import BacktestingEngine
from .metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_information_ratio,
    compute_max_drawdown,
    compute_combined_score,
    compute_all_metrics
)
from .slippage import VolumeShareSlippage, VolumeShareSlippageModel

__all__ = [
    'BacktestingEngine',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_information_ratio',
    'compute_max_drawdown',
    'compute_combined_score',
    'compute_all_metrics',
    'VolumeShareSlippage',
    'VolumeShareSlippageModel'
]
