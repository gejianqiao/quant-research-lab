"""
Performance metrics computation for QuantEvolve.

This module implements all performance metrics used for strategy evaluation,
including Sharpe Ratio, Sortino Ratio, Information Ratio, Max Drawdown,
and the Combined Score (Equation 3 from the paper).

Metrics:
- SR (Sharpe Ratio): (mean(returns) - risk_free_rate) / std(returns)
- SOR (Sortino Ratio): (mean(returns) - risk_free_rate) / std(negative_returns)
- IR (Information Ratio): mean(excess_returns) / std(excess_returns)
- MDD (Max Drawdown): min((portfolio_values - running_max) / running_max)
- Combined Score: SR + IR + MDD (Eq 3)
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, List[float]],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Formula: SR = (mean(returns) - risk_free_rate) / std(returns)
    
    Args:
        returns: Array-like of periodic returns (daily by default)
        risk_free_rate: Risk-free rate (default 0.0 as per paper)
        annualization_factor: Number of periods per year (252 for daily)
    
    Returns:
        float: Annualized Sharpe Ratio
    
    Notes:
        - Returns are assumed to be in decimal form (e.g., 0.01 for 1%)
        - Annualized by multiplying by sqrt(annualization_factor)
        - If std(returns) is 0 or NaN, returns 0.0
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    
    # Filter out NaN and Inf values
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 2:
        logger.warning("Insufficient data for Sharpe Ratio calculation")
        return 0.0
    
    mean_return = np.mean(valid_returns)
    std_return = np.std(valid_returns, ddof=1)  # Sample standard deviation
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    # Calculate periodic Sharpe ratio
    periodic_sr = (mean_return - risk_free_rate) / std_return
    
    # Annualize
    annualized_sr = periodic_sr * np.sqrt(annualization_factor)
    
    return float(annualized_sr)


def compute_sortino_ratio(
    returns: Union[np.ndarray, pd.Series, List[float]],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio.
    
    Formula: SOR = (mean(returns) - risk_free_rate) / std(negative_returns)
    
    Args:
        returns: Array-like of periodic returns
        risk_free_rate: Risk-free rate (default 0.0)
        annualization_factor: Number of periods per year (252 for daily)
        target_return: Target return for downside deviation (default 0.0)
    
    Returns:
        float: Annualized Sortino Ratio
    
    Notes:
        - Only considers negative returns (below target) in denominator
        - More appropriate for asymmetric return distributions
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    
    # Filter out NaN and Inf values
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 2:
        logger.warning("Insufficient data for Sortino Ratio calculation")
        return 0.0
    
    mean_return = np.mean(valid_returns)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = valid_returns[valid_returns < target_return]
    
    if len(downside_returns) < 2:
        # If no negative returns, use a small positive value to avoid division by zero
        logger.info("No negative returns found, using fallback for Sortino Ratio")
        return float((mean_return - risk_free_rate) * np.sqrt(annualization_factor) / 0.0001)
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    # Calculate periodic Sortino ratio
    periodic_sor = (mean_return - risk_free_rate) / downside_std
    
    # Annualize
    annualized_sor = periodic_sor * np.sqrt(annualization_factor)
    
    return float(annualized_sor)


def compute_information_ratio(
    portfolio_returns: Union[np.ndarray, pd.Series, List[float]],
    benchmark_returns: Union[np.ndarray, pd.Series, List[float]],
    annualization_factor: int = 252
) -> float:
    """
    Calculate Information Ratio.
    
    Formula: IR = mean(portfolio_returns - benchmark_returns) / std(excess_returns)
    
    Args:
        portfolio_returns: Array-like of portfolio periodic returns
        benchmark_returns: Array-like of benchmark periodic returns
        annualization_factor: Number of periods per year (252 for daily)
    
    Returns:
        float: Annualized Information Ratio
    
    Notes:
        - Measures active return per unit of tracking error
        - Higher IR indicates better risk-adjusted outperformance
    """
    portfolio_array = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark_array = np.asarray(benchmark_returns, dtype=np.float64)
    
    # Ensure same length
    min_len = min(len(portfolio_array), len(benchmark_array))
    if min_len < 2:
        logger.warning("Insufficient data for Information Ratio calculation")
        return 0.0
    
    portfolio_array = portfolio_array[:min_len]
    benchmark_array = benchmark_array[:min_len]
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(portfolio_array) & np.isfinite(benchmark_array)
    valid_portfolio = portfolio_array[valid_mask]
    valid_benchmark = benchmark_array[valid_mask]
    
    if len(valid_portfolio) < 2:
        logger.warning("Insufficient valid data for Information Ratio calculation")
        return 0.0
    
    # Calculate excess returns
    excess_returns = valid_portfolio - valid_benchmark
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0 or np.isnan(std_excess):
        return 0.0
    
    # Calculate periodic IR
    periodic_ir = mean_excess / std_excess
    
    # Annualize
    annualized_ir = periodic_ir * np.sqrt(annualization_factor)
    
    return float(annualized_ir)


def compute_max_drawdown(
    values: Union[np.ndarray, pd.Series, List[float]]
) -> float:
    """
    Calculate Maximum Drawdown.
    
    Formula: MDD = min((portfolio_values - running_max) / running_max)
    
    Args:
        values: Array-like of portfolio values (cumulative, not returns)
    
    Returns:
        float: Maximum Drawdown (negative value, e.g., -0.15 for -15%)
    
    Notes:
        - Returns negative value (drawdown is a loss)
        - If values are returns, convert to cumulative first
        - MDD of 0 means no drawdown occurred
    """
    values_array = np.asarray(values, dtype=np.float64)
    
    # Filter out NaN and Inf values
    valid_values = values_array[np.isfinite(values_array)]
    
    if len(valid_values) < 2:
        logger.warning("Insufficient data for Max Drawdown calculation")
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(valid_values)
    
    # Avoid division by zero
    running_max = np.where(running_max == 0, 1e-10, running_max)
    
    # Calculate drawdowns
    drawdowns = (valid_values - running_max) / running_max
    
    # Maximum drawdown is the minimum (most negative)
    max_dd = float(np.min(drawdowns))
    
    return max_dd


def compute_max_drawdown_from_returns(
    returns: Union[np.ndarray, pd.Series, List[float]]
) -> float:
    """
    Calculate Maximum Drawdown from returns series.
    
    Converts returns to cumulative values, then calculates MDD.
    
    Args:
        returns: Array-like of periodic returns
    
    Returns:
        float: Maximum Drawdown (negative value)
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    
    # Filter out NaN and Inf values
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 2:
        logger.warning("Insufficient data for Max Drawdown calculation")
        return 0.0
    
    # Convert to cumulative values (starting from 1.0)
    cumulative = np.cumprod(1 + valid_returns)
    
    return compute_max_drawdown(cumulative)


def compute_total_return(
    returns: Union[np.ndarray, pd.Series, List[float]]
) -> float:
    """
    Calculate Total (Cumulative) Return.
    
    Formula: CR = prod(1 + returns) - 1
    
    Args:
        returns: Array-like of periodic returns
    
    Returns:
        float: Total return (e.g., 2.5 for 250%)
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    
    # Filter out NaN and Inf values
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 1:
        logger.warning("Insufficient data for Total Return calculation")
        return 0.0
    
    # Calculate cumulative return
    total_return = float(np.prod(1 + valid_returns) - 1)
    
    return total_return


def compute_calmar_ratio(
    returns: Union[np.ndarray, pd.Series, List[float]],
    annualization_factor: int = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Formula: Calmar = Annualized Return / |Max Drawdown|
    
    Args:
        returns: Array-like of periodic returns
        annualization_factor: Number of periods per year (252 for daily)
    
    Returns:
        float: Calmar Ratio
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 2:
        return 0.0
    
    # Annualized return
    total_return = compute_total_return(valid_returns)
    n_periods = len(valid_returns)
    annualized_return = ((1 + total_return) ** (annualization_factor / n_periods)) - 1
    
    # Max drawdown
    mdd = compute_max_drawdown_from_returns(valid_returns)
    
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    
    return float(annualized_return / abs(mdd))


def compute_combined_score(
    sr: float,
    ir: float,
    mdd: float
) -> float:
    """
    Calculate Combined Score (Equation 3 from the paper).
    
    Formula: Score = SR + IR + MDD
    
    Args:
        sr: Sharpe Ratio
        ir: Information Ratio
        mdd: Maximum Drawdown (negative value)
    
    Returns:
        float: Combined Score
    
    Notes:
        - MDD is already negative, so adding it penalizes drawdown
        - This is the primary optimization objective in QuantEvolve
        - Higher score indicates better overall strategy quality
    """
    # Ensure inputs are valid
    sr = float(sr) if np.isfinite(sr) else 0.0
    ir = float(ir) if np.isfinite(ir) else 0.0
    mdd = float(mdd) if np.isfinite(mdd) else 0.0
    
    # Combined Score = SR + IR + MDD (Eq 3)
    # MDD is negative, so this penalizes large drawdowns
    combined_score = sr + ir + mdd
    
    return float(combined_score)


def compute_all_metrics(
    returns: Union[np.ndarray, pd.Series, List[float]],
    benchmark_returns: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> Dict[str, Any]:
    """
    Compute all performance metrics for a strategy.
    
    Args:
        returns: Array-like of periodic returns
        benchmark_returns: Optional benchmark returns for IR calculation
        risk_free_rate: Risk-free rate (default 0.0)
        annualization_factor: Number of periods per year (252 for daily)
    
    Returns:
        Dict[str, Any]: Dictionary containing all metrics:
            - sharpe_ratio: Sharpe Ratio
            - sortino_ratio: Sortino Ratio
            - information_ratio: Information Ratio (or 0 if no benchmark)
            - max_drawdown: Maximum Drawdown (negative)
            - total_return: Total/Cumulative Return
            - calmar_ratio: Calmar Ratio
            - combined_score: Combined Score (Eq 3)
            - annualized_return: Annualized Return
            - volatility: Annualized Volatility
            - win_rate: Percentage of positive returns
            - avg_win: Average positive return
            - avg_loss: Average negative return
            - profit_factor: Gross Profit / Gross Loss
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) < 2:
        logger.warning("Insufficient data for comprehensive metrics calculation")
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'calmar_ratio': 0.0,
            'combined_score': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    # Basic metrics
    sharpe = compute_sharpe_ratio(valid_returns, risk_free_rate, annualization_factor)
    sortino = compute_sortino_ratio(valid_returns, risk_free_rate, annualization_factor)
    mdd = compute_max_drawdown_from_returns(valid_returns)
    total_ret = compute_total_return(valid_returns)
    calmar = compute_calmar_ratio(valid_returns, annualization_factor)
    
    # Information Ratio (if benchmark provided)
    if benchmark_returns is not None:
        ir = compute_information_ratio(valid_returns, benchmark_returns, annualization_factor)
    else:
        ir = 0.0
    
    # Combined Score (Eq 3)
    combined = compute_combined_score(sharpe, ir, mdd)
    
    # Annualized return
    n_periods = len(valid_returns)
    annualized_ret = ((1 + total_ret) ** (annualization_factor / n_periods)) - 1
    
    # Volatility (annualized)
    volatility = np.std(valid_returns, ddof=1) * np.sqrt(annualization_factor)
    
    # Win/Loss statistics
    positive_returns = valid_returns[valid_returns > 0]
    negative_returns = valid_returns[valid_returns < 0]
    
    win_rate = len(positive_returns) / len(valid_returns) if len(valid_returns) > 0 else 0.0
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
    avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0.0
    
    # Profit Factor
    gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
    gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    if not np.isfinite(profit_factor):
        profit_factor = 0.0
    
    return {
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'information_ratio': float(ir),
        'max_drawdown': float(mdd),
        'total_return': float(total_ret),
        'calmar_ratio': float(calmar),
        'combined_score': float(combined),
        'annualized_return': float(annualized_ret),
        'volatility': float(volatility),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor)
    }


def compute_trading_frequency(
    positions: Union[np.ndarray, pd.Series, List[int]],
    total_periods: int
) -> float:
    """
    Calculate trading frequency (for feature map dimension).
    
    Args:
        positions: Array-like of position indicators (0=no position, 1=long, -1=short)
        total_periods: Total number of periods in the backtest
    
    Returns:
        float: Trading frequency (0.0 to 1.0)
            - 0.0: No trades (buy and hold)
            - 1.0: Trading every period
    """
    positions_array = np.asarray(positions, dtype=np.float64)
    
    if len(positions_array) < 2:
        return 0.0
    
    # Count position changes (trades)
    position_changes = np.sum(np.diff(positions_array) != 0)
    
    # Normalize by total periods
    frequency = position_changes / (total_periods - 1) if total_periods > 1 else 0.0
    
    return float(min(1.0, max(0.0, frequency)))


def validate_metrics(
    metrics: Dict[str, Any],
    min_sharpe: float = -5.0,
    max_sharpe: float = 10.0,
    min_mdd: float = -1.0,
    max_mdd: float = 0.0
) -> bool:
    """
    Validate that metrics are within reasonable bounds.
    
    Args:
        metrics: Dictionary of computed metrics
        min_sharpe: Minimum acceptable Sharpe Ratio
        max_sharpe: Maximum acceptable Sharpe Ratio
        min_mdd: Minimum acceptable MDD (most negative)
        max_mdd: Maximum acceptable MDD (should be 0 or close)
    
    Returns:
        bool: True if metrics are valid, False otherwise
    """
    try:
        sr = metrics.get('sharpe_ratio', 0.0)
        mdd = metrics.get('max_drawdown', 0.0)
        
        if not (min_sharpe <= sr <= max_sharpe):
            logger.warning(f"Sharpe Ratio {sr} outside valid range [{min_sharpe}, {max_sharpe}]")
            return False
        
        if not (min_mdd <= mdd <= max_mdd):
            logger.warning(f"Max Drawdown {mdd} outside valid range [{min_mdd}, {max_mdd}]")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating metrics: {e}")
        return False
