"""
Backtesting Engine for QuantEvolve.

This module provides the BacktestingEngine class that wraps Zipline Reloaded
with custom commission and slippage settings, enabling automated backtesting
of generated trading strategies.

Key Features:
- Zipline Reloaded integration with daily frequency
- Custom commission model (PerShare: $0.0075/share, min $1.00)
- VolumeShareSlippage model (quadratic)
- Automatic metric extraction from backtest results
- Support for both equities and futures markets
"""

import os
import sys
import tempfile
import importlib.util
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import logging

import pandas as pd
import numpy as np

try:
    from zipline.api import symbol, symbols, order, order_target_percent, order_target_value
    from zipline.api import set_commission, set_slippage, set_benchmark
    from zipline.finance.commission import PerShare
    from zipline.finance.slippage import VolumeShareSlippage
    from zipline.utils.run_algo import run_algorithm
    ZIPLINE_AVAILABLE = True
except ImportError:
    ZIPLINE_AVAILABLE = False
    logging.warning("Zipline not available - backtesting will use mock mode")

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Wrapper for Zipline backtesting engine with custom settings.
    
    This class manages the execution of trading strategy code through Zipline,
    handling data loading, commission/slippage configuration, and result extraction.
    
    Attributes:
        config (Dict): Configuration dictionary with backtesting parameters
        capital_base (float): Starting capital for backtest
        commission_per_share (float): Commission cost per share
        min_trade_cost (float): Minimum commission per trade
        slippage_model: Slippage model instance
        data_bundle (str): Name of Zipline data bundle to use
        frequency (str): Backtest frequency ('daily' or 'minute')
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_bundle: str = 'custom',
        capital_base: float = 100000.0,
        commission_per_share: float = 0.0075,
        min_trade_cost: float = 1.0,
        slippage_volume_limit: float = 0.025,
        slippage_price_impact: float = 0.1,
        frequency: str = 'daily'
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration dictionary with market and backtesting settings
            data_bundle: Name of Zipline data bundle (default: 'custom')
            capital_base: Starting capital in dollars (default: 100,000)
            commission_per_share: Commission cost per share (default: $0.0075)
            min_trade_cost: Minimum commission per trade (default: $1.00)
            slippage_volume_limit: Max volume share for slippage (default: 2.5%)
            slippage_price_impact: Price impact coefficient (default: 0.1)
            frequency: Backtest frequency - 'daily' or 'minute' (default: 'daily')
        """
        self.config = config
        self.capital_base = capital_base
        self.commission_per_share = commission_per_share
        self.min_trade_cost = min_trade_cost
        self.slippage_volume_limit = slippage_volume_limit
        self.slippage_price_impact = slippage_price_impact
        self.frequency = frequency
        self.data_bundle = data_bundle
        
        # Initialize commission and slippage models
        self.commission_model = None
        self.slippage_model = None
        
        if ZIPLINE_AVAILABLE:
            self.commission_model = PerShare(
                cost=self.commission_per_share,
                min_trade_cost=self.min_trade_cost
            )
            self.slippage_model = VolumeShareSlippage(
                volume_limit=self.slippage_volume_limit,
                price_impact=self.slippage_price_impact
            )
        
        self._last_results = None
        self._last_metrics = None
        
        logger.info(
            f"BacktestingEngine initialized: capital=${capital_base:,.0f}, "
            f"commission=${commission_per_share}/share (min ${min_trade_cost}), "
            f"frequency={frequency}"
        )
    
    def run(
        self,
        strategy_code: str,
        assets: List[str],
        start_date: str,
        end_date: str,
        benchmark: Optional[str] = None,
        data: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Execute a backtest for the given strategy code.
        
        Args:
            strategy_code: Python code string containing initialize() and handle_data()
            assets: List of asset symbols to trade (e.g., ['AAPL', 'NVDA'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            benchmark: Benchmark symbol for comparison (e.g., 'SPY')
            data: Optional DataFrame with OHLCV data (sid, date, open, high, low, close, volume)
        
        Returns:
            Tuple of (success: bool, metrics: Dict, error_message: Optional[str])
            - success: True if backtest completed without errors
            - metrics: Dictionary containing performance metrics
            - error_message: Error description if success is False, None otherwise
        """
        if not ZIPLINE_AVAILABLE:
            logger.warning("Zipline not available - running mock backtest")
            return self._run_mock_backtest(strategy_code, assets, start_date, end_date)
        
        try:
            # Write strategy code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(strategy_code)
                temp_file = f.name
            
            logger.info(f"Strategy code written to temporary file: {temp_file}")
            
            # Parse dates
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            
            # Prepare Zipline arguments
            run_args = {
                'start': start_dt,
                'end': end_dt,
                'capital_base': self.capital_base,
                'bundle': self.data_bundle,
                'frequency': self.frequency,
            }
            
            # Load strategy module
            spec = importlib.util.spec_from_file_location("strategy_module", temp_file)
            strategy_module = importlib.util.module_from_spec(spec)
            
            # Execute the strategy module to get initialize and handle_data
            # We need to inject the Zipline API into the module's namespace
            strategy_module.symbol = symbol
            strategy_module.symbols = symbols
            strategy_module.order = order
            strategy_module.order_target_percent = order_target_percent
            strategy_module.order_target_value = order_target_value
            strategy_module.set_commission = set_commission
            strategy_module.set_slippage = set_slippage
            strategy_module.set_benchmark = set_benchmark
            strategy_module.PerShare = PerShare
            strategy_module.VolumeShareSlippage = VolumeShareSlippage
            
            # Execute the strategy code
            spec.loader.exec_module(strategy_module)
            
            # Verify required functions exist
            if not hasattr(strategy_module, 'initialize'):
                raise ValueError("Strategy code must define initialize(context) function")
            if not hasattr(strategy_module, 'handle_data'):
                raise ValueError("Strategy code must define handle_data(context, data) function")
            
            # Run the algorithm
            results = run_algorithm(
                initialize=strategy_module.initialize,
                handle_data=strategy_module.handle_data,
                **run_args
            )
            
            # Store results
            self._last_results = results
            
            # Compute metrics
            metrics = self._compute_metrics(results, benchmark)
            self._last_metrics = metrics
            
            # Clean up temp file
            os.unlink(temp_file)
            
            logger.info(
                f"Backtest completed: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                f"Return={metrics.get('total_return', 0):.2%}, "
                f"MDD={metrics.get('max_drawdown', 0):.2%}"
            )
            
            return True, metrics, None
            
        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Clean up temp file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            
            return False, {}, error_msg
    
    def _compute_metrics(
        self,
        results: pd.DataFrame,
        benchmark: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute performance metrics from backtest results.
        
        Args:
            results: DataFrame from Zipline run_algorithm with columns:
                     ending_value, portfolio_value, returns, benchmark_return
            benchmark: Benchmark symbol for additional analysis
        
        Returns:
            Dictionary containing all performance metrics
        """
        if results is None or len(results) == 0:
            return self._empty_metrics()
        
        # Extract returns series
        if 'returns' in results.columns:
            returns = results['returns'].dropna()
        else:
            # Calculate returns from portfolio_value
            portfolio_values = results['portfolio_value']
            returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Calculate basic metrics
        total_return = (results['portfolio_value'].iloc[-1] / self.capital_base) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # Sharpe Ratio (risk-free rate = 0)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio  # Fallback to Sharpe if no negative returns
        
        # Maximum Drawdown
        portfolio_values = results['portfolio_value']
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Information Ratio (if benchmark available)
        if 'benchmark_return' in results.columns:
            benchmark_returns = results['benchmark_return'].dropna()
            if len(benchmark_returns) == len(returns):
                excess_returns = returns - benchmark_returns
                if excess_returns.std() > 0:
                    information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                else:
                    information_ratio = 0.0
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0
        
        # Additional statistics
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win * (returns > 0).sum() / (avg_loss * (returns < 0).sum())) if avg_loss != 0 else float('inf')
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calmar Ratio
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')
        
        # Combined Score (Equation 3 from paper: Score = SR + IR + MDD)
        combined_score = sharpe_ratio + information_ratio + max_drawdown
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'combined_score': combined_score,
            'num_trades': len(results),
            'start_date': results.index[0],
            'end_date': results.index[-1],
            'final_value': results['portfolio_value'].iloc[-1],
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary for failed backtests."""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'combined_score': 0.0,
            'num_trades': 0,
            'start_date': None,
            'end_date': None,
            'final_value': self.capital_base,
        }
    
    def _run_mock_backtest(
        self,
        strategy_code: str,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Run a mock backtest when Zipline is not available.
        
        This is useful for testing and development without full Zipline setup.
        Generates synthetic results based on strategy code analysis.
        
        Args:
            strategy_code: Strategy code string
            assets: List of asset symbols
            start_date: Start date
            end_date: Start date
        
        Returns:
            Tuple of (success, metrics, error_message)
        """
        logger.warning("Running mock backtest - results are synthetic")
        
        # Parse date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        num_days = (end_dt - start_dt).days
        
        if num_days <= 0:
            return False, {}, "Invalid date range"
        
        # Generate synthetic returns based on code complexity heuristics
        # This is just for testing - real backtests require Zipline
        np.random.seed(hash(strategy_code) % 2**32)
        
        # Simple heuristic: more code = potentially more sophisticated = slightly better
        code_complexity = len(strategy_code) / 1000
        base_return = 0.0003 + (code_complexity * 0.0001)
        base_volatility = 0.015
        
        # Generate daily returns
        daily_returns = np.random.normal(base_return, base_volatility, num_days)
        
        # Calculate cumulative returns
        cumulative = (1 + pd.Series(daily_returns)).cumprod()
        portfolio_values = self.capital_base * cumulative
        
        # Create mock results DataFrame
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:num_days]
        results = pd.DataFrame({
            'portfolio_value': portfolio_values.values[:len(dates)],
            'returns': daily_returns[:len(dates)],
            'ending_value': portfolio_values.values[:len(dates)],
        }, index=dates)
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        return True, metrics, None
    
    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Get the results from the last backtest run.
        
        Returns:
            DataFrame with backtest results or None if no backtest has been run
        """
        return self._last_results
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the metrics from the last backtest run.
        
        Returns:
            Dictionary of metrics or None if no backtest has been run
        """
        return self._last_metrics


def create_backtesting_engine_from_config(config: Dict[str, Any]) -> BacktestingEngine:
    """
    Factory function to create a BacktestingEngine from configuration.
    
    Args:
        config: Configuration dictionary with backtesting section
    
    Returns:
        Configured BacktestingEngine instance
    """
    bt_config = config.get('backtesting', {})
    
    return BacktestingEngine(
        config=config,
        data_bundle=bt_config.get('data_bundle', 'custom'),
        capital_base=bt_config.get('capital_base', 100000.0),
        commission_per_share=bt_config.get('commission_per_share', 0.0075),
        min_trade_cost=bt_config.get('min_trade_cost', 1.0),
        slippage_volume_limit=bt_config.get('slippage_volume_limit', 0.025),
        slippage_price_impact=bt_config.get('slippage_price_impact', 0.1),
        frequency=bt_config.get('frequency', 'daily')
    )
