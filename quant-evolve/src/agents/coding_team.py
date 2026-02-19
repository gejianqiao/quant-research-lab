"""
Coding Team Agent - Translates hypotheses into executable Zipline trading strategies.

This module implements the CodingTeam class responsible for:
1. Translating XML hypotheses into Python code with initialize() and handle_data() functions
2. Enforcing Zipline structure (commission, slippage, no lookahead bias)
3. Running backtests via Zipline Reloaded
4. Iteratively debugging and regenerating code if backtest fails (max 3 iterations)
"""

import os
import sys
import tempfile
import traceback
from typing import Dict, Any, List, Optional, Tuple
from logging import getLogger

import pandas as pd

from .base_agent import BaseAgent

logger = getLogger(__name__)


class CodingTeam(BaseAgent):
    """
    Coding Team Agent - Generates and tests trading strategy code.
    
    Translates research hypotheses into executable Zipline strategies,
    runs backtests, and iteratively refines code based on errors.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen3-30B-A3B-Instruct-2507",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        max_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize CodingTeam agent.
        
        Args:
            model_name: LLM model for code generation (fast model recommended)
            api_key: API key for LLM service
            api_base: Base URL for LLM API
            temperature: Generation temperature for creativity
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens in generated code
            max_iterations: Maximum debug/regenerate cycles
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        self.max_iterations = max_iterations
        self.backtest_results = []
        
    def generate_code(
        self,
        hypothesis: Dict[str, Any],
        market_context: Dict[str, Any],
        parent_code: Optional[str] = None,
        cousin_codes: Optional[List[str]] = None
    ) -> str:
        """
        Generate trading strategy code from hypothesis.
        
        Args:
            hypothesis: Dictionary containing hypothesis components from ResearchAgent
            market_context: Market data schema and asset information
            parent_code: Optional parent strategy code for reference
            cousin_codes: Optional list of cousin strategy codes for reference
            
        Returns:
            str: Generated Python code for Zipline strategy
        """
        # Build the coding prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            hypothesis=hypothesis,
            market_context=market_context,
            parent_code=parent_code,
            cousin_codes=cousin_codes
        )
        
        # Generate code via LLM
        logger.info("Generating strategy code from hypothesis...")
        raw_response = self.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=2
        )
        
        # Extract code from response (handle markdown formatting)
        code = self._extract_code(raw_response)
        
        logger.info(f"Generated code length: {len(code)} characters")
        return code
    
    def run_backtest(
        self,
        code: str,
        assets: List[str],
        start_date: str,
        end_date: str,
        bundle: str = "equities",
        capital_base: float = 100000.0,
        commission_per_share: float = 0.0075,
        min_trade_cost: float = 1.0
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Run Zipline backtest on generated strategy code.
        
        Args:
            code: Python strategy code to test
            assets: List of asset symbols to trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            capital_base: Starting capital
            commission_per_share: Commission cost per share
            min_trade_cost: Minimum trade cost
            
        Returns:
            Tuple[bool, Dict, Optional[str]]: (success, metrics_dict, error_message)
        """
        logger.info("Running Zipline backtest...")
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Import and run the strategy
                metrics = self._execute_backtest(
                    temp_file=temp_file,
                    assets=assets,
                    start_date=start_date,
                    end_date=end_date,
                    bundle=bundle,
                    capital_base=capital_base,
                    commission_per_share=commission_per_share,
                    min_trade_cost=min_trade_cost
                )
                
                # Clean up temp file
                os.unlink(temp_file)
                
                logger.info(f"Backtest completed successfully")
                return True, metrics, None
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Backtest execution failed: {error_msg}")
                os.unlink(temp_file)
                return False, {}, error_msg
                
        except Exception as e:
            error_msg = f"Failed to write temp file: {str(e)}"
            logger.error(error_msg)
            return False, {}, error_msg
    
    def generate_and_test(
        self,
        hypothesis: Dict[str, Any],
        market_context: Dict[str, Any],
        assets: List[str],
        start_date: str,
        end_date: str,
        parent_code: Optional[str] = None,
        cousin_codes: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any], bool, str]:
        """
        Generate code and run backtest with iterative debugging.
        
        Args:
            hypothesis: Research hypothesis dictionary
            market_context: Market data and configuration
            assets: List of asset symbols
            start_date: Backtest start date
            end_date: Backtest end date
            parent_code: Optional parent strategy code
            cousin_codes: Optional cousin strategy codes
            
        Returns:
            Tuple[str, Dict, bool, str]: (code, metrics, success, error_message)
        """
        last_error = None
        
        for iteration in range(self.max_iterations):
            logger.info(f"Coding iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate code
            code = self.generate_code(
                hypothesis=hypothesis,
                market_context=market_context,
                parent_code=parent_code,
                cousin_codes=cousin_codes
            )
            
            # Run backtest
            success, metrics, error = self.run_backtest(
                code=code,
                assets=assets,
                start_date=start_date,
                end_date=end_date,
                bundle=(
                    "futures"
                    if str(market_context.get("market_type", "")).lower() == "futures"
                    else "equities"
                )
            )
            
            if success:
                logger.info(f"Strategy passed backtest on iteration {iteration + 1}")
                return code, metrics, True, ""
            
            last_error = error
            
            # Prepare error context for next iteration
            error_context = {
                "error": error,
                "traceback": traceback.format_exc()
            }
            
            # Regenerate with error feedback
            logger.info(f"Backtest failed, regenerating with error feedback...")
            hypothesis["previous_error"] = error_context
            
        logger.warning(f"Failed to generate valid strategy after {self.max_iterations} iterations")
        return code, {}, False, last_error
    
    def execute(
        self,
        hypothesis: Dict[str, Any],
        market_context: Dict[str, Any],
        assets: List[str],
        start_date: str,
        end_date: str,
        parent_code: Optional[str] = None,
        cousin_codes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the coding team workflow.
        
        Args:
            hypothesis: Research hypothesis from ResearchAgent
            market_context: Market configuration and data schema
            assets: List of asset symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            parent_code: Optional parent strategy code
            cousin_codes: Optional cousin strategy codes
            
        Returns:
            Dict[str, Any]: Result dictionary with code, metrics, and status
        """
        code, metrics, success, error = self.generate_and_test(
            hypothesis=hypothesis,
            market_context=market_context,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            parent_code=parent_code,
            cousin_codes=cousin_codes
        )
        
        return {
            "code": code,
            "metrics": metrics,
            "success": success,
            "error": error,
            "iterations_used": self.max_iterations if not success else 1,
            "returns_series": metrics.get("returns_series", []),
            "benchmark_returns": metrics.get("benchmark_returns", []),
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for code generation."""
        return """You are an expert quantitative developer specializing in Zipline backtesting framework.
Your task is to translate trading strategy hypotheses into executable Python code.

CRITICAL REQUIREMENTS:
1. Code MUST have exactly two functions: initialize(context) and handle_data(context, data)
2. Use ONLY data.history() for price data - NO lookahead bias allowed
3. Always check available cash before ordering
4. Handle all exceptions gracefully
5. Use commission.PerShare(cost=0.0075, min_trade_cost=1.0)
6. Use slippage.VolumeShareSlippage()
7. Code must be complete and runnable - no placeholders

STRATEGY STRUCTURE:
```python
from zipline import run_algorithm
from zipline.api import symbol, order, order_target_percent, record
from zipline.finance import commission, slippage
import pandas as pd
import numpy as np

def initialize(context):
    # Set assets
    context.assets = [symbol('AAPL'), symbol('NVDA'), ...]
    
    # Set commission and slippage
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    
    # Initialize strategy parameters
    context.lookback = 20
    context.threshold = 0.02
    
def handle_data(context, data):
    try:
        # Get historical data using data.history()
        prices = data.history(context.assets, 'price', context.lookback, '1d')
        
        # Implement strategy logic here
        # ...
        
        # Place orders (check cash availability)
        # order(asset, shares) or order_target_percent(asset, percent)
        
    except Exception as e:
        print(f"Error in handle_data: {e}")
```

Generate complete, production-ready code."""
    
    def _build_user_prompt(
        self,
        hypothesis: Dict[str, Any],
        market_context: Dict[str, Any],
        parent_code: Optional[str],
        cousin_codes: Optional[List[str]]
    ) -> str:
        """Build the user prompt with hypothesis and context."""
        prompt_parts = []
        
        # Hypothesis section
        prompt_parts.append("=== TRADING STRATEGY HYPOTHESIS ===")
        prompt_parts.append(f"Hypothesis: {hypothesis.get('hypothesis', 'N/A')}")
        prompt_parts.append(f"Rationale: {hypothesis.get('rationale', 'N/A')}")
        prompt_parts.append(f"Objectives: {hypothesis.get('objectives', 'N/A')}")
        prompt_parts.append(f"Expected Insights: {hypothesis.get('expected_insights', 'N/A')}")
        
        # Market context
        prompt_parts.append("\n=== MARKET CONTEXT ===")
        assets = market_context.get('assets', [])
        prompt_parts.append(f"Assets: {', '.join(assets)}")
        prompt_parts.append(f"Benchmark: {market_context.get('benchmark', 'N/A')}")
        prompt_parts.append(f"Data Frequency: {market_context.get('frequency', 'daily')}")
        
        # Previous error (if any)
        if 'previous_error' in hypothesis:
            prompt_parts.append("\n=== PREVIOUS ERROR (FIX THIS) ===")
            prompt_parts.append(f"Error: {hypothesis['previous_error'].get('error', 'Unknown')}")
        
        # Parent code reference
        if parent_code:
            prompt_parts.append("\n=== PARENT STRATEGY CODE (for reference) ===")
            prompt_parts.append(parent_code[:2000] + "..." if len(parent_code) > 2000 else parent_code)
        
        # Cousin codes reference
        if cousin_codes:
            prompt_parts.append("\n=== COUSIN STRATEGY CODES (for inspiration) ===")
            for i, cousin_code in enumerate(cousin_codes[:3]):  # Limit to 3 cousins
                prompt_parts.append(f"\n--- Cousin {i+1} ---")
                prompt_parts.append(cousin_code[:1000] + "..." if len(cousin_code) > 1000 else cousin_code)
        
        prompt_parts.append("\n=== INSTRUCTION ===")
        prompt_parts.append("Generate complete, executable Zipline strategy code based on the hypothesis above.")
        prompt_parts.append("Ensure the code follows the required structure and handles all edge cases.")
        
        return "\n".join(prompt_parts)
    
    def _extract_code(self, raw_response: str) -> str:
        """Extract Python code from LLM response."""
        import re
        
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```python\s*(.*?)\s*```'
        match = re.search(code_block_pattern, raw_response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        code_block_pattern = r'```\s*(.*?)\s*```'
        match = re.search(code_block_pattern, raw_response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code blocks, return the raw response (might be plain code)
        return raw_response.strip()
    
    def _execute_backtest(
        self,
        temp_file: str,
        assets: List[str],
        start_date: str,
        end_date: str,
        bundle: str,
        capital_base: float,
        commission_per_share: float,
        min_trade_cost: float
    ) -> Dict[str, Any]:
        """
        Execute Zipline backtest from temporary file.
        
        Args:
            temp_file: Path to temporary Python file with strategy code
            assets: List of asset symbols
            start_date: Start date string
            end_date: End date string
            capital_base: Starting capital
            commission_per_share: Commission per share
            min_trade_cost: Minimum trade cost
            
        Returns:
            Dict[str, Any]: Backtest metrics dictionary
        """
        # Import zipline and run the algorithm
        from zipline import run_algorithm
        from zipline.finance import commission, slippage
        import pandas as pd
        import numpy as np
        
        # Load the strategy module
        import importlib.util
        spec = importlib.util.spec_from_file_location("strategy", temp_file)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Verify required functions exist
        if not hasattr(strategy_module, 'initialize'):
            raise ValueError("Strategy code missing initialize() function")
        if not hasattr(strategy_module, 'handle_data'):
            raise ValueError("Strategy code missing handle_data() function")
        
        # Run the backtest
        results = run_algorithm(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            initialize=strategy_module.initialize,
            handle_data=strategy_module.handle_data,
            capital_base=capital_base,
            data_frequency='daily',
            bundle=bundle
        )
        
        # Compute metrics from results
        metrics = self._compute_backtest_metrics(results)
        metrics["returns_series"] = (
            results["returns"].dropna().astype(float).tolist()
            if "returns" in results.columns
            else []
        )
        metrics["benchmark_returns"] = (
            results["benchmark_return"].dropna().astype(float).tolist()
            if "benchmark_return" in results.columns
            else []
        )
        
        return metrics
    
    def _compute_backtest_metrics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute performance metrics from backtest results.
        
        Args:
            results: DataFrame with backtest results (returns, portfolio_value, etc.)
            
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        import numpy as np
        
        if 'returns' not in results.columns or len(results) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'num_trades': 0
            }
        
        returns = results['returns'].dropna()
        
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'num_trades': 0
            }
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized Sharpe Ratio (assuming daily data, 252 trading days)
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / negative_returns.std()
        else:
            sortino_ratio = sharpe_ratio  # Fallback if no negative returns
        
        # Maximum Drawdown
        if 'portfolio_value' in results.columns:
            portfolio_values = results['portfolio_value'].dropna()
            running_max = portfolio_values.cummax()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            # Estimate from returns
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Number of trades (if available)
        num_trades = 0
        if 'transactions' in results.columns:
            # Count total transactions
            num_trades = sum(len(txns) for txns in results['transactions'] if txns)
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'num_trades': int(num_trades),
            'num_periods': len(returns)
        }
