"""
Equal Weight Baseline Strategy

This module implements a simple equal-weight portfolio strategy that serves as a baseline
for comparing evolved strategies. The portfolio is rebalanced daily to maintain equal
weights (1/N) across all assets in the universe.

Strategy Logic:
- Universe: AAPL, NVDA, AMZN, GOOGL, MSFT, TSLA (6 assets)
- Weight: Equal weight (16.67% each for 6 assets)
- Rebalancing: Daily
- Commission: $0.0075 per share (minimum $1.00)
- Slippage: VolumeShareSlippage (quadratic model)

This strategy represents a naive diversification approach and is commonly used as a
benchmark in portfolio optimization literature. Despite its simplicity, equal-weight
portfolios often outperform market-cap weighted indices due to the "small firm effect"
and automatic rebalancing from winners to losers.

Author: QuantEvolve Team
"""

from zipline.api import symbol, order_target_percent, get_open_orders
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage
import pandas as pd


def initialize(context):
    """
    Initialize the equal weight strategy.
    
    Sets up the asset universe, configures commission and slippage models,
    and initializes tracking variables for rebalancing.
    
    Args:
        context: Zipline context object containing strategy state
    """
    # Define asset universe (6 large-cap tech stocks)
    context.assets = [
        symbol('AAPL'),
        symbol('NVDA'),
        symbol('AMZN'),
        symbol('GOOGL'),
        symbol('MSFT'),
        symbol('TSLA')
    ]
    
    # Configure commission: $0.0075 per share, minimum $1.00 per trade
    context.set_commission(PerShare(cost=0.0075, min_trade_cost=1.0))
    
    # Configure slippage: VolumeShareSlippage with quadratic price impact
    context.set_slippage(VolumeShareSlippage())
    
    # Set benchmark (SPY for comparison)
    context.set_benchmark(symbol('SPY'))
    
    # Calculate equal weight per asset
    context.num_assets = len(context.assets)
    context.target_weight = 1.0 / context.num_assets
    
    # Initialize tracking variables
    context.rebalance_counter = 0
    context.rebalance_frequency = 1  # Daily rebalancing
    
    # Log initialization
    log.info(f"Equal Weight Strategy Initialized")
    log.info(f"Number of assets: {context.num_assets}")
    log.info(f"Target weight per asset: {context.target_weight:.4f} ({context.target_weight*100:.2f}%)")
    log.info(f"Rebalancing frequency: Every {context.rebalance_frequency} day(s)")


def handle_data(context, data):
    """
    Execute daily trading logic.
    
    Checks if rebalancing is due and executes trades to maintain equal weights.
    Uses only historical data via data.history() to avoid lookahead bias.
    
    Args:
        context: Zipline context object containing strategy state
        data: Zipline data object providing access to market data
    """
    # Increment rebalancing counter
    context.rebalance_counter += 1
    
    # Check if rebalancing is due (daily = every 1 bar)
    if context.rebalance_counter >= context.rebalance_frequency:
        # Reset counter
        context.rebalance_counter = 0
        
        # Execute rebalancing
        rebalance_portfolio(context, data)


def rebalance_portfolio(context, data):
    """
    Rebalance portfolio to equal weights.
    
    Orders each asset to its target weight (1/N). Uses order_target_percent
    which automatically calculates the required shares based on current portfolio
    value and asset price.
    
    Args:
        context: Zipline context object containing strategy state
        data: Zipline data object providing access to market data
    """
    # Track number of orders placed
    orders_placed = 0
    
    # Iterate through all assets in universe
    for asset in context.assets:
        # Check if we have data for this asset (skip if delisted or suspended)
        if not data.can_trade(asset):
            log.warning(f"Asset {asset.symbol} is not tradable, skipping")
            continue
        
        # Get current price for logging
        try:
            current_price = data.current(asset, 'price')
        except Exception:
            log.warning(f"Cannot get price for {asset.symbol}, skipping")
            continue
        
        # Get current position
        current_position = context.portfolio.positions[asset].amount if asset in context.portfolio.positions else 0
        
        # Calculate target value
        target_percent = context.target_weight
        current_value = context.portfolio.portfolio_value
        
        # Place order to achieve target weight
        # order_target_percent automatically handles:
        # - Calculating target shares based on current price
        # - Checking available cash
        # - Respecting position limits
        order_target_percent(asset, target_percent)
        orders_placed += 1
        
        # Log trade (only if position changed significantly)
        if current_position != 0 or target_percent > 0.01:
            log.debug(f"Rebalanced {asset.symbol}: price=${current_price:.2f}, target={target_percent*100:.2f}%")
    
    # Log rebalancing summary
    log.debug(f"Rebalanced portfolio: {orders_placed} orders placed")
    log.debug(f"Portfolio value: ${context.portfolio.portfolio_value:.2f}")
    log.debug(f"Cash: ${context.portfolio.cash:.2f}")


def analyze_portfolio(context, data):
    """
    Analyze current portfolio state (optional diagnostic function).
    
    Calculates current weights, turnover, and other portfolio metrics.
    Useful for debugging and performance analysis.
    
    Args:
        context: Zipline context object containing strategy state
        data: Zipline data object providing access to market data
    
    Returns:
        dict: Portfolio analysis including weights, turnover, and exposure
    """
    portfolio_analysis = {
        'positions': {},
        'total_weight': 0.0,
        'cash_weight': 0.0,
        'num_positions': 0
    }
    
    portfolio_value = context.portfolio.portfolio_value
    
    # Analyze each position
    for asset, position in context.portfolio.positions.items():
        if position.amount != 0:
            try:
                current_price = data.current(asset, 'price')
                position_value = position.amount * current_price
                position_weight = position_value / portfolio_value
                
                portfolio_analysis['positions'][asset.symbol] = {
                    'amount': position.amount,
                    'price': current_price,
                    'value': position_value,
                    'weight': position_weight
                }
                
                portfolio_analysis['total_weight'] += position_weight
                portfolio_analysis['num_positions'] += 1
                
            except Exception:
                log.warning(f"Cannot analyze position for {asset.symbol}")
    
    # Calculate cash weight
    portfolio_analysis['cash_weight'] = context.portfolio.cash / portfolio_value
    
    return portfolio_analysis
