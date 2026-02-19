"""
Market Capitalization Weighted Strategy

This baseline strategy weights assets by their market capitalization,
rebalancing monthly to maintain cap-weighted exposure.

Larger companies receive higher weights, mimicking traditional index funds.
"""

from zipline.api import symbol, order_target_percent, get_open_orders
import pandas as pd


def initialize(context):
    """
    Initialize the market cap weighted strategy.
    
    Sets up the asset universe and scheduling for monthly rebalancing.
    """
    # Define the universe of assets
    context.assets = [symbol('AAPL'), symbol('NVDA'), symbol('AMZN'), 
                      symbol('GOOGL'), symbol('MSFT'), symbol('TSLA')]
    
    # Set benchmark
    context.set_benchmark(symbol('SPY'))
    
    # Set commission and slippage
    from zipline.finance.commission import PerShare
    from zipline.finance.slippage import VolumeShareSlippage
    
    context.set_commission(PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(VolumeShareSlippage())
    
    # Store market cap data (in a real implementation, this would be fetched dynamically)
    # Using approximate market caps as of 2024 (in billions)
    context.market_caps = {
        'AAPL': 3000,
        'NVDA': 1800,
        'AMZN': 1700,
        'GOOGL': 1900,
        'MSFT': 3100,
        'TSLA': 600
    }
    
    # Schedule monthly rebalancing (first trading day of each month)
    context.rebalance_day = 1
    context.month_counter = 0


def handle_data(context, data):
    """
    Handle daily data and execute monthly rebalancing.
    
    Rebalances portfolio to market-cap weights on the first trading day of each month.
    """
    # Track trading days to approximate monthly rebalancing
    context.month_counter += 1
    
    # Rebalance approximately every 21 trading days (one month)
    if context.month_counter % 21 == 0:
        rebalance_portfolio(context, data)


def rebalance_portfolio(context, data):
    """
    Rebalance portfolio to market-cap weighted allocation.
    
    Calculates weights based on market capitalization and orders target positions.
    """
    # Calculate total market cap
    total_cap = sum(context.market_caps.values())
    
    # Calculate weights for each asset
    weights = {}
    for asset in context.assets:
        symbol_name = asset.symbol
        if symbol_name in context.market_caps:
            weights[asset] = context.market_caps[symbol_name] / total_cap
        else:
            # Equal weight for assets without market cap data
            weights[asset] = 1.0 / len(context.assets)
    
    # Order target weights
    for asset, target_weight in weights.items():
        if data.can_trade(asset):
            order_target_percent(asset, target_weight)
    
    # Log rebalancing event
    log.info(f"Rebalanced portfolio to market-cap weights: {[f'{a.symbol}: {weights[a]:.2%}' for a in context.assets]}")
