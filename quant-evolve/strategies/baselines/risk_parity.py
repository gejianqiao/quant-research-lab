"""
Risk Parity Baseline Strategy

Implements a risk parity approach where assets are weighted inversely to their volatility,
ensuring each asset contributes equally to portfolio risk. This serves as a sophisticated
baseline for comparing evolved strategies.

Reference: Table 4 in QuantEvolve paper (Risk Parity baseline)
"""

import numpy as np
import pandas as pd
from zipline.api import symbol, order_target_percent, get_open_orders
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage


def initialize(context):
    """
    Initialize the risk parity strategy.
    
    Sets up:
    - Asset universe: 6 large-cap tech stocks
    - Commission: $0.0075/share (min $1.00)
    - Slippage: VolumeShareSlippage
    - Volatility lookback: 60 days
    - Rebalancing: Monthly (21 trading days)
    """
    # Asset universe
    context.assets = [
        symbol('AAPL'),
        symbol('NVDA'),
        symbol('AMZN'),
        symbol('GOOGL'),
        symbol('MSFT'),
        symbol('TSLA')
    ]
    
    # Commission and slippage (matching paper specifications)
    context.set_commission(PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(VolumeShareSlippage())
    
    # Risk parity parameters
    context.volatility_lookback = 60  # 60-day rolling volatility
    context.rebalance_frequency = 21  # Monthly rebalancing
    context.trading_days = 0
    
    # State tracking
    context.volatilities = {}
    context.weights = {}
    
    # Benchmark
    context.set_benchmark(symbol('SPY'))


def handle_data(context, data):
    """
    Main trading loop executed daily.
    
    Tracks trading days and triggers rebalancing monthly.
    """
    context.trading_days += 1
    
    # Rebalance monthly
    if context.trading_days % context.rebalance_frequency == 0:
        rebalance_portfolio(context, data)


def rebalance_portfolio(context, data):
    """
    Calculate risk parity weights and rebalance portfolio.
    
    Risk Parity Formula:
    - Calculate 60-day rolling volatility for each asset
    - Weight = (1/volatility) / sum(1/volatility for all assets)
    - This ensures each asset contributes equally to portfolio risk
    """
    weights = {}
    inverse_vols = {}
    
    # Calculate volatility for each asset
    for asset in context.assets:
        try:
            # Get historical prices
            prices = data.history(asset, 'price', context.volatility_lookback, '1d')
            
            if len(prices) < context.volatility_lookback:
                # Not enough data, skip this asset
                continue
            
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            if volatility > 0 and np.isfinite(volatility):
                context.volatilities[asset.symbol] = volatility
                inverse_vols[asset] = 1.0 / volatility
            else:
                # Invalid volatility, exclude from portfolio
                continue
                
        except Exception as e:
            # Log error and skip asset
            continue
    
    # Calculate risk parity weights
    if len(inverse_vols) > 0:
        total_inverse_vol = sum(inverse_vols.values())
        
        for asset, inv_vol in inverse_vols.items():
            # Normalize weights to sum to 1.0
            weights[asset] = inv_vol / total_inverse_vol
    else:
        # Fallback to equal weight if no valid volatilities
        n_assets = len(context.assets)
        weights = {asset: 1.0 / n_assets for asset in context.assets}
    
    # Execute trades
    for asset in context.assets:
        try:
            # Check if asset is tradable
            if not data.can_trade(asset):
                continue
            
            # Get target weight (default to 0 if not calculated)
            target_weight = weights.get(asset, 0.0)
            
            # Execute order
            order_target_percent(asset, target_weight)
            
        except Exception as e:
            # Log error and continue
            continue
    
    # Store weights for analysis
    context.weights = {asset.symbol: weight for asset, weight in weights.items()}


def analyze_portfolio(context, data):
    """
    Diagnostic function to analyze current portfolio state.
    
    Returns:
        dict: Current position weights, values, and risk metrics
    """
    portfolio_info = {
        'positions': {},
        'weights': {},
        'volatilities': context.volatilities.copy(),
        'target_weights': context.weights.copy(),
        'cash': context.portfolio.cash,
        'total_value': context.portfolio.portfolio_value
    }
    
    for asset in context.assets:
        if asset in context.portfolio.positions:
            position = context.portfolio.positions[asset]
            portfolio_info['positions'][asset.symbol] = {
                'amount': position.amount,
                'value': position.last_sale_price * position.amount if position.last_sale_price else 0
            }
    
    return portfolio_info
