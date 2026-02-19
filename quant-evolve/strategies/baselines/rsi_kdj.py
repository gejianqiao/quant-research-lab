"""
RSI-KDJ Baseline Strategy

Implements a combined RSI (Relative Strength Index) and KDJ (Stochastic Oscillator) 
trading strategy as a baseline for comparison with evolved strategies.

Parameters (from paper):
- RSI Period: 14
- RSI Oversold: 30
- RSI Overbought: 70
- KDJ Period: 9 (standard)
- KDJ K-period: 3
- KDJ D-period: 3

This strategy generates buy signals when RSI < 30 (oversold) AND KDJ shows bullish 
crossover, and sell signals when RSI > 70 (overbought) AND KDJ shows bearish crossover.
"""

import numpy as np
import pandas as pd
from zipline.api import symbol, order_target_percent, get_open_orders
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage


def initialize(context):
    """
    Initialize the RSI-KDJ strategy.
    
    Sets up:
    - Asset universe (6 large-cap tech stocks)
    - Commission: $0.0075 per share, minimum $1.00
    - Slippage: VolumeShareSlippage
    - RSI parameters (period=14, oversold=30, overbought=70)
    - KDJ parameters (period=9, k_period=3, d_period=3)
    - State tracking dictionaries for each asset
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
    
    # RSI parameters
    context.rsi_period = 14
    context.rsi_oversold = 30
    context.rsi_overbought = 70
    
    # KDJ parameters
    context.kdj_period = 9
    context.kdj_k_period = 3
    context.kdj_d_period = 3
    
    # Lookback window for indicator calculation
    context.lookback = max(context.rsi_period, context.kdj_period) + 10
    
    # State tracking for each asset
    context.prev_k = {}
    context.prev_d = {}
    context.in_position = {}
    
    for asset in context.assets:
        context.prev_k[asset] = None
        context.prev_d[asset] = None
        context.in_position[asset] = False


def handle_data(context, data):
    """
    Main trading logic executed at each bar.
    
    For each asset:
    1. Retrieve historical price data using data.history()
    2. Calculate RSI and KDJ indicators
    3. Generate buy/sell signals based on combined conditions
    4. Execute trades using order_target_percent
    
    Buy Signal: RSI < 30 AND K crosses above D (bullish crossover)
    Sell Signal: RSI > 70 AND K crosses below D (bearish crossover)
    """
    for asset in context.assets:
        try:
            # Skip if not tradable
            if not data.can_trade(asset):
                continue
            
            # Get historical price data (no lookahead bias)
            prices = data.history(asset, 'price', context.lookback, '1d')
            
            # Skip if insufficient data
            if len(prices) < context.lookback:
                continue
            
            # Calculate RSI
            rsi = calculate_rsi(prices, context.rsi_period)
            if rsi is None or len(rsi) < 1:
                continue
            
            current_rsi = rsi.iloc[-1]
            
            # Calculate KDJ
            k, d, j = calculate_kdj(
                prices, 
                context.kdj_period, 
                context.kdj_k_period, 
                context.kdj_d_period
            )
            
            if k is None or len(k) < 2 or d is None or len(d) < 2:
                continue
            
            current_k = k.iloc[-1]
            current_d = d.iloc[-1]
            prev_k = k.iloc[-2] if len(k) >= 2 else context.prev_k.get(asset, 50)
            prev_d = d.iloc[-2] if len(d) >= 2 else context.prev_d.get(asset, 50)
            
            # Update previous values for next iteration
            context.prev_k[asset] = current_k
            context.prev_d[asset] = current_d
            
            # Generate signals
            buy_signal = (
                current_rsi < context.rsi_oversold and  # RSI oversold
                prev_k <= prev_d and  # Previous: K <= D
                current_k > current_d  # Current: K > D (bullish crossover)
            )
            
            sell_signal = (
                current_rsi > context.rsi_overbought and  # RSI overbought
                prev_k >= prev_d and  # Previous: K >= D
                current_k < current_d  # Current: K < D (bearish crossover)
            )
            
            # Execute trades
            if buy_signal and not context.in_position[asset]:
                # Buy signal: enter long position
                order_target_percent(asset, 0.16)  # ~16% per asset (6 assets)
                context.in_position[asset] = True
                
            elif sell_signal and context.in_position[asset]:
                # Sell signal: exit position
                order_target_percent(asset, 0)
                context.in_position[asset] = False
                
        except Exception as e:
            # Log error but continue with other assets
            pass


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Formula:
    - RS = Average Gain / Average Loss (over period)
    - RSI = 100 - (100 / (1 + RS))
    
    Parameters:
    - prices: pandas Series of closing prices
    - period: RSI calculation period (default 14)
    
    Returns:
    - pandas Series of RSI values
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses using Wilder's smoothing method
    # First average: simple moving average
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Subsequent averages: Wilder's smoothing
    # avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
    for i in range(period, len(prices)):
        if i == period:
            continue
        avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
        avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_kdj(prices, period=9, k_period=3, d_period=3):
    """
    Calculate KDJ (Stochastic Oscillator) indicators.
    
    Formula:
    - RSV = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - K = SMA(RSV, k_period)
    - D = SMA(K, d_period)
    - J = 3*K - 2*D
    
    Parameters:
    - prices: pandas Series of closing prices (or DataFrame with OHLC)
    - period: Stochastic period (default 9)
    - k_period: K line smoothing period (default 3)
    - d_period: D line smoothing period (default 3)
    
    Returns:
    - tuple: (k_line, d_line, j_line) as pandas Series
    """
    if len(prices) < period:
        return None, None, None
    
    # If prices is a Series, we need OHLC data
    # For simplicity, assume prices represents close and use same for high/low
    # In real implementation, this would use data.history with multiple fields
    # Here we'll use a simplified version assuming prices is close
    
    # For proper KDJ, we need high, low, close
    # Since we only have close in this simplified version, we'll approximate
    # In production, use: data.history(asset, ['high', 'low', 'close'], ...)
    
    # Simplified KDJ using price range
    rolling_high = prices.rolling(window=period).max()
    rolling_low = prices.rolling(window=period).min()
    
    # Calculate RSV (Raw Stochastic Value)
    rsv = ((prices - rolling_low) / (rolling_high - rolling_low).replace(0, np.inf)) * 100
    rsv = rsv.fillna(50)  # Default to 50 for initial values
    
    # Calculate K line (smoothed RSV)
    k = rsv.rolling(window=k_period).mean()
    k = k.fillna(50)
    
    # Calculate D line (smoothed K)
    d = k.rolling(window=d_period).mean()
    d = d.fillna(50)
    
    # Calculate J line
    j = 3 * k - 2 * d
    
    return k, d, j


def analyze_portfolio(context, data):
    """
    Diagnostic function to analyze current portfolio state.
    
    Returns:
    - dict: Current positions, weights, RSI values, and KDJ values
    """
    portfolio_info = {
        'positions': {},
        'rsi_values': {},
        'kdj_values': {},
        'cash': context.portfolio.cash,
        'total_value': context.portfolio.portfolio_value
    }
    
    for asset in context.assets:
        try:
            # Get position info
            if asset in context.portfolio.positions:
                position = context.portfolio.positions[asset]
                portfolio_info['positions'][asset.symbol] = {
                    'amount': position.amount,
                    'cost_basis': position.cost_basis,
                    'current_price': data.current(asset, 'price')
                }
            
            # Calculate current RSI
            prices = data.history(asset, 'price', context.lookback, '1d')
            if len(prices) >= context.lookback:
                rsi = calculate_rsi(prices, context.rsi_period)
                if rsi is not None:
                    portfolio_info['rsi_values'][asset.symbol] = rsi.iloc[-1]
                
                k, d, j = calculate_kdj(prices, context.kdj_period, 
                                        context.kdj_k_period, context.kdj_d_period)
                if k is not None:
                    portfolio_info['kdj_values'][asset.symbol] = {
                        'k': k.iloc[-1],
                        'd': d.iloc[-1],
                        'j': j.iloc[-1]
                    }
        except Exception:
            continue
    
    return portfolio_info
