"""
MACD Baseline Strategy

Implements a Moving Average Convergence Divergence (MACD) strategy as a baseline
for comparison with evolved strategies. Uses standard parameters (12, 26, 9) and
generates buy/sell signals based on MACD line crossovers with the signal line.

Strategy Logic:
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Buy when MACD crosses above Signal (bullish momentum)
- Sell when MACD crosses below Signal (bearish momentum)
"""

from zipline.api import symbol, order_target_percent, get_open_orders
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage
import pandas as pd
import numpy as np


def initialize(context):
    """
    Initialize the MACD strategy.
    
    Sets up:
    - Asset universe (6 large-cap tech stocks)
    - Commission: $0.0075 per share (min $1.00)
    - Slippage: VolumeShareSlippage model
    - MACD parameters: fast=12, slow=26, signal=9
    - Lookback period for indicator calculation
    """
    # Define asset universe
    context.assets = [
        symbol('AAPL'),
        symbol('NVDA'),
        symbol('AMZN'),
        symbol('GOOGL'),
        symbol('MSFT'),
        symbol('TSLA')
    ]
    
    # Set commission and slippage (matching QuantEvolve specs)
    context.set_commission(PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(VolumeShareSlippage())
    
    # MACD parameters (standard settings)
    context.fast_period = 12
    context.slow_period = 26
    context.signal_period = 9
    
    # Lookback period for indicator calculation (slow + buffer)
    context.lookback = context.slow_period + context.signal_period + 10
    
    # Initialize tracking dictionaries
    context.macd_values = {asset: [] for asset in context.assets}
    context.signal_values = {asset: [] for asset in context.assets}
    context.prev_macd = {asset: None for asset in context.assets}
    context.prev_signal = {asset: None for asset in context.assets}
    
    # Trading state tracking
    context.in_position = {asset: False for asset in context.assets}
    
    # Set benchmark
    context.set_benchmark(symbol('SPY'))


def handle_data(context, data):
    """
    Main trading loop executed at each bar.
    
    For each asset:
    1. Fetch historical price data using data.history() (no lookahead bias)
    2. Calculate MACD and Signal line
    3. Detect crossovers
    4. Generate buy/sell signals
    5. Execute orders
    """
    # Check if we have enough data
    if len(data.history(context.assets[0], 'price', context.lookback, '1d')) < context.lookback:
        return
    
    for asset in context.assets:
        try:
            # Get historical closing prices (no lookahead bias)
            prices = data.history(asset, 'close', context.lookback, '1d')
            
            if len(prices) < context.lookback or prices.isnull().any():
                continue
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=context.fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=context.slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate Signal line (EMA of MACD)
            signal_line = macd_line.ewm(span=context.signal_period, adjust=False).mean()
            
            # Get current and previous values
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            # Skip if we don't have previous values
            if context.prev_macd[asset] is None or context.prev_signal[asset] is None:
                context.prev_macd[asset] = current_macd
                context.prev_signal[asset] = current_signal
                context.macd_values[asset].append(current_macd)
                context.signal_values[asset].append(current_signal)
                continue
            
            prev_macd = context.prev_macd[asset]
            prev_signal = context.prev_signal[asset]
            
            # Detect crossovers
            # Bullish crossover: MACD crosses above Signal
            bullish_crossover = (prev_macd <= prev_signal) and (current_macd > current_signal)
            
            # Bearish crossover: MACD crosses below Signal
            bearish_crossover = (prev_macd >= prev_signal) and (current_macd < current_signal)
            
            # Execute trades based on signals
            if bullish_crossover and not context.in_position[asset]:
                # Buy signal - enter long position
                order_target_percent(asset, 1.0 / len(context.assets))
                context.in_position[asset] = True
                
            elif bearish_crossover and context.in_position[asset]:
                # Sell signal - exit position
                order_target_percent(asset, 0.0)
                context.in_position[asset] = False
            
            # Update tracking values
            context.prev_macd[asset] = current_macd
            context.prev_signal[asset] = current_signal
            context.macd_values[asset].append(current_macd)
            context.signal_values[asset].append(current_signal)
            
        except Exception as e:
            # Log error but continue with other assets
            continue


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Helper function to calculate MACD indicator values.
    
    Args:
        prices: pandas Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
