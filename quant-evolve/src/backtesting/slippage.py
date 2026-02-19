"""
Slippage Models for QuantEvolve Backtesting Engine.

This module implements various slippage models used in the Zipline backtesting framework,
including the VolumeShareSlippage model which uses a quadratic price impact function.

References:
    - QuantEvolve Paper Section 4.3: Backtesting Framework
    - Zipline Reloaded Documentation: Slippage Models
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order."""
    sid: int
    amount: int  # Positive for buy, negative for sell
    limit: Optional[float] = None
    stop: Optional[float] = None


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def process_order(self, data: Any, order: Order) -> Tuple[float, int]:
        """
        Process an order and return the execution price and filled amount.
        
        Parameters
        ----------
        data : Any
            Current market data (includes current price, volume, etc.)
        order : Order
            The order to process
            
        Returns
        -------
        Tuple[float, int]
            (execution_price, filled_amount)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for this slippage model."""
        pass


class VolumeShareSlippage(SlippageModel):
    """
    Volume Share Slippage Model with quadratic price impact.
    
    This model simulates the price impact of trading based on the volume share
    of the order relative to the total trading volume. The price impact follows
    a quadratic function, meaning larger orders have disproportionately higher
    slippage costs.
    
    Price Impact Formula:
        price_impact = price_impact_factor * (volume_share ** 2) * sign
        
    Where:
        - volume_share = |order_amount| / total_volume
        - sign = +1 for buy orders, -1 for sell orders
        - price_impact_factor controls the severity of slippage
    
    For buy orders: execution_price = current_price * (1 + price_impact)
    For sell orders: execution_price = current_price * (1 - price_impact)
    
    Parameters
    ----------
    volume_limit : float, optional
        Maximum fraction of total volume that can be traded (default: 0.025 = 2.5%)
    price_impact : float, optional
        Coefficient for the quadratic price impact function (default: 0.1)
    
    Attributes
    ----------
    volume_limit : float
        Maximum volume share allowed per order
    price_impact_factor : float
        Quadratic price impact coefficient
    
    Examples
    --------
    >>> slippage = VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)
    >>> order = Order(sid=1, amount=1000)
    >>> data = {'current_price': 100.0, 'volume': 100000}
    >>> exec_price, filled = slippage.process_order(data, order)
    """
    
    def __init__(self, volume_limit: float = 0.025, price_impact: float = 0.1):
        """
        Initialize the VolumeShareSlippage model.
        
        Parameters
        ----------
        volume_limit : float, optional
            Maximum fraction of total volume that can be traded (default: 0.025)
        price_impact : float, optional
            Coefficient for the quadratic price impact function (default: 0.1)
        """
        self.volume_limit = volume_limit
        self.price_impact_factor = price_impact
        logger.debug(f"VolumeShareSlippage initialized: volume_limit={volume_limit}, "
                    f"price_impact={price_impact}")
    
    def process_order(self, data: Any, order: Order) -> Tuple[float, int]:
        """
        Process an order with volume-based slippage.
        
        Parameters
        ----------
        data : Any
            Market data dictionary with 'current_price' and 'volume' keys,
            or an object with current_price and volume attributes
        order : Order
            The order to process
            
        Returns
        -------
        Tuple[float, int]
            (execution_price, filled_amount)
            
        Notes
        -----
        - If order size exceeds volume_limit * total_volume, the order is capped
        - Price impact is quadratic in volume share
        - Buy orders pay higher prices, sell orders receive lower prices
        """
        # Extract market data
        if isinstance(data, dict):
            current_price = data.get('current_price', data.get('price', 100.0))
            total_volume = data.get('volume', 1e6)  # Default high volume if not specified
        else:
            # Assume object with attributes
            current_price = getattr(data, 'current_price', getattr(data, 'price', 100.0))
            total_volume = getattr(data, 'volume', 1e6)
        
        # Handle edge cases
        if total_volume <= 0:
            logger.warning(f"Zero or negative volume detected: {total_volume}. "
                          f"Using default volume.")
            total_volume = 1e6
        
        if current_price <= 0:
            logger.warning(f"Zero or negative price detected: {current_price}. "
                          f"Cannot process order.")
            return 0.0, 0
        
        # Determine order direction and magnitude
        order_direction = np.sign(order.amount)  # +1 for buy, -1 for sell
        order_magnitude = abs(order.amount)
        
        # Calculate maximum allowed volume based on volume limit
        max_allowed_volume = int(total_volume * self.volume_limit)
        
        # Cap the order size if it exceeds the volume limit
        if order_magnitude > max_allowed_volume:
            logger.debug(f"Order size {order_magnitude} exceeds volume limit "
                        f"{max_allowed_volume}. Capping order.")
            filled_amount = order_direction * max_allowed_volume
        else:
            filled_amount = order.amount
        
        # Calculate volume share (always positive)
        volume_share = abs(filled_amount) / total_volume
        
        # Calculate price impact using quadratic model
        # price_impact = factor * (volume_share)^2
        price_impact = self.price_impact_factor * (volume_share ** 2)
        
        # Adjust price based on order direction
        # Buy orders: price goes up (pay more)
        # Sell orders: price goes down (receive less)
        if order_direction > 0:  # Buy order
            execution_price = current_price * (1 + price_impact)
        else:  # Sell order
            execution_price = current_price * (1 - price_impact)
        
        # Apply limit price constraint if specified
        if order.limit is not None:
            if order_direction > 0 and execution_price > order.limit:
                # Buy limit order: don't execute if price is too high
                logger.debug(f"Buy limit order not executed: "
                            f"exec_price={execution_price:.4f} > limit={order.limit:.4f}")
                return 0.0, 0
            elif order_direction < 0 and execution_price < order.limit:
                # Sell limit order: don't execute if price is too low
                logger.debug(f"Sell limit order not executed: "
                            f"exec_price={execution_price:.4f} < limit={order.limit:.4f}")
                return 0.0, 0
        
        # Apply stop price constraint if specified
        if order.stop is not None:
            if order_direction > 0 and current_price < order.stop:
                # Buy stop order: only trigger if price rises above stop
                logger.debug(f"Buy stop order not triggered: "
                            f"current_price={current_price:.4f} < stop={order.stop:.4f}")
                return 0.0, 0
            elif order_direction < 0 and current_price > order.stop:
                # Sell stop order: only trigger if price falls below stop
                logger.debug(f"Sell stop order not triggered: "
                            f"current_price={current_price:.4f} < stop={order.stop:.4f}")
                return 0.0, 0
        
        logger.debug(f"Order processed: amount={filled_amount}, "
                    f"current_price={current_price:.4f}, "
                    f"exec_price={execution_price:.4f}, "
                    f"slippage={abs(execution_price - current_price):.4f} "
                    f"({price_impact*100:.4f}%)")
        
        return execution_price, int(filled_amount)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for this slippage model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model type and parameters
        """
        return {
            'model_type': 'volume_share_slippage',
            'volume_limit': self.volume_limit,
            'price_impact': self.price_impact_factor
        }


class FixedSlippage(SlippageModel):
    """
    Fixed Slippage Model with constant spread.
    
    This model applies a fixed spread to all trades, regardless of order size
    or market volume. It's a simpler model useful for baseline comparisons.
    
    For buy orders: execution_price = current_price + spread/2
    For sell orders: execution_price = current_price - spread/2
    
    Parameters
    ----------
    spread : float, optional
        Fixed spread in price units (default: 0.01 = 1 cent)
    
    Examples
    --------
    >>> slippage = FixedSlippage(spread=0.05)
    >>> order = Order(sid=1, amount=1000)
    >>> data = {'current_price': 100.0}
    >>> exec_price, filled = slippage.process_order(data, order)
    """
    
    def __init__(self, spread: float = 0.01):
        """
        Initialize the FixedSlippage model.
        
        Parameters
        ----------
        spread : float, optional
            Fixed spread in price units (default: 0.01)
        """
        self.spread = spread
        logger.debug(f"FixedSlippage initialized: spread={spread}")
    
    def process_order(self, data: Any, order: Order) -> Tuple[float, int]:
        """
        Process an order with fixed slippage.
        
        Parameters
        ----------
        data : Any
            Market data with 'current_price' key or attribute
        order : Order
            The order to process
            
        Returns
        -------
        Tuple[float, int]
            (execution_price, filled_amount)
        """
        # Extract current price
        if isinstance(data, dict):
            current_price = data.get('current_price', data.get('price', 100.0))
        else:
            current_price = getattr(data, 'current_price', getattr(data, 'price', 100.0))
        
        if current_price <= 0:
            logger.warning(f"Invalid price: {current_price}")
            return 0.0, 0
        
        # Calculate execution price based on order direction
        order_direction = np.sign(order.amount)
        
        if order_direction > 0:  # Buy order
            execution_price = current_price + self.spread / 2
        else:  # Sell order
            execution_price = current_price - self.spread / 2
        
        # Ensure price doesn't go negative
        execution_price = max(0.01, execution_price)
        
        logger.debug(f"Fixed slippage order: amount={order.amount}, "
                    f"current_price={current_price:.4f}, "
                    f"exec_price={execution_price:.4f}")
        
        return execution_price, order.amount
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'model_type': 'fixed_slippage',
            'spread': self.spread
        }


class PriceImpactSlippage(SlippageModel):
    """
    Linear Price Impact Slippage Model.
    
    This model simulates slippage as a linear function of order size relative
    to a reference volume. It's less severe than the quadratic model but more
    realistic than fixed slippage for moderate order sizes.
    
    Price Impact Formula:
        price_impact = impact_factor * (order_size / reference_volume)
    
    Parameters
    ----------
    impact_factor : float, optional
        Linear price impact coefficient (default: 0.01)
    reference_volume : float, optional
        Reference volume for normalizing order size (default: 1e6)
    """
    
    def __init__(self, impact_factor: float = 0.01, reference_volume: float = 1e6):
        """
        Initialize the PriceImpactSlippage model.
        
        Parameters
        ----------
        impact_factor : float, optional
            Linear price impact coefficient (default: 0.01)
        reference_volume : float, optional
            Reference volume for normalizing (default: 1e6)
        """
        self.impact_factor = impact_factor
        self.reference_volume = reference_volume
        logger.debug(f"PriceImpactSlippage initialized: impact_factor={impact_factor}, "
                    f"reference_volume={reference_volume}")
    
    def process_order(self, data: Any, order: Order) -> Tuple[float, int]:
        """
        Process an order with linear price impact.
        
        Parameters
        ----------
        data : Any
            Market data with 'current_price' key or attribute
        order : Order
            The order to process
            
        Returns
        -------
        Tuple[float, int]
            (execution_price, filled_amount)
        """
        # Extract current price
        if isinstance(data, dict):
            current_price = data.get('current_price', data.get('price', 100.0))
        else:
            current_price = getattr(data, 'current_price', getattr(data, 'price', 100.0))
        
        if current_price <= 0:
            logger.warning(f"Invalid price: {current_price}")
            return 0.0, 0
        
        # Calculate price impact
        order_magnitude = abs(order.amount)
        volume_ratio = order_magnitude / self.reference_volume
        price_impact = self.impact_factor * volume_ratio
        
        # Adjust price based on order direction
        order_direction = np.sign(order.amount)
        
        if order_direction > 0:  # Buy order
            execution_price = current_price * (1 + price_impact)
        else:  # Sell order
            execution_price = current_price * (1 - price_impact)
        
        execution_price = max(0.01, execution_price)
        
        logger.debug(f"Price impact slippage: amount={order.amount}, "
                    f"volume_ratio={volume_ratio:.6f}, "
                    f"price_impact={price_impact:.6f}, "
                    f"exec_price={execution_price:.4f}")
        
        return execution_price, order.amount
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'model_type': 'price_impact_slippage',
            'impact_factor': self.impact_factor,
            'reference_volume': self.reference_volume
        }


def create_slippage_model(model_type: str, **kwargs) -> SlippageModel:
    """
    Factory function to create slippage models by name.
    
    Parameters
    ----------
    model_type : str
        Type of slippage model: 'volume_share', 'fixed', or 'price_impact'
    **kwargs
        Additional arguments passed to the model constructor
        
    Returns
    -------
    SlippageModel
        Instantiated slippage model
        
    Raises
    ------
    ValueError
        If model_type is not recognized
        
    Examples
    --------
    >>> slippage = create_slippage_model('volume_share', volume_limit=0.05)
    >>> slippage = create_slippage_model('fixed', spread=0.02)
    """
    model_map = {
        'volume_share': VolumeShareSlippage,
        'fixed': FixedSlippage,
        'price_impact': PriceImpactSlippage
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown slippage model type: {model_type}. "
                        f"Available types: {list(model_map.keys())}")
    
    model_class = model_map[model_type]
    return model_class(**kwargs)


def get_default_slippage_config() -> Dict[str, Any]:
    """
    Return the default slippage configuration for QuantEvolve.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary matching the paper's specifications
    """
    return {
        'model_type': 'volume_share',
        'volume_limit': 0.025,  # 2.5% of daily volume
        'price_impact': 0.1     # Quadratic impact factor
    }


# Backward-compatible alias for legacy imports.
VolumeShareSlippageModel = VolumeShareSlippage
