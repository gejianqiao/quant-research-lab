"""
Data loading utilities for QuantEvolve.

This module provides functions for loading OHLCV (Open, High, Low, Close, Volume)
market data from various sources including CSV files, Parquet files, and Zipline bundles.
It handles data validation, preprocessing, and formatting for use in backtesting.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_ohlcv_data(
    symbol: str,
    data_path: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    file_format: str = 'auto',
    date_column: str = 'date',
    ohlcv_columns: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load OHLCV data for a single symbol from file.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'ES')
        data_path: Path to data file or directory
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        file_format: File format ('csv', 'parquet', 'auto')
        date_column: Name of the date column
        ohlcv_columns: Mapping of standard names to actual column names
                      {'open': 'Open', 'high': 'High', ...}
    
    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    data_path = Path(data_path)
    
    # Determine file path
    if data_path.is_dir():
        # Try common file naming conventions
        possible_files = [
            data_path / f"{symbol}.csv",
            data_path / f"{symbol}.parquet",
            data_path / f"{symbol.lower()}.csv",
            data_path / f"{symbol.lower()}.parquet",
        ]
        file_path = None
        for pf in possible_files:
            if pf.exists():
                file_path = pf
                break
        if file_path is None:
            raise FileNotFoundError(f"Could not find data file for {symbol} in {data_path}")
    else:
        file_path = data_path
    
    # Auto-detect format if needed
    if file_format == 'auto':
        if file_path.suffix == '.csv':
            file_format = 'csv'
        elif file_path.suffix == '.parquet':
            file_format = 'parquet'
        else:
            raise ValueError(f"Cannot auto-detect format for {file_path.suffix}")
    
    # Load data
    logger.debug(f"Loading {symbol} data from {file_path}")
    try:
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        raise
    
    # Standardize column names
    df = _standardize_columns(df, date_column, ohlcv_columns)
    
    # Parse dates and set index
    df = _set_datetime_index(df, date_column)
    
    # Filter by date range
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    # Validate data
    _validate_ohlcv_data(df, symbol)
    
    logger.info(f"Loaded {len(df)} rows for {symbol} ({df.index.min()} to {df.index.max()})")
    
    return df


def load_data_bundle(
    bundle_name: str,
    data_dir: Union[str, Path],
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols from a data bundle directory.
    
    Args:
        bundle_name: Name of the data bundle (e.g., 'equities', 'futures')
        data_dir: Base directory containing data bundles
        symbols: List of asset symbols to load
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames
    
    Raises:
        FileNotFoundError: If bundle directory doesn't exist
    """
    bundle_path = Path(data_dir) / bundle_name
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Data bundle '{bundle_name}' not found at {bundle_path}")
    
    data_dict = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            df = load_ohlcv_data(
                symbol=symbol,
                data_path=bundle_path,
                start_date=start_date,
                end_date=end_date
            )
            data_dict[symbol] = df
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        logger.warning(f"Failed to load {len(failed_symbols)} symbols: {failed_symbols}")
    
    logger.info(f"Successfully loaded {len(data_dict)}/{len(symbols)} symbols from {bundle_name}")
    
    return data_dict


def validate_data(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    min_rows: int = 100,
    required_columns: Optional[List[str]] = None,
    check_gaps: bool = True,
    max_gap_days: int = 5
) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for completeness and correctness.
    
    Args:
        data: Single DataFrame or dict of DataFrames to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names (default: OHLCV)
        check_gaps: Whether to check for missing dates
        max_gap_days: Maximum allowed gap in trading days
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    issues = []
    
    # Handle dict of DataFrames
    if isinstance(data, dict):
        for symbol, df in data.items():
            symbol_issues = _validate_single_dataframe(
                df, symbol, min_rows, required_columns, check_gaps, max_gap_days
            )
            issues.extend(symbol_issues)
    else:
        issues = _validate_single_dataframe(
            data, 'data', min_rows, required_columns, check_gaps, max_gap_days
        )
    
    is_valid = len(issues) == 0
    return is_valid, issues


def _validate_single_dataframe(
    df: pd.DataFrame,
    name: str,
    min_rows: int,
    required_columns: List[str],
    check_gaps: bool,
    max_gap_days: int
) -> List[str]:
    """Validate a single DataFrame."""
    issues = []
    
    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"{name}: Only {len(df)} rows (minimum: {min_rows})")
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"{name}: Missing columns: {missing_cols}")
    
    # Check for NaN values
    nan_counts = df[required_columns].isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            issues.append(f"{name}: Column '{col}' has {count} NaN values")
    
    # Check for negative prices or volume
    if 'open' in df.columns and (df['open'] <= 0).any():
        issues.append(f"{name}: Negative or zero open prices detected")
    if 'volume' in df.columns and (df['volume'] < 0).any():
        issues.append(f"{name}: Negative volume detected")
    
    # Check OHLC consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_hl = (df['high'] < df['low']).any()
        invalid_oc = ((df['open'] > df['high']) | (df['open'] < df['low'])).any()
        if invalid_hl:
            issues.append(f"{name}: High < Low detected")
        if invalid_oc:
            issues.append(f"{name}: Open outside High-Low range detected")
    
    # Check for date gaps
    if check_gaps and len(df) > 1:
        date_diffs = df.index.to_series().diff().dt.days
        large_gaps = date_diffs[date_diffs > max_gap_days]
        if len(large_gaps) > 0:
            issues.append(f"{name}: {len(large_gaps)} gaps > {max_gap_days} days detected")
    
    return issues


def _standardize_columns(
    df: pd.DataFrame,
    date_column: str,
    ohlcv_columns: Optional[Dict[str, str]]
) -> pd.DataFrame:
    """Standardize column names to lowercase OHLCV format."""
    # Default mapping
    standard_mapping = {
        'open': ['open', 'Open', 'OPEN', 'o'],
        'high': ['high', 'High', 'HIGH', 'h'],
        'low': ['low', 'Low', 'LOW', 'l'],
        'close': ['close', 'Close', 'CLOSE', 'c', 'adj_close', 'Adj Close'],
        'volume': ['volume', 'Volume', 'VOLUME', 'v', 'vol']
    }
    
    # Use custom mapping if provided
    if ohlcv_columns:
        for standard_name, actual_name in ohlcv_columns.items():
            if actual_name in df.columns:
                df = df.rename(columns={actual_name: standard_name})
    else:
        # Auto-detect and rename
        rename_dict = {}
        for standard_name, variants in standard_mapping.items():
            for variant in variants:
                if variant in df.columns and standard_name not in df.columns:
                    rename_dict[variant] = standard_name
                    break
        df = df.rename(columns=rename_dict)
    
    # Ensure date column is named 'date' temporarily
    if date_column != 'date' and date_column in df.columns:
        df = df.rename(columns={date_column: 'date'})
    
    return df


def _set_datetime_index(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Convert date column to DatetimeIndex."""
    if 'date' not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Set index and sort
    df = df.set_index('date').sort_index()
    
    # Remove duplicate indices
    if df.index.has_duplicates:
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate dates, keeping first")
        df = df[~df.index.duplicated(keep='first')]
    
    return df


def _validate_ohlcv_data(df: pd.DataFrame, symbol: str) -> None:
    """Raise ValueError if data validation fails."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"{symbol}: Missing required columns: {missing}")
    
    if len(df) == 0:
        raise ValueError(f"{symbol}: Empty dataset after filtering")


def create_zipline_bundle(
    data: Dict[str, pd.DataFrame],
    bundle_name: str,
    output_dir: Union[str, Path],
    start_date: str,
    end_date: str
) -> bool:
    """
    Create a Zipline-compatible data bundle from OHLCV DataFrames.
    
    Args:
        data: Dictionary of symbol -> DataFrame
        bundle_name: Name for the bundle
        output_dir: Output directory for bundle files
        start_date: Bundle start date
        end_date: Bundle end date
    
    Returns:
        bool: True if successful
    
    Note:
        This function creates a simplified bundle structure. For production use,
        consider using zipline's official bundle ingestion API.
    """
    output_path = Path(output_dir) / bundle_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        for symbol, df in data.items():
            # Save individual symbol data
            output_file = output_path / f"{symbol}.csv"
            df.to_csv(output_file)
            logger.debug(f"Saved {symbol} to {output_file}")
        
        # Save bundle metadata
        metadata = {
            'bundle_name': bundle_name,
            'symbols': list(data.keys()),
            'start_date': start_date,
            'end_date': end_date,
            'created_at': datetime.now().isoformat(),
            'num_symbols': len(data)
        }
        
        import json
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created bundle '{bundle_name}' with {len(data)} symbols at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create bundle: {e}")
        return False


def resample_data(
    df: pd.DataFrame,
    frequency: str = 'D',
    ohlcv_aggregation: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        frequency: Target frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
        ohlcv_aggregation: Custom aggregation rules (optional)
    
    Returns:
        pd.DataFrame: Resampled OHLCV data
    """
    if ohlcv_aggregation is None:
        ohlcv_aggregation = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Filter to only OHLCV columns that exist
    agg_dict = {k: v for k, v in ohlcv_aggregation.items() if k in df.columns}
    
    if not agg_dict:
        raise ValueError("No OHLCV columns found for resampling")
    
    resampled = df.resample(frequency).agg(agg_dict)
    
    # Drop rows with all NaN
    resampled = resampled.dropna(how='all')
    
    logger.debug(f"Resampled data from {len(df)} to {len(resampled)} rows ({frequency})")
    
    return resampled


def align_data(
    data: Dict[str, pd.DataFrame],
    method: str = 'intersection'
) -> Dict[str, pd.DataFrame]:
    """
    Align multiple DataFrames to have the same date index.
    
    Args:
        data: Dictionary of symbol -> DataFrame
        method: Alignment method ('intersection', 'union', 'forward_fill')
    
    Returns:
        Dict[str, pd.DataFrame]: Aligned DataFrames
    """
    if not data:
        return data
    
    # Get all indices
    indices = [df.index for df in data.values()]
    
    if method == 'intersection':
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
    elif method == 'union':
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.union(idx)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Reindex all DataFrames
    aligned_data = {}
    for symbol, df in data.items():
        aligned_df = df.reindex(common_index)
        
        if method == 'forward_fill':
            aligned_df = aligned_df.fillna(method='ffill')
        
        aligned_data[symbol] = aligned_df
    
    logger.info(f"Aligned {len(data)} DataFrames to {len(common_index)} dates ({method})")
    
    return aligned_data


def compute_returns(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    price_column: str = 'close',
    method: str = 'log'
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute returns from price data.
    
    Args:
        data: Single DataFrame or dict of DataFrames
        price_column: Column to use for return calculation
        method: Return method ('simple' or 'log')
    
    Returns:
        DataFrame or dict of DataFrames with returns
    """
    def _compute_single(df: pd.DataFrame) -> pd.DataFrame:
        prices = df[price_column]
        
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:  # simple
            returns = prices.pct_change()
        
        return returns.to_frame(name='returns')
    
    if isinstance(data, dict):
        return {symbol: _compute_single(df) for symbol, df in data.items()}
    else:
        return _compute_single(data)
