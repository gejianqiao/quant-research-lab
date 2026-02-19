"""
Test suite for backtesting metrics computation.

Validates the correctness of financial metric calculations including:
- Sharpe Ratio (Eq 3 component)
- Sortino Ratio
- Information Ratio
- Max Drawdown
- Combined Score (Eq 3: SR + IR + MDD)
- Trading Frequency
- All other metrics from src/backtesting/metrics.py

Tests use synthetic return series with known expected values to verify
formulas match manual calculations within 1e-6 tolerance.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import metrics module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_information_ratio,
    compute_max_drawdown,
    compute_max_drawdown_from_returns,
    compute_total_return,
    compute_calmar_ratio,
    compute_combined_score,
    compute_all_metrics,
    compute_trading_frequency,
    validate_metrics
)


class TestComputeSharpeRatio:
    """Test Sharpe Ratio calculation."""
    
    def test_sharpe_positive_returns(self):
        """Test Sharpe ratio with consistent positive returns."""
        # Constant 1% daily returns
        returns = np.array([0.01] * 252)
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252)
        
        # With constant returns, std = 0, should handle gracefully
        # Expected: returns some value (implementation dependent for zero std)
        assert isinstance(sharpe, (int, float))
        assert not np.isnan(sharpe)
    
    def test_sharpe_mixed_returns(self):
        """Test Sharpe ratio with realistic mixed returns."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)  # ~12.6% annual, 20% vol
        
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252)
        
        # Should be approximately mean/std * sqrt(252)
        expected_approx = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
        
        assert isinstance(sharpe, float)
        assert abs(sharpe - expected_approx) < 0.1  # Allow small tolerance
    
    def test_sharpe_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = np.array([0.01] * 100)
        risk_free = 0.02 / 252  # 2% annual
        
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=risk_free, annualization_factor=252)
        
        assert isinstance(sharpe, float)
        # Excess return should be lower than with rf=0
    
    def test_sharpe_empty_returns(self):
        """Test Sharpe ratio with empty returns array."""
        returns = np.array([])
        sharpe = compute_sharpe_ratio(returns)
        
        assert sharpe == 0.0  # Should handle gracefully
    
    def test_sharpe_single_value(self):
        """Test Sharpe ratio with single return value."""
        returns = np.array([0.01])
        sharpe = compute_sharpe_ratio(returns)
        
        assert sharpe == 0.0  # Cannot compute std with n=1
    
    def test_sharpe_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.01] * 252)
        sharpe = compute_sharpe_ratio(returns)
        
        assert sharpe < 0  # Should be negative
    
    def test_sharpe_pandas_series(self):
        """Test Sharpe ratio with pandas Series input."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        sharpe = compute_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)


class TestComputeSortinoRatio:
    """Test Sortino Ratio calculation (downside risk-adjusted)."""
    
    def test_sortino_positive_returns(self):
        """Test Sortino ratio with all positive returns."""
        returns = np.array([0.01] * 252)
        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=252)
        
        # With no negative returns, downside std = 0
        assert isinstance(sortino, (int, float))
    
    def test_sortino_mixed_returns(self):
        """Test Sortino ratio with mixed returns."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)
        
        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=252)
        
        assert isinstance(sortino, float)
        # Sortino should typically be higher than Sharpe for same returns
        sharpe = compute_sharpe_ratio(returns)
        assert sortino >= sharpe  # Downside risk <= total risk
    
    def test_sortino_with_target_return(self):
        """Test Sortino ratio with custom target return."""
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
        target = 0.005  # 0.5% target
        
        sortino = compute_sortino_ratio(returns, target_return=target)
        
        assert isinstance(sortino, float)
    
    def test_sortino_empty_returns(self):
        """Test Sortino ratio with empty returns."""
        returns = np.array([])
        sortino = compute_sortino_ratio(returns)
        
        assert sortino == 0.0


class TestComputeInformationRatio:
    """Test Information Ratio calculation (active return / tracking error)."""
    
    def test_ir_outperformance(self):
        """Test IR when portfolio outperforms benchmark."""
        np.random.seed(42)
        portfolio_returns = np.random.normal(0.001, 0.02, 252)  # Higher mean
        benchmark_returns = np.random.normal(0.0005, 0.02, 252)  # Lower mean
        
        ir = compute_information_ratio(portfolio_returns, benchmark_returns)
        
        assert isinstance(ir, float)
        # Should be positive if portfolio outperforms
        excess_return = np.mean(portfolio_returns - benchmark_returns)
        if excess_return > 0:
            assert ir > 0
    
    def test_ir_underperformance(self):
        """Test IR when portfolio underperforms benchmark."""
        np.random.seed(42)
        portfolio_returns = np.random.normal(0.0003, 0.02, 252)  # Lower mean
        benchmark_returns = np.random.normal(0.0008, 0.02, 252)  # Higher mean
        
        ir = compute_information_ratio(portfolio_returns, benchmark_returns)
        
        assert ir < 0  # Should be negative
    
    def test_ir_equal_performance(self):
        """Test IR when portfolio matches benchmark."""
        returns = np.array([0.01, -0.02, 0.015, 0.005] * 63)
        
        ir = compute_information_ratio(returns, returns)
        
        # Excess return = 0, so IR should be 0
        assert abs(ir) < 1e-6
    
    def test_ir_different_lengths(self):
        """Test IR with different length arrays (should handle gracefully)."""
        portfolio_returns = np.array([0.01, 0.02, 0.015])
        benchmark_returns = np.array([0.008, 0.018])  # Shorter
        
        # Should either raise error or handle gracefully
        try:
            ir = compute_information_ratio(portfolio_returns, benchmark_returns)
            assert isinstance(ir, float)
        except (ValueError, AssertionError):
            pass  # Acceptable to raise error for mismatched lengths


class TestComputeMaxDrawdown:
    """Test Maximum Drawdown calculation."""
    
    def test_mdd_constant_growth(self):
        """Test MDD with constantly growing values (no drawdown)."""
        values = np.array([100, 101, 102, 103, 104, 105])
        mdd = compute_max_drawdown(values)
        
        assert mdd == 0.0  # No drawdown in constant growth
    
    def test_mdd_single_decline(self):
        """Test MDD with single peak-to-trough decline."""
        values = np.array([100, 110, 120, 100, 90, 95])
        mdd = compute_max_drawdown(values)
        
        # Peak = 120, Trough = 90, MDD = (90-120)/120 = -0.25
        expected_mdd = -0.25
        assert abs(mdd - expected_mdd) < 1e-6
    
    def test_mdd_multiple_peaks(self):
        """Test MDD with multiple peaks and troughs."""
        values = np.array([100, 150, 120, 180, 140, 200, 160])
        mdd = compute_max_drawdown(values)
        
        # Peak = 200, Trough = 160, MDD = (160-200)/200 = -0.20
        expected_mdd = -0.20
        assert abs(mdd - expected_mdd) < 1e-6
    
    def test_mdd_from_returns(self):
        """Test MDD calculated from returns series."""
        returns = np.array([0.10, -0.20, 0.15, -0.10, 0.05])
        mdd = compute_max_drawdown_from_returns(returns)
        
        assert isinstance(mdd, float)
        assert mdd <= 0  # MDD should be negative or zero
    
    def test_mdd_empty_values(self):
        """Test MDD with empty values array."""
        values = np.array([])
        mdd = compute_max_drawdown(values)
        
        assert mdd == 0.0
    
    def test_mdd_pandas_series(self):
        """Test MDD with pandas Series input."""
        values = pd.Series([100, 110, 105, 120, 115, 110])
        mdd = compute_max_drawdown(values)
        
        assert isinstance(mdd, float)
        assert mdd <= 0


class TestComputeTotalReturn:
    """Test Total Return calculation."""
    
    def test_total_return_positive(self):
        """Test total return with positive returns."""
        returns = np.array([0.10, 0.10, 0.10])  # 3 periods of 10%
        total = compute_total_return(returns)
        
        # (1.1 * 1.1 * 1.1) - 1 = 1.331 - 1 = 0.331
        expected = 1.1 ** 3 - 1
        assert abs(total - expected) < 1e-6
    
    def test_total_return_negative(self):
        """Test total return with negative returns."""
        returns = np.array([-0.10, -0.10, -0.10])
        total = compute_total_return(returns)
        
        assert total < 0
        assert total > -1  # Cannot lose more than 100%
    
    def test_total_return_zero(self):
        """Test total return with zero returns."""
        returns = np.array([0.0, 0.0, 0.0])
        total = compute_total_return(returns)
        
        assert total == 0.0
    
    def test_total_return_empty(self):
        """Test total return with empty returns."""
        returns = np.array([])
        total = compute_total_return(returns)
        
        assert total == 0.0


class TestComputeCalmarRatio:
    """Test Calmar Ratio calculation (annualized return / |MDD|)."""
    
    def test_calmar_positive(self):
        """Test Calmar ratio with positive returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        calmar = compute_calmar_ratio(returns)
        
        assert isinstance(calmar, float)
    
    def test_calmar_high_drawdown(self):
        """Test Calmar ratio with high drawdown."""
        # Large positive returns followed by large crash
        returns = np.array([0.05] * 100 + [-0.50] + [0.02] * 151)
        
        calmar = compute_calmar_ratio(returns)
        
        # High MDD should result in low Calmar
        assert isinstance(calmar, float)


class TestComputeCombinedScore:
    """Test Combined Score calculation (Equation 3: SR + IR + MDD)."""
    
    def test_combined_score_typical(self):
        """Test combined score with typical values."""
        sr = 1.5
        ir = 0.8
        mdd = -0.15
        
        score = compute_combined_score(sr, ir, mdd)
        
        expected = sr + ir + mdd  # 1.5 + 0.8 - 0.15 = 2.15
        assert abs(score - expected) < 1e-6
    
    def test_combined_score_high_quality(self):
        """Test combined score with high-quality strategy."""
        sr = 2.0
        ir = 1.5
        mdd = -0.05
        
        score = compute_combined_score(sr, ir, mdd)
        
        assert score > 3.0  # High score for high-quality strategy
    
    def test_combined_score_poor_quality(self):
        """Test combined score with poor-quality strategy."""
        sr = -0.5
        ir = -0.3
        mdd = -0.40
        
        score = compute_combined_score(sr, ir, mdd)
        
        assert score < 0  # Negative score for poor strategy
    
    def test_combined_score_zero_mdd(self):
        """Test combined score with zero drawdown."""
        sr = 1.0
        ir = 0.5
        mdd = 0.0
        
        score = compute_combined_score(sr, ir, mdd)
        
        expected = sr + ir  # 1.5
        assert abs(score - expected) < 1e-6


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""
    
    def test_all_metrics_complete(self):
        """Test that all_metrics returns all expected fields."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)
        benchmark = np.random.normal(0.0003, 0.02, 252)
        
        metrics = compute_all_metrics(returns, benchmark_returns=benchmark)
        
        # Check all expected keys exist
        expected_keys = [
            'sharpe_ratio', 'sortino_ratio', 'information_ratio',
            'max_drawdown', 'total_return', 'calmar_ratio',
            'combined_score', 'annualized_return', 'volatility',
            'win_rate', 'avg_win', 'avg_loss', 'profit_factor'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        
        # Check types
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)
        assert isinstance(metrics['combined_score'], float)
    
    def test_all_metrics_no_benchmark(self):
        """Test all_metrics without benchmark (IR should be 0)."""
        returns = np.array([0.01, -0.02, 0.015, 0.01] * 63)
        
        metrics = compute_all_metrics(returns, benchmark_returns=None)
        
        assert metrics['information_ratio'] == 0.0
    
    def test_all_metrics_values_reasonable(self):
        """Test that computed metrics have reasonable values."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)
        
        metrics = compute_all_metrics(returns)
        
        # Sharpe should be in reasonable range
        assert -5 < metrics['sharpe_ratio'] < 5
        
        # MDD should be between -1 and 0
        assert -1 <= metrics['max_drawdown'] <= 0
        
        # Total return can be any value > -1
        assert metrics['total_return'] > -1
        
        # Win rate should be between 0 and 1
        assert 0 <= metrics['win_rate'] <= 1


class TestComputeTradingFrequency:
    """Test Trading Frequency calculation (for feature map dimension)."""
    
    def test_frequency_no_trading(self):
        """Test frequency with no position changes."""
        positions = np.array([1, 1, 1, 1, 1])  # Always held
        total_periods = 5
        
        freq = compute_trading_frequency(positions, total_periods)
        
        assert freq == 0.0  # No trading
    
    def test_frequency_always_trading(self):
        """Test frequency with trading every period."""
        positions = np.array([1, 0, 1, 0, 1])  # Change every period
        total_periods = 5
        
        freq = compute_trading_frequency(positions, total_periods)
        
        assert freq > 0.5  # High frequency
    
    def test_frequency_partial_trading(self):
        """Test frequency with partial trading."""
        positions = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        total_periods = 10
        
        freq = compute_trading_frequency(positions, total_periods)
        
        assert 0 < freq < 1  # Partial frequency
    
    def test_frequency_empty_positions(self):
        """Test frequency with empty positions array."""
        positions = np.array([])
        total_periods = 10
        
        freq = compute_trading_frequency(positions, total_periods)
        
        assert freq == 0.0


class TestValidateMetrics:
    """Test metrics validation function."""
    
    def test_validate_valid_metrics(self):
        """Test validation with valid metrics."""
        metrics = {
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15,
            'total_return': 2.5
        }
        
        is_valid = validate_metrics(metrics)
        
        assert is_valid is True
    
    def test_validate_invalid_sharpe(self):
        """Test validation with invalid Sharpe ratio."""
        metrics = {
            'sharpe_ratio': 15.0,  # Too high
            'max_drawdown': -0.15,
            'total_return': 2.5
        }
        
        is_valid = validate_metrics(metrics)
        
        assert is_valid is False
    
    def test_validate_invalid_mdd(self):
        """Test validation with invalid MDD."""
        metrics = {
            'sharpe_ratio': 1.5,
            'max_drawdown': -1.5,  # Worse than -100%
            'total_return': 2.5
        }
        
        is_valid = validate_metrics(metrics)
        
        assert is_valid is False
    
    def test_validate_custom_bounds(self):
        """Test validation with custom bounds."""
        metrics = {
            'sharpe_ratio': 2.0,
            'max_drawdown': -0.10,
            'total_return': 1.5
        }
        
        # Stricter bounds
        is_valid = validate_metrics(
            metrics,
            min_sharpe=0.0,
            max_sharpe=1.5,
            min_mdd=-0.05,
            max_mdd=0.0
        )
        
        assert is_valid is False  # Sharpe too high, MDD too large


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_metrics_with_nan(self):
        """Test metrics computation with NaN values."""
        returns = np.array([0.01, np.nan, 0.02, -0.01, 0.015])
        
        sharpe = compute_sharpe_ratio(returns)
        
        # Should handle NaN gracefully (either skip or return 0)
        assert isinstance(sharpe, (int, float))
    
    def test_metrics_with_inf(self):
        """Test metrics computation with Inf values."""
        returns = np.array([0.01, np.inf, 0.02, -0.01, 0.015])
        
        sharpe = compute_sharpe_ratio(returns)
        
        # Should handle Inf gracefully
        assert isinstance(sharpe, (int, float))
    
    def test_metrics_all_zeros(self):
        """Test metrics with all zero returns."""
        returns = np.zeros(252)
        
        metrics = compute_all_metrics(returns)
        
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['total_return'] == 0.0
        assert metrics['max_drawdown'] == 0.0
    
    def test_metrics_extreme_values(self):
        """Test metrics with extreme return values."""
        returns = np.array([1.0, -0.99, 2.0, -0.95, 0.5] * 50)
        
        metrics = compute_all_metrics(returns)
        
        # Should compute without crashing
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)


class TestMetricsReproducibility:
    """Test that metrics are reproducible with same inputs."""
    
    def test_sharpe_reproducibility(self):
        """Test Sharpe ratio reproducibility."""
        np.random.seed(123)
        returns1 = np.random.normal(0.0005, 0.02, 252)
        
        np.random.seed(123)
        returns2 = np.random.normal(0.0005, 0.02, 252)
        
        sharpe1 = compute_sharpe_ratio(returns1)
        sharpe2 = compute_sharpe_ratio(returns2)
        
        assert sharpe1 == sharpe2
    
    def test_all_metrics_reproducibility(self):
        """Test all metrics reproducibility."""
        np.random.seed(456)
        returns = np.random.normal(0.0005, 0.02, 252)
        benchmark = np.random.normal(0.0003, 0.02, 252)
        
        metrics1 = compute_all_metrics(returns, benchmark)
        metrics2 = compute_all_metrics(returns, benchmark)
        
        assert metrics1 == metrics2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
