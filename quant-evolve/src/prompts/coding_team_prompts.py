"""
Prompt templates for the Coding Team agent.

This module defines all prompt templates used by the Coding Team to:
1. Translate research hypotheses into executable Zipline trading strategies
2. Debug and fix code that fails backtesting
3. Ensure compliance with Zipline structure and no-lookahead-bias constraints
"""

# System prompt for the Coding Team
CODING_TEAM_SYSTEM_PROMPT = """You are an expert quantitative developer specializing in Zipline backtesting framework.
Your role is to translate trading strategy hypotheses into clean, efficient, and executable Python code.

CONSTRAINTS:
1. MUST use Zipline Reloaded API (zipline.api)
2. MUST include initialize() and handle_data() functions
3. MUST set commission: context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
4. MUST set slippage: context.set_slippage(slippage.VolumeShareSlippage())
5. MUST use data.history() for all price data access (NO lookahead bias)
6. MUST handle exceptions gracefully in handle_data()
7. MUST respect position limits and cash constraints
8. MUST use only daily frequency data

OUTPUT FORMAT:
- Return ONLY valid Python code
- No markdown code blocks (no ```)
- No explanations outside of code comments
- Include docstrings for both functions

COMMON PITFALLS TO AVOID:
- Do NOT use future data (e.g., data.current() for prices not yet available)
- Do NOT assume unlimited capital
- Do NOT forget to import required Zipline modules
- Do NOT use global variables for state (use context object)
"""

# Main code generation prompt
CODE_GENERATION_PROMPT = """
Translate the following trading strategy hypothesis into executable Zipline code.

HYPOTHESIS:
{hypothesis}

RATIONALE:
{rationale}

OBJECTIVES:
{objectives}

PARENT STRATEGY CODE (for reference, do NOT copy directly):
{parent_code}

COUSIN STRATEGIES (for diversity inspiration):
{cousin_codes}

MARKET CONTEXT:
{market_context}

REQUIREMENTS:
1. Implement initialize(context) function with:
   - Asset selection using symbols()
   - Commission setup: PerShare(cost=0.0075, min_trade_cost=1.0)
   - Slippage setup: VolumeShareSlippage()
   - Benchmark setting if applicable
   - Any custom context variables needed

2. Implement handle_data(context, data) function with:
   - Use data.history(context.asset, 'price', N, '1d') for price data
   - Calculate indicators based on hypothesis
   - Generate trading signals
   - Execute orders with order() or order_target_percent()
   - Handle exceptions with try-except blocks
   - Log key decisions with context.log.info()

3. Strategy-specific requirements:
   - Lookback period: {lookback_period} days
   - Rebalancing frequency: {rebalance_frequency}
   - Position sizing: {position_sizing}
   - Risk constraints: {risk_constraints}

4. Code quality:
   - Add comments explaining key logic
   - Use meaningful variable names
   - Keep functions modular and readable
   - Include error handling for data gaps

Generate the complete Python code for this strategy:
"""

# Code debugging prompt (used when backtest fails)
CODE_DEBUGGING_PROMPT = """
The following Zipline strategy code failed during backtesting.

ORIGINAL CODE:
{code}

ERROR MESSAGE:
{error_message}

TRACEBACK:
{traceback}

HYPOTHESIS (for context):
{hypothesis}

TASK:
1. Analyze the error and identify the root cause
2. Fix the code while preserving the original strategy logic
3. Ensure the fix complies with Zipline requirements:
   - Proper initialize() and handle_data() structure
   - Correct use of data.history() (no lookahead bias)
   - Proper commission and slippage setup
   - Valid order sizing (within cash limits)
   - Exception handling in handle_data()

4. Common issues to check:
   - Syntax errors (missing colons, parentheses, etc.)
   - Import errors (missing zipline.api imports)
   - Data access errors (using data.current() instead of data.history())
   - Order sizing errors (ordering more than available cash)
   - Division by zero (e.g., when calculating indicators with insufficient data)
   - Attribute errors (accessing non-existent context variables)

Return ONLY the corrected Python code:
"""

# Code optimization prompt (optional, for improving performance)
CODE_OPTIMIZATION_PROMPT = """
The following strategy code is functional but may be optimized for better performance.

CURRENT CODE:
{code}

BACKTEST METRICS:
{metrics}

OPTIMIZATION GOALS:
1. Reduce unnecessary computations in handle_data()
2. Cache indicator calculations where possible
3. Optimize data.history() calls (avoid redundant lookbacks)
4. Improve order execution logic (reduce turnover if not needed)
5. Add early-exit conditions for edge cases

CONSTRAINTS:
- Do NOT change the core strategy logic
- Maintain Zipline compatibility
- Preserve commission and slippage settings
- Keep code readable and well-commented

Return the optimized Python code:
"""

# Zipline structure template (for reference in prompts)
ZIPLINE_STRUCTURE_TEMPLATE = """
from zipline.api import (
    symbol, symbols,
    order, order_target_percent, order_target_value,
    set_commission, set_slippage, set_benchmark,
    commission, slippage
)

def initialize(context):
    '''
    Initialize trading strategy context.
    Called once at the start of the backtest.
    '''
    # Asset selection
    context.assets = symbols({assets})
    context.asset = context.assets[0]  # Primary asset
    
    # Commission and slippage (REQUIRED)
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    
    # Benchmark
    context.set_benchmark(symbol('{benchmark}'))
    
    # Custom context variables
    context.lookback = {lookback}
    context.trading_day = 0
    
    # Initialize indicators/storage
    context.prices = []
    context.signals = []

def handle_data(context, data):
    '''
    Main trading logic executed at each time step.
    '''
    try:
        context.trading_day += 1
        
        # Get historical price data (NO lookahead bias)
        prices = data.history(context.asset, 'price', context.lookback, '1d')
        
        # Check for sufficient data
        if len(prices) < context.lookback:
            return
        
        # Calculate indicators
        # ... (strategy-specific logic)
        
        # Generate trading signal
        # ... (strategy-specific logic)
        
        # Execute orders
        # ... (strategy-specific logic)
        
        # Log decisions
        # context.log.info(f'Day {context.trading_day}: Signal={signal}')
        
    except Exception as e:
        context.log.info(f'Error in handle_data: {str(e)}')
"""

# Code validation checklist prompt
CODE_VALIDATION_PROMPT = """
Before submitting strategy code, validate it against this checklist:

CODE TO VALIDATE:
{code}

VALIDATION CHECKLIST:
[ ] 1. Contains initialize(context) function
[ ] 2. Contains handle_data(context, data) function
[ ] 3. Imports zipline.api modules correctly
[ ] 4. Sets commission: commission.PerShare(cost=0.0075, min_trade_cost=1.0)
[ ] 5. Sets slippage: slippage.VolumeShareSlippage()
[ ] 6. Uses data.history() for ALL price data access
[ ] 7. Does NOT use data.current() for prices (lookahead bias)
[ ] 8. Handles exceptions in handle_data() with try-except
[ ] 9. Respects position limits (no over-leveraging)
[ ] 10. Uses valid Zipline order functions (order, order_target_percent, etc.)
[ ] 11. No syntax errors (proper indentation, colons, parentheses)
[ ] 12. No undefined variables or functions

If any item is unchecked, fix the code before returning.

Return the validated code:
"""

# Strategy variation prompt (for generating diverse implementations)
STRATEGY_VARIATION_PROMPT = """
Generate a variation of the following strategy that maintains the core hypothesis
but implements it differently to increase population diversity.

ORIGINAL HYPOTHESIS:
{hypothesis}

ORIGINAL CODE:
{code}

VARIATION REQUIREMENTS:
1. Keep the same core trading logic (e.g., momentum, mean-reversion)
2. Change implementation details:
   - Use different indicator parameters (e.g., 20-day MA instead of 50-day)
   - Add additional filters (e.g., volume confirmation, volatility filter)
   - Modify entry/exit conditions (e.g., threshold levels)
   - Change position sizing method (e.g., volatility-weighted)
   - Add risk management rules (e.g., stop-loss, take-profit)

3. Ensure the variation is meaningfully different (not just cosmetic changes)
4. Maintain Zipline compatibility and all required structure

Return the varied Python code:
"""

# Multi-asset strategy prompt
MULTI_ASSET_CODE_GENERATION_PROMPT = """
Translate the following multi-asset trading strategy hypothesis into Zipline code.

HYPOTHESIS:
{hypothesis}

ASSET UNIVERSE:
{assets}

BENCHMARK:
{benchmark}

STRATEGY TYPE:
{strategy_type} (e.g., long-short, sector rotation, pairs trading)

REQUIREMENTS:
1. Initialize context with multiple assets using symbols()
2. Implement asset selection logic (e.g., rank by momentum, select top N)
3. Handle portfolio construction (e.g., equal weight, risk parity)
4. Implement rebalancing logic (e.g., monthly, quarterly)
5. Manage long and short positions if applicable
6. Ensure gross exposure <= {max_exposure}
7. Ensure net exposure within {net_exposure_range}

CODE STRUCTURE:
- initialize(context): Set up assets, commission, slippage, benchmark
- handle_data(context, data): Calculate signals, select assets, rebalance portfolio

Return the complete Python code:
"""

# All prompts export list
__all__ = [
    'CODING_TEAM_SYSTEM_PROMPT',
    'CODE_GENERATION_PROMPT',
    'CODE_DEBUGGING_PROMPT',
    'CODE_OPTIMIZATION_PROMPT',
    'ZIPLINE_STRUCTURE_TEMPLATE',
    'CODE_VALIDATION_PROMPT',
    'STRATEGY_VARIATION_PROMPT',
    'MULTI_ASSET_CODE_GENERATION_PROMPT',
]
