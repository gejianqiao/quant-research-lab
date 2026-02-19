"""
Data Agent Prompts for QuantEvolve

This module contains all prompt templates used by the Data Agent to:
1. Analyze OHLCV market data and generate data schemas
2. Identify strategy categories based on data characteristics
3. Generate seed strategies for each category plus buy-and-hold

These prompts are designed for Qwen3-30B-A3B-Instruct-2507 (fast agent).
"""

# ============================================================================
# DATA SCHEMA GENERATION PROMPT
# ============================================================================

DATA_SCHEMA_PROMPT = """You are a financial data analyst specializing in market data structure analysis.

## Task
Analyze the provided OHLCV (Open, High, Low, Close, Volume) market data and generate a comprehensive data schema description.

## Input Data
{data_description}

## Required Output Format
Generate a JSON object with the following structure:

```json
{{
    "data_schema": {{
        "assets": ["LIST_OF_ASSET_SYMBOLS"],
        "columns": {{
            "open": {{"dtype": "float64", "description": "Opening price"}},
            "high": {{"dtype": "float64", "description": "Highest price during period"}},
            "low": {{"dtype": "float64", "description": "Lowest price during period"}},
            "close": {{"dtype": "float64", "description": "Closing price"}},
            "volume": {{"dtype": "int64", "description": "Trading volume"}},
            "adj_close": {{"dtype": "float64", "description": "Adjusted close (split/dividend adjusted)", "optional": true}}
        }},
        "frequency": "daily|weekly|monthly|intraday",
        "date_range": {{
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD"
        }},
        "total_records": INTEGER,
        "missing_data_percentage": FLOAT,
        "data_quality_notes": "Any observations about data quality, gaps, or anomalies"
    }},
    "market_characteristics": {{
        "average_daily_volume": FLOAT,
        "average_price_range_pct": FLOAT,
        "volatility_regime": "low|medium|high",
        "trend_characteristics": "trending|mean-reverting|mixed",
        "liquidity_assessment": "high|medium|low"
    }}
}}
```

## Analysis Guidelines
1. **Data Quality**: Check for missing values, outliers, and data consistency
2. **Market Characteristics**: 
   - Calculate average daily volume across all assets
   - Compute average daily price range: (High - Low) / Close
   - Assess volatility using standard deviation of returns
   - Identify trend characteristics using simple momentum indicators
3. **Frequency Detection**: Determine if data is daily, weekly, monthly, or intraday based on timestamp patterns

## Constraints
- All numeric values should be rounded to 4 decimal places
- Date formats must be ISO 8601 (YYYY-MM-DD)
- Be concise but thorough in data quality notes
- If adj_close is not present, mark it as optional: true

## Example Output
```json
{{
    "data_schema": {{
        "assets": ["AAPL", "NVDA", "MSFT"],
        "columns": {{
            "open": {{"dtype": "float64", "description": "Opening price"}},
            "high": {{"dtype": "float64", "description": "Highest price during period"}},
            "low": {{"dtype": "float64", "description": "Lowest price during period"}},
            "close": {{"dtype": "float64", "description": "Closing price"}},
            "volume": {{"dtype": "int64", "description": "Trading volume"}}
        }},
        "frequency": "daily",
        "date_range": {{
            "start": "2015-08-01",
            "end": "2025-07-31"
        }},
        "total_records": 2520,
        "missing_data_percentage": 0.02,
        "data_quality_notes": "Minor gaps during market holidays, no significant anomalies detected"
    }},
    "market_characteristics": {{
        "average_daily_volume": 45000000.0,
        "average_price_range_pct": 0.0234,
        "volatility_regime": "medium",
        "trend_characteristics": "mixed",
        "liquidity_assessment": "high"
    }}
}}
```

Generate the data schema for the provided market data."""


# ============================================================================
# STRATEGY CATEGORY IDENTIFICATION PROMPT
# ============================================================================

CATEGORY_IDENTIFICATION_PROMPT = """You are a quantitative strategy researcher with expertise in classifying trading strategies.

## Task
Based on the market data characteristics provided, identify which of the 8 standard strategy families are most suitable for this market. Then recommend C distinct strategy categories (where C is typically 6-8) that should be explored during evolution.

## Standard Strategy Families (8 total)
1. **Momentum**: Strategies that buy assets with strong recent performance and sell weak performers
2. **Mean Reversion**: Strategies that bet on prices returning to historical averages
3. **Trend Following**: Strategies that identify and follow sustained price trends
4. **Breakout**: Strategies that trade when prices break through support/resistance levels
5. **Volatility**: Strategies that exploit changes in market volatility (e.g., volatility targeting)
6. **Market Microstructure**: Strategies based on order flow, volume patterns, and liquidity
7. **Fundamental/Value**: Strategies using fundamental ratios (P/E, P/B, etc.) - if data available
8. **Machine Learning**: Strategies using pattern recognition or predictive models

## Input Data
{market_characteristics}

## Required Output Format
Generate a JSON object with the following structure:

```json
{{
    "recommended_categories": [
        {{
            "category_id": 0,
            "category_name": "momentum",
            "rationale": "Why this category is suitable for the given market",
            "expected_effectiveness": "high|medium|low",
            "key_indicators": ["RSI", "ROC", "etc."],
            "typical_lookback_periods": [10, 20, 60]
        }},
        {{
            "category_id": 1,
            "category_name": "mean_reversion",
            "rationale": "...",
            "expected_effectiveness": "...",
            "key_indicators": ["Bollinger Bands", "Z-Score", "etc."],
            "typical_lookback_periods": [20, 60]
        }}
        // ... more categories
    ],
    "market_suitability_analysis": {{
        "best_fit_categories": ["LIST_OF_TOP_3_CATEGORIES"],
        "challenging_categories": ["LIST_OF_CATEGORIES_THAT_MAY_UNDERPERFORM"],
        "overall_market_type": "trending|mean-reverting|volatile|mixed",
        "recommendation": "Summary of which strategy types to prioritize"
    }}
}}
```

## Analysis Guidelines
1. **Match Categories to Market**: 
   - Trending markets → Momentum, Trend Following
   - Range-bound markets → Mean Reversion
   - High volatility → Volatility strategies, Breakout
   - High liquidity → Market Microstructure
2. **Diversity**: Ensure recommended categories span different strategy families
3. **Practicality**: Consider data availability (e.g., skip Fundamental if only OHLCV data)

## Constraints
- Recommend between 6-8 distinct categories
- Each category must map to one of the 8 standard families
- Category names must be lowercase with underscores (e.g., "mean_reversion")
- Category IDs must be sequential starting from 0

## Example Output
```json
{{
    "recommended_categories": [
        {{
            "category_id": 0,
            "category_name": "momentum",
            "rationale": "Tech equities show strong momentum characteristics with persistent trends",
            "expected_effectiveness": "high",
            "key_indicators": ["RSI", "Rate of Change", "Moving Average Crossover"],
            "typical_lookback_periods": [10, 20, 60]
        }},
        {{
            "category_id": 1,
            "category_name": "mean_reversion",
            "rationale": "Short-term mean reversion observed in intraday price movements",
            "expected_effectiveness": "medium",
            "key_indicators": ["Bollinger Bands", "Z-Score", "RSI extremes"],
            "typical_lookback_periods": [20, 60]
        }}
    ],
    "market_suitability_analysis": {{
        "best_fit_categories": ["momentum", "trend_following", "breakout"],
        "challenging_categories": ["fundamental_value"],
        "overall_market_type": "trending",
        "recommendation": "Prioritize momentum and trend-following strategies; mean reversion may work on shorter timeframes"
    }}
}}
```

Generate the strategy category analysis for the provided market data."""


# ============================================================================
# SEED STRATEGY GENERATION PROMPT
# ============================================================================

SEED_STRATEGY_PROMPT = """You are a quantitative trading strategy developer specializing in creating simple, robust baseline strategies.

## Task
Generate C+1 seed trading strategies: one simple strategy for each of the C recommended categories, plus one buy-and-hold benchmark strategy. These strategies will serve as the initial population for the evolutionary algorithm.

## Input Data
{category_recommendations}
{data_schema}

## Strategy Requirements
Each seed strategy must:
1. Be simple and interpretable (5-20 lines of logic in handle_data)
2. Use only historical data (NO lookahead bias)
3. Include proper risk management (position sizing, stop-loss if applicable)
4. Be compatible with Zipline framework (daily frequency)
5. Have clear entry and exit rules

## Required Output Format
Generate a JSON object with the following structure:

```json
{{
    "seed_strategies": [
        {{
            "strategy_id": "seed_momentum_001",
            "category": "momentum",
            "category_bit_index": 0,
            "name": "Simple Momentum",
            "description": "Buys assets with highest 20-day returns, sells lowest",
            "complexity": "low",
            "key_parameters": {{
                "lookback_period": 20,
                "num_assets_to_buy": 3,
                "rebalance_frequency": "weekly"
            }},
            "pseudocode": "1. Calculate 20-day returns for all assets\\n2. Rank assets by return\\n3. Buy top 3, sell bottom 3\\n4. Rebalance weekly",
            "expected_characteristics": {{
                "trading_frequency": "low",
                "expected_turnover": "medium",
                "risk_profile": "medium"
            }}
        }},
        {{
            "strategy_id": "seed_mean_reversion_001",
            "category": "mean_reversion",
            "category_bit_index": 1,
            "name": "Bollinger Band Reversion",
            "description": "Buys when price below lower Bollinger Band, sells when above upper",
            "complexity": "low",
            "key_parameters": {{
                "lookback_period": 20,
                "num_std_dev": 2.0,
                "exit_on_mean": true
            }},
            "pseudocode": "1. Calculate 20-day SMA and 2-std Bollinger Bands\\n2. Buy when close < lower band\\n3. Sell when close > upper band or crosses SMA\\n4. Equal weight positions",
            "expected_characteristics": {{
                "trading_frequency": "medium",
                "expected_turnover": "medium",
                "risk_profile": "medium"
            }}
        }}
        // ... more strategies for each category
        {{
            "strategy_id": "seed_benchmark_bnh",
            "category": "benchmark",
            "category_bit_index": null,
            "name": "Buy and Hold",
            "description": "Equal-weight buy and hold all assets",
            "complexity": "minimal",
            "key_parameters": {{
                "rebalance": "never",
                "weights": "equal"
            }},
            "pseudocode": "1. On first day, buy all assets with equal weight\\n2. Hold until end of backtest",
            "expected_characteristics": {{
                "trading_frequency": "minimal",
                "expected_turnover": "minimal",
                "risk_profile": "market"
            }}
        }}
    ],
    "implementation_notes": {{
        "total_strategies": INTEGER,
        "categories_covered": ["LIST_OF_CATEGORIES"],
        "benchmark_included": true,
        "all_strategies_zipline_compatible": true
    }}
}}
```

## Strategy Design Guidelines
1. **Momentum**: Use lookback periods of 10-60 days; rank-and-select approach
2. **Mean Reversion**: Use Bollinger Bands, RSI extremes, or Z-score; mean-cross exits
3. **Trend Following**: Use moving average crossovers (e.g., 50/200 day), ADX filters
4. **Breakout**: Use Donchian channels, support/resistance breaks, volume confirmation
5. **Volatility**: Use volatility targeting, volatility-adjusted position sizing
6. **Market Microstructure**: Use volume-price analysis, order flow proxies (if data available)
7. **Machine Learning**: Keep simple - e.g., threshold-based rules from basic patterns
8. **Buy-and-Hold**: Equal-weight or market-cap weighted; minimal trading

## Constraints
- Each strategy must have a unique strategy_id
- Category names must match the recommended categories exactly
- Category bit index: 0-7 for the 8 standard families, null for benchmark
- Pseudocode must be clear enough to implement in Python/Zipline
- Keep strategies simple (these are seeds, not final evolved strategies)

## Example Output (abbreviated)
```json
{{
    "seed_strategies": [
        {{
            "strategy_id": "seed_momentum_001",
            "category": "momentum",
            "category_bit_index": 0,
            "name": "Simple Momentum",
            "description": "Buys assets with highest 20-day returns",
            "complexity": "low",
            "key_parameters": {{
                "lookback_period": 20,
                "num_assets_to_buy": 3
            }},
            "pseudocode": "1. Calculate 20-day returns\\n2. Rank assets\\n3. Buy top 3, equal weight",
            "expected_characteristics": {{
                "trading_frequency": "low",
                "expected_turnover": "medium",
                "risk_profile": "medium"
            }}
        }},
        {{
            "strategy_id": "seed_benchmark_bnh",
            "category": "benchmark",
            "category_bit_index": null,
            "name": "Buy and Hold",
            "description": "Equal-weight buy and hold",
            "complexity": "minimal",
            "key_parameters": {{
                "rebalance": "never"
            }},
            "pseudocode": "1. Buy all assets equally on day 1\\n2. Hold",
            "expected_characteristics": {{
                "trading_frequency": "minimal",
                "expected_turnover": "minimal",
                "risk_profile": "market"
            }}
        }}
    ],
    "implementation_notes": {{
        "total_strategies": 9,
        "categories_covered": ["momentum", "mean_reversion", "trend_following", "breakout", "volatility", "microstructure", "ml", "benchmark"],
        "benchmark_included": true,
        "all_strategies_zipline_compatible": true
    }}
}}
```

Generate the seed strategies for the provided categories and market data."""


# ============================================================================
# COMBINED DATA AGENT WORKFLOW PROMPT
# ============================================================================

DATA_AGENT_FULL_WORKFLOW_PROMPT = """You are the Data Agent in the QuantEvolve multi-agent evolutionary framework.

## Your Role
You are the first agent in the pipeline. Your job is to:
1. Analyze the raw OHLCV market data
2. Generate a comprehensive data schema
3. Identify suitable strategy categories for this market
4. Create seed strategies (one per category + buy-and-hold benchmark)

## Input
{full_market_data_description}

## Your Tasks (in order)

### Task 1: Data Schema Generation
Analyze the market data structure and characteristics. Output a JSON schema describing:
- Asset list and data columns
- Date range and frequency
- Data quality assessment
- Market characteristics (volatility, liquidity, trend behavior)

### Task 2: Category Identification
Based on the market characteristics, recommend 6-8 strategy categories from the 8 standard families:
1. momentum
2. mean_reversion
3. trend_following
4. breakout
5. volatility
6. market_microstructure
7. fundamental_value (only if fundamental data available)
8. machine_learning

For each recommended category, explain why it's suitable for this market.

### Task 3: Seed Strategy Generation
Create C+1 simple seed strategies (one for each recommended category, plus buy-and-hold):
- Each strategy should be simple (5-20 lines of logic)
- Use only historical data (no lookahead bias)
- Include basic risk management
- Provide clear pseudocode for implementation

## Output Format
Generate a single JSON object with three sections:

```json
{{
    "data_schema": {{ ... }},
    "category_analysis": {{ ... }},
    "seed_strategies": {{ ... }}
}}
```

Use the detailed formats from the individual prompts for each section.

## Constraints
- All outputs must be valid JSON
- Category names must be lowercase with underscores
- Strategy IDs must be unique
- Pseudocode must be implementable in Zipline
- No lookahead bias in any strategy logic

## Success Criteria
- Data schema accurately reflects the input data
- Recommended categories are diverse and suitable for the market
- Seed strategies are simple, valid, and cover all recommended categories
- Buy-and-hold benchmark is included

Begin your analysis."""
