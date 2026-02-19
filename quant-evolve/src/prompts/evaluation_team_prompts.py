"""
Evaluation Team Prompts for QuantEvolve

This module defines all prompt templates used by the Evaluation Team agent to:
1. Score hypotheses, code quality, and backtest results
2. Extract actionable insights from strategy performance
3. Categorize strategies into the 8 predefined families for feature map encoding
4. Manage and curate the insights repository

All prompts follow the exact specifications from Appendix A of the QuantEvolve paper.
"""

# ============================================================================
# HYPOTHESIS EVALUATION PROMPT
# ============================================================================

HYPOTHESIS_EVALUATION_PROMPT = """You are an expert quantitative trading researcher evaluating a trading strategy hypothesis.

## Task
Evaluate the following hypothesis on a scale of 0.0 to 1.0 based on:
1. Novelty: Is this hypothesis original or a minor variation of existing strategies?
2. Financial Soundness: Is the rationale grounded in market microstructure or behavioral finance?
3. Testability: Can this hypothesis be clearly translated into executable code?
4. Potential Impact: Could this hypothesis lead to significant risk-adjusted returns?

## Hypothesis to Evaluate
<hypothesis>
{hypothesis}
</hypothesis>

<rationale>
{rationale}
</rationale>

<objectives>
{objectives}
</objectives>

## Evaluation Criteria
- 0.0-0.2: Poor - Flawed logic, untestable, or trivial
- 0.2-0.4: Below Average - Weak rationale or low novelty
- 0.4-0.6: Average - Sound but common idea
- 0.6-0.8: Good - Novel with strong financial rationale
- 0.8-1.0: Excellent - Highly novel, well-grounded, high potential

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "novelty_score": 0.0-1.0,
    "financial_soundness_score": 0.0-1.0,
    "testability_score": 0.0-1.0,
    "potential_impact_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "recommendation": "proceed" | "modify" | "reject",
    "suggested_modifications": ["list of specific improvements"]
}}

Calculate overall_score as the weighted average:
overall_score = 0.25 * novelty + 0.30 * financial_soundness + 0.20 * testability + 0.25 * potential_impact

Be critical and objective. Do not inflate scores."""


# ============================================================================
# CODE EVALUATION PROMPT
# ============================================================================

CODE_EVALUATION_PROMPT = """You are a senior quantitative developer reviewing trading strategy code for Zipline Reloaded.

## Task
Evaluate the following Python code on a scale of 0.0 to 1.0 based on:
1. Correctness: Does it follow Zipline API conventions?
2. Safety: Does it avoid lookahead bias and handle exceptions?
3. Efficiency: Is the code computationally efficient for backtesting?
4. Robustness: Does it handle edge cases (missing data, zero volumes, etc.)?

## Code to Evaluate
```python
{code}
```

## Mandatory Requirements Checklist
1. [ ] Uses `data.history()` for all price/volume access (NO `data.current()`)
2. [ ] Sets commission: `context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))`
3. [ ] Sets slippage: `context.set_slippage(slippage.VolumeShareSlippage())`
4. [ ] Has both `initialize(context)` and `handle_data(context, data)` functions
5. [ ] Orders respect available cash (no over-leveraging)
6. [ ] Handles exceptions with try-except blocks
7. [ ] Uses lookback windows (no single-bar decisions without context)
8. [ ] No hardcoded future dates or values

## Evaluation Criteria
- 0.0-0.2: Critical errors, will not run
- 0.2-0.4: Major issues, likely to fail or produce incorrect results
- 0.4-0.6: Minor issues, runs but may have bugs
- 0.6-0.8: Good code, follows most best practices
- 0.8-1.0: Excellent code, production-ready

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "correctness_score": 0.0-1.0,
    "safety_score": 0.0-1.0,
    "efficiency_score": 0.0-1.0,
    "robustness_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "checklist_results": {{
        "uses_data_history": true/false,
        "sets_commission": true/false,
        "sets_slippage": true/false,
        "has_initialize": true/false,
        "has_handle_data": true/false,
        "respects_cash": true/false,
        "handles_exceptions": true/false,
        "uses_lookback": true/false
    }},
    "errors_found": ["list of specific errors or violations"],
    "suggested_fixes": ["list of specific code improvements"],
    "ready_for_backtest": true/false
}}

Calculate overall_score as the average of the four component scores.

Be thorough and identify all violations of the mandatory requirements."""


# ============================================================================
# RESULTS ANALYSIS PROMPT
# ============================================================================

RESULTS_ANALYSIS_PROMPT = """You are a quantitative analyst evaluating backtest results for a trading strategy.

## Task
Analyze the following performance metrics and provide a comprehensive evaluation on a scale of 0.0 to 1.0.

## Strategy Performance Metrics
- Sharpe Ratio (SR): {sharpe_ratio}
- Sortino Ratio (SOR): {sortino_ratio}
- Information Ratio (IR): {information_ratio}
- Max Drawdown (MDD): {max_drawdown}
- Total Return: {total_return}
- Annualized Return: {annualized_return}
- Annualized Volatility: {volatility}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}
- Trading Frequency: {trading_frequency}
- Calmar Ratio: {calmar_ratio}
- Combined Score (SR + IR + MDD): {combined_score}

## Benchmark Comparison
- Benchmark Sharpe Ratio: {benchmark_sharpe}
- Benchmark Total Return: {benchmark_return}
- Excess Return (Alpha): {excess_return}

## Evaluation Criteria
Score based on:
1. Risk-Adjusted Returns: Is SR > 1.0? Is SOR > SR?
2. Drawdown Control: Is MDD > -20%? Is Calmar > 1.0?
3. Consistency: Is win rate > 45%? Is profit factor > 1.2?
4. Benchmark Outperformance: Does it beat the benchmark on SR and total return?
5. Trading Efficiency: Is the strategy overtrading (frequency > 0.8) or undertrading (frequency < 0.05)?

## Scoring Scale
- 0.0-0.2: Poor - Loses money or extreme drawdowns
- 0.2-0.4: Below Average - Positive returns but underperforms benchmark
- 0.4-0.6: Average - Modest outperformance, acceptable risk
- 0.6-0.8: Good - Strong risk-adjusted returns, beats benchmark
- 0.8-1.0: Excellent - Exceptional performance across all metrics

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "risk_adjusted_score": 0.0-1.0,
    "drawdown_score": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "benchmark_score": 0.0-1.0,
    "efficiency_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "performance_summary": "2-3 sentence summary of performance",
    "key_strengths": ["list of top 3 strengths"],
    "key_weaknesses": ["list of top 3 weaknesses"],
    "risk_assessment": "low" | "medium" | "high",
    "recommendation": "archive" | "further_testing" | "reject",
    "suggested_improvements": ["list of specific strategy improvements"]
}}

Calculate overall_score as:
overall_score = 0.30 * risk_adjusted + 0.25 * drawdown + 0.20 * consistency + 0.15 * benchmark + 0.10 * efficiency

Be objective and data-driven in your assessment."""


# ============================================================================
# INSIGHT EXTRACTION PROMPT
# ============================================================================

INSIGHT_EXTRACTION_PROMPT = """You are a knowledge extraction specialist analyzing trading strategy evolution.

## Task
Extract actionable insights from the strategy analysis that can guide future hypothesis generation.

## Strategy Context
- Generation: {generation}
- Island ID: {island_id}
- Strategy Category: {category}
- Hypothesis: {hypothesis}
- Performance Metrics: SR={sharpe_ratio}, MDD={max_drawdown}, Return={total_return}
- Evaluation Scores: Hypothesis={hypothesis_score}, Code={code_score}, Results={results_score}

## Insight Categories
Extract insights for the following categories:
1. Market Regime: What market conditions favor this strategy?
2. Parameter Sensitivity: Which parameters are most critical?
3. Risk Management: What risk controls are effective?
4. Entry/Exit Logic: What timing signals work well?
5. Asset Selection: Which assets or asset classes are suitable?
6. Combination Potential: Can this be combined with other strategies?

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "insight_id": "unique_identifier_string",
    "generation": {generation},
    "strategy_id": "{strategy_id}",
    "insight_categories": {{
        "market_regime": "insight text or null",
        "parameter_sensitivity": "insight text or null",
        "risk_management": "insight text or null",
        "entry_exit_logic": "insight text or null",
        "asset_selection": "insight text or null",
        "combination_potential": "insight text or null"
    }},
    "key_finding": "single most important insight (1-2 sentences)",
    "actionable_recommendation": "specific recommendation for future strategies",
    "confidence_level": "low" | "medium" | "high",
    "novelty": "known" | "incremental" | "novel",
    "transferability": "market_specific" | "asset_class" | "universal",
    "tags": ["list of relevant keywords"]
}}

Guidelines:
- Only extract insights with medium or high confidence
- Focus on novel or incremental insights (skip known patterns)
- Make recommendations specific and actionable
- Use tags for efficient retrieval (e.g., "momentum", "mean-reversion", "volatility", "breakout")

If no significant insights can be extracted, return:
{{
    "insight_id": null,
    "generation": {generation},
    "strategy_id": "{strategy_id}",
    "insight_categories": {{}},
    "key_finding": "No significant insights extracted",
    "actionable_recommendation": null,
    "confidence_level": "low",
    "novelty": "known",
    "transferability": "market_specific",
    "tags": []
}}"""


# ============================================================================
# STRATEGY CATEGORIZATION PROMPT
# ============================================================================

STRATEGY_CATEGORIZATION_PROMPT = """You are a strategy classification expert mapping trading code to predefined strategy families.

## Task
Analyze the following trading strategy code and categorize it into one or more of the 8 standard strategy families.

## Strategy Families (with bit indices for binary encoding)
0. Momentum: Trends, breakouts, moving average crossovers
1. Mean Reversion: Oversold/overbought, statistical arbitrage, pairs trading
2. Value: Fundamental ratios, P/E, P/B, dividend yield
3. Growth: Earnings growth, revenue growth, momentum fundamentals
4. Quality: Profitability, stability, low debt, high ROE
5. Volatility: Volatility targeting, variance risk premium, straddles
6. Carry: Term structure, roll yield, interest rate differentials
7. Event-Driven: Earnings announcements, economic data, corporate actions

## Code to Analyze
```python
{code}
```

## Analysis Guidelines
1. Identify the primary signal generation logic
2. Determine the holding period (intraday, daily, weekly, monthly)
3. Analyze the entry and exit conditions
4. Check for risk management techniques
5. Map to the most appropriate strategy family/families

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "primary_category": "category_name",
    "primary_category_bit": 0-7,
    "secondary_categories": ["list of other applicable categories"],
    "secondary_category_bits": [list of bit indices],
    "category_bits": "8-bit binary string (e.g., '10000000' for pure momentum)",
    "confidence": 0.0-1.0,
    "rationale": "2-3 sentence explanation of categorization",
    "trading_frequency": "intraday" | "daily" | "weekly" | "monthly",
    "key_indicators": ["list of technical/fundamental indicators used"],
    "strategy_complexity": "simple" | "moderate" | "complex"
}}

To compute category_bits:
- Start with "00000000"
- Set bit to "1" for each applicable category (primary and secondary)
- Example: If momentum (bit 0) and mean reversion (bit 1) apply, return "11000000"

Be precise in your categorization as this affects the feature map diversity tracking."""


# ============================================================================
# COMPREHENSIVE EVALUATION WORKFLOW PROMPT
# ============================================================================

COMPREHENSIVE_EVALUATION_PROMPT = """You are the Evaluation Team for QuantEvolve, responsible for comprehensive strategy assessment.

## Task
Perform a complete evaluation of a newly generated trading strategy, including:
1. Hypothesis quality assessment
2. Code quality review
3. Backtest results analysis
4. Strategy categorization
5. Insight extraction

## Input Data
### Hypothesis
<hypothesis>
{hypothesis}
</hypothesis>

<rationale>
{rationale}
</rationale>

<objectives>
{objectives}
</objectives>

### Code
```python
{code}
```

### Backtest Results
- Sharpe Ratio: {sharpe_ratio}
- Sortino Ratio: {sortino_ratio}
- Information Ratio: {information_ratio}
- Max Drawdown: {max_drawdown}
- Total Return: {total_return}
- Annualized Return: {annualized_return}
- Volatility: {volatility}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}
- Trading Frequency: {trading_frequency}
- Calmar Ratio: {calmar_ratio}
- Combined Score: {combined_score}

### Benchmark
- Benchmark Sharpe: {benchmark_sharpe}
- Benchmark Return: {benchmark_return}

### Context
- Generation: {generation}
- Island ID: {island_id}
- Parent Strategy ID: {parent_id}
- Cousin Strategy IDs: {cousin_ids}

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "strategy_id": "{strategy_id}",
    "generation": {generation},
    "island_id": {island_id},
    "evaluation_scores": {{
        "hypothesis_score": 0.0-1.0,
        "code_score": 0.0-1.0,
        "results_score": 0.0-1.0,
        "weighted_score": 0.0-1.0
    }},
    "categorization": {{
        "primary_category": "category_name",
        "category_bits": "8-bit binary string",
        "trading_frequency": "frequency",
        "confidence": 0.0-1.0
    }},
    "feature_vector": {{
        "category_bits": "8-bit string",
        "trading_frequency_bin": 0-15,
        "max_drawdown_bin": 0-15,
        "sharpe_ratio_bin": 0-15,
        "sortino_ratio_bin": 0-15,
        "total_return_bin": 0-15
    }},
    "insights": {{
        "insight_id": "id or null",
        "key_finding": "finding text",
        "actionable_recommendation": "recommendation text",
        "confidence_level": "low|medium|high",
        "tags": ["tags"]
    }},
    "recommendation": "archive" | "further_testing" | "reject",
    "next_steps": ["list of recommended actions"]
}}

Calculate weighted_score as:
weighted_score = 0.30 * hypothesis_score + 0.30 * code_score + 0.40 * results_score

This comprehensive evaluation will be used to update the feature map and guide future evolution."""


# ============================================================================
# INSIGHT CURATION PROMPT
# ============================================================================

INSIGHT_CURATION_PROMPT = """You are a knowledge curator managing the insights repository for QuantEvolve.

## Task
Review the accumulated insights and curate them by:
1. Removing duplicates or near-duplicates
2. Consolidating related insights
3. Ranking by novelty and actionability
4. Selecting the top K most valuable insights

## Current Insights Repository
{insights_json}

## Curation Criteria
1. Novelty: Prefer novel insights over known patterns
2. Actionability: Prefer insights with clear recommendations
3. Confidence: Prefer high-confidence insights
4. Diversity: Ensure coverage across different strategy categories
5. Recency: Weight recent insights slightly higher (last 50 generations)

## Output Format
Return ONLY a JSON object with the following structure:
{{
    "curated_insights": [
        {{
            "insight_id": "id",
            "key_finding": "finding",
            "actionable_recommendation": "recommendation",
            "confidence_level": "level",
            "novelty": "novel|incremental",
            "tags": ["tags"],
            "priority_score": 0.0-1.0
        }}
    ],
    "removed_duplicates": ["list of removed insight IDs"],
    "consolidated_groups": [
        {{
            "representative_id": "id",
            "merged_ids": ["list of merged insight IDs"],
            "rationale": "why these were consolidated"
        }}
    ],
    "summary_statistics": {{
        "total_insights_reviewed": int,
        "insights_retained": int,
        "duplicates_removed": int,
        "groups_consolidated": int,
        "category_distribution": {{
            "momentum": int,
            "mean_reversion": int,
            "value": int,
            "growth": int,
            "quality": int,
            "volatility": int,
            "carry": int,
            "event_driven": int
        }}
    }}
}}

Select the top {max_insights} insights based on priority_score.

Calculate priority_score as:
priority_score = 0.35 * novelty_weight + 0.30 * actionability_weight + 0.20 * confidence_weight + 0.15 * recency_weight

Where:
- novelty_weight: novel=1.0, incremental=0.6, known=0.2
- actionability_weight: 1.0 if recommendation exists and is specific, 0.5 if vague, 0.0 if null
- confidence_weight: high=1.0, medium=0.6, low=0.2
- recency_weight: normalized based on generation (recent=1.0, old=0.5)"""


# ============================================================================
# EXPORT ALL PROMPTS
# ============================================================================

__all__ = [
    'HYPOTHESIS_EVALUATION_PROMPT',
    'CODE_EVALUATION_PROMPT',
    'RESULTS_ANALYSIS_PROMPT',
    'INSIGHT_EXTRACTION_PROMPT',
    'STRATEGY_CATEGORIZATION_PROMPT',
    'COMPREHENSIVE_EVALUATION_PROMPT',
    'INSIGHT_CURATION_PROMPT',
]
