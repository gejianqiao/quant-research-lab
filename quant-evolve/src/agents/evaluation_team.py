"""
Evaluation Team Agent for QuantEvolve.

This module implements the EvaluationTeam agent responsible for:
- Scoring hypotheses, code, and backtest results (0.0-1.0)
- Extracting actionable insights in JSON format
- Categorizing strategies into 8 strategy families for feature map
- Managing the insight repository with curation every K generations
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvaluationTeam(BaseAgent):
    """
    Evaluation Team Agent for analyzing and scoring trading strategies.
    
    This agent uses deep reasoning LLMs (Qwen3-80B) to:
    1. Evaluate hypothesis quality and novelty
    2. Assess code quality and adherence to constraints
    3. Analyze backtest results and performance metrics
    4. Extract structured insights for future generations
    5. Categorize strategies into predefined families
    """
    
    def __init__(
        self,
        model_name: str = "Qwen3-Next-80B-A3B-Instruct",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize the Evaluation Team agent.
        
        Args:
            model_name: LLM model for deep reasoning (default: Qwen3-80B)
            api_key: API key for LLM service
            api_base: Base URL for LLM API endpoint
            temperature: Generation temperature (lower for consistent scoring)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Canonical category ordering for bit encoding.
        self.strategy_categories = [
            "momentum",
            "mean_reversion",
            "trend_following",
            "breakout",
            "statistical_arb",
            "machine_learning",
            "sentiment_analysis",
            "multi_factor",
        ]
        
        # Insight repository for curation
        self.insights_repository: List[Dict[str, Any]] = []
        
        # Scoring history for calibration
        self.scoring_history: List[Dict[str, float]] = []
    
    def evaluate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        generation: int,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality and novelty of a trading hypothesis.
        
        Args:
            hypothesis: Dictionary containing hypothesis components from ResearchAgent
            generation: Current generation number
            market_context: Market data schema and characteristics
            
        Returns:
            Dictionary containing:
                - hypothesis_evaluation_score: 0.0-1.0
                - novelty_score: 0.0-1.0
                - feasibility_score: 0.0-1.0
                - detailed_feedback: Text explanation
        """
        system_prompt = self._get_hypothesis_evaluation_system_prompt()
        
        user_prompt = self._format_hypothesis_evaluation_prompt(
            hypothesis=hypothesis,
            generation=generation,
            market_context=market_context
        )
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                max_retries=3
            )
            
            # Parse and validate response
            evaluation = self._parse_hypothesis_evaluation(response)
            
            logger.info(
                f"Hypothesis evaluation complete - "
                f"Score: {evaluation.get('hypothesis_evaluation_score', 0):.3f}, "
                f"Novelty: {evaluation.get('novelty_score', 0):.3f}"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Hypothesis evaluation failed: {e}")
            return self._create_fallback_hypothesis_evaluation()
    
    def evaluate_code(
        self,
        code: str,
        hypothesis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate code quality and adherence to constraints.
        
        Args:
            code: Generated Python strategy code
            hypothesis: Original hypothesis that code implements
            constraints: Trading constraints (no lookahead, position limits, etc.)
            
        Returns:
            Dictionary containing:
                - program_alignment_evaluation_score: 0.0-1.0
                - code_quality_score: 0.0-1.0
                - constraint_compliance: Dict of constraint checks
                - code_issues: List of identified problems
        """
        system_prompt = self._get_code_evaluation_system_prompt()
        
        user_prompt = self._format_code_evaluation_prompt(
            code=code,
            hypothesis=hypothesis,
            constraints=constraints
        )
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                max_retries=3
            )
            
            evaluation = self._parse_code_evaluation(response)
            
            logger.info(
                f"Code evaluation complete - "
                f"Alignment: {evaluation.get('program_alignment_evaluation_score', 0):.3f}, "
                f"Quality: {evaluation.get('code_quality_score', 0):.3f}"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Code evaluation failed: {e}")
            return self._create_fallback_code_evaluation()
    
    def analyze_results(
        self,
        metrics: Dict[str, float],
        returns_series: List[float],
        benchmark_returns: List[float],
        strategy_code: str,
        hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze backtest results and extract insights.
        
        Args:
            metrics: Performance metrics (SR, SOR, IR, MDD, etc.)
            returns_series: List of strategy returns
            benchmark_returns: List of benchmark returns for comparison
            strategy_code: Generated strategy code
            hypothesis: Original hypothesis
            
        Returns:
            Dictionary containing:
                - results_analysis_score: 0.0-1.0
                - performance_insights: List of key observations
                - risk_analysis: Dict of risk metrics
                - improvement_suggestions: List of recommendations
        """
        system_prompt = self._get_results_analysis_system_prompt()
        
        user_prompt = self._format_results_analysis_prompt(
            metrics=metrics,
            returns_series=returns_series,
            benchmark_returns=benchmark_returns,
            strategy_code=strategy_code,
            hypothesis=hypothesis
        )
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                max_retries=3
            )
            
            analysis = self._parse_results_analysis(response)
            
            logger.info(
                f"Results analysis complete - "
                f"Score: {analysis.get('results_analysis_score', 0):.3f}, "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Results analysis failed: {e}")
            return self._create_fallback_results_analysis(metrics)
    
    def categorize_strategy(self, code: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize strategy into one or more of the 8 strategy families.
        
        Args:
            code: Strategy source code
            metrics: Performance metrics
            
        Returns:
            Dictionary containing:
                - primary_category: String category name
                - category_bits: 8-bit binary string for feature map
                - confidence: 0.0-1.0 confidence in categorization
                - secondary_categories: List of other relevant categories
        """
        system_prompt = self._get_categorization_system_prompt()
        
        user_prompt = self._format_categorization_prompt(code, metrics)
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                max_retries=3
            )
            
            categorization = self._parse_categorization(response)
            
            # Convert category to binary encoding
            category_bits = self._category_to_bits(categorization.get('primary_category', 'hybrid'))
            categorization['category_bits'] = category_bits
            
            logger.info(
                f"Strategy categorized: {categorization.get('primary_category')} "
                f"(bits: {category_bits})"
            )
            
            return categorization
            
        except Exception as e:
            logger.error(f"Strategy categorization failed: {e}")
            return self._create_fallback_categorization()
    
    def extract_insights(
        self,
        hypothesis: Dict[str, Any],
        code: str,
        metrics: Dict[str, float],
        analysis: Dict[str, Any],
        generation: int
    ) -> Dict[str, Any]:
        """
        Extract structured insights from a complete strategy evaluation.
        
        Args:
            hypothesis: Research hypothesis
            code: Strategy code
            metrics: Performance metrics
            analysis: Results analysis from analyze_results()
            generation: Current generation number
            
        Returns:
            Dictionary containing structured insights with all required fields:
                - insight_id: Unique identifier
                - generation: Generation number
                - category: Strategy category
                - key_finding: Main insight
                - evidence: Supporting data
                - actionability: How to use this insight
                - confidence: 0.0-1.0
                - timestamp: ISO format timestamp
        """
        system_prompt = self._get_insight_extraction_system_prompt()
        
        user_prompt = self._format_insight_extraction_prompt(
            hypothesis=hypothesis,
            code=code,
            metrics=metrics,
            analysis=analysis,
            generation=generation
        )
        
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                max_retries=3
            )
            
            insight = self._parse_insight(response, generation)
            
            # Add to repository
            self.insights_repository.append(insight)
            
            logger.info(
                f"Insight extracted: {insight.get('key_finding', '')[:50]}... "
                f"(generation {generation})"
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")
            return self._create_fallback_insight(generation)
    
    def curate_insights(self, max_insights: int = 500) -> List[Dict[str, Any]]:
        """
        Curate the insights repository by removing duplicates and consolidating.
        
        Called every K=50 generations to manage repository size.
        
        Args:
            max_insights: Maximum number of insights to retain
            
        Returns:
            List of curated insights
        """
        if len(self.insights_repository) <= max_insights:
            return self.insights_repository
        
        logger.info(
            f"Curating insights repository: {len(self.insights_repository)} -> {max_insights}"
        )
        
        # Sort by generation (most recent first) and confidence
        sorted_insights = sorted(
            self.insights_repository,
            key=lambda x: (x.get('generation', 0), x.get('confidence', 0)),
            reverse=True
        )
        
        # Remove duplicates based on key_finding similarity
        unique_insights = []
        seen_findings = set()
        
        for insight in sorted_insights:
            finding = insight.get('key_finding', '').lower()
            # Simple deduplication by hashing first 50 chars
            finding_hash = hash(finding[:50])
            
            if finding_hash not in seen_findings:
                unique_insights.append(insight)
                seen_findings.add(finding_hash)
                
                if len(unique_insights) >= max_insights:
                    break
        
        self.insights_repository = unique_insights
        
        logger.info(f"Insights curated: {len(unique_insights)} unique insights retained")
        
        return unique_insights
    
    def get_recent_insights(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get the n most recent insights for context in hypothesis generation.
        
        Args:
            n: Number of insights to return
            
        Returns:
            List of recent insight dictionaries
        """
        return self.insights_repository[-n:]
    
    def execute(
        self,
        hypothesis: Dict[str, Any],
        code: str,
        metrics: Dict[str, float],
        returns_series: List[float],
        benchmark_returns: List[float],
        generation: int,
        market_context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute complete evaluation workflow.
        
        Args:
            hypothesis: Research hypothesis
            code: Strategy code
            metrics: Backtest metrics
            returns_series: Strategy returns
            benchmark_returns: Benchmark returns
            generation: Current generation
            market_context: Market data context
            constraints: Trading constraints
            
        Returns:
            Dictionary containing all evaluation results:
                - hypothesis_score: Hypothesis evaluation score
                - code_score: Code alignment score
                - results_score: Results analysis score
                - combined_score: Weighted average
                - category: Strategy category
                - insights: Extracted insights
                - feedback: Detailed feedback
        """
        logger.info(f"Starting evaluation for generation {generation}")
        
        # 1. Evaluate hypothesis
        hypothesis_eval = self.evaluate_hypothesis(
            hypothesis=hypothesis,
            generation=generation,
            market_context=market_context
        )
        
        # 2. Evaluate code
        code_eval = self.evaluate_code(
            code=code,
            hypothesis=hypothesis,
            constraints=constraints
        )
        
        # 3. Analyze results
        results_analysis = self.analyze_results(
            metrics=metrics,
            returns_series=returns_series,
            benchmark_returns=benchmark_returns,
            strategy_code=code,
            hypothesis=hypothesis
        )
        
        # 4. Categorize strategy
        categorization = self.categorize_strategy(code=code, metrics=metrics)
        
        # 5. Extract insights
        insights = self.extract_insights(
            hypothesis=hypothesis,
            code=code,
            metrics=metrics,
            analysis=results_analysis,
            generation=generation
        )
        
        # Calculate combined score (weighted average)
        combined_score = (
            0.3 * hypothesis_eval.get('hypothesis_evaluation_score', 0) +
            0.3 * code_eval.get('program_alignment_evaluation_score', 0) +
            0.4 * results_analysis.get('results_analysis_score', 0)
        )
        
        # Record scoring history
        self.scoring_history.append({
            'generation': generation,
            'hypothesis_score': hypothesis_eval.get('hypothesis_evaluation_score', 0),
            'code_score': code_eval.get('program_alignment_evaluation_score', 0),
            'results_score': results_analysis.get('results_analysis_score', 0),
            'combined_score': combined_score
        })
        
        result = {
            'hypothesis_evaluation': hypothesis_eval,
            'code_evaluation': code_eval,
            'results_analysis': results_analysis,
            'categorization': categorization,
            'insights': insights,
            'combined_score': combined_score,
            'feedback': {
                'hypothesis': hypothesis_eval.get('detailed_feedback', ''),
                'code': code_eval.get('code_issues', []),
                'results': results_analysis.get('improvement_suggestions', [])
            }
        }
        
        logger.info(
            f"Evaluation complete for generation {generation} - "
            f"Combined Score: {combined_score:.3f}"
        )
        
        return result
    
    # =========================================================================
    # Prompt Generation Methods
    # =========================================================================
    
    def _get_hypothesis_evaluation_system_prompt(self) -> str:
        """Get system prompt for hypothesis evaluation."""
        return """You are an expert quantitative research evaluator for trading strategies.
Your task is to evaluate the quality, novelty, and feasibility of trading hypotheses.

Evaluation Criteria:
1. Novelty: Is the hypothesis original or a rehash of known strategies?
2. Financial Soundness: Does the rationale make economic/financial sense?
3. Testability: Can the hypothesis be tested with available data?
4. Clarity: Is the hypothesis clearly stated with specific mechanisms?
5. Potential Impact: Could this hypothesis lead to profitable strategies?

Score each criterion from 0.0 to 1.0 and provide detailed feedback."""
    
    def _format_hypothesis_evaluation_prompt(
        self,
        hypothesis: Dict[str, Any],
        generation: int,
        market_context: Dict[str, Any]
    ) -> str:
        """Format user prompt for hypothesis evaluation."""
        return f"""Evaluate the following trading hypothesis:

GENERATION: {generation}

HYPOTHESIS:
{hypothesis.get('hypothesis', 'N/A')}

RATIONALE:
{hypothesis.get('rationale', 'N/A')}

OBJECTIVES:
{hypothesis.get('objectives', 'N/A')}

EXPECTED INSIGHTS:
{hypothesis.get('expected_insights', 'N/A')}

RISKS AND LIMITATIONS:
{hypothesis.get('risks_limitations', 'N/A')}

MARKET CONTEXT:
- Assets: {market_context.get('assets', [])}
- Data Frequency: {market_context.get('frequency', 'daily')}
- Market Type: {market_context.get('market_type', 'equity')}

Provide your evaluation in JSON format with these fields:
{{
    "hypothesis_evaluation_score": 0.0-1.0,
    "novelty_score": 0.0-1.0,
    "feasibility_score": 0.0-1.0,
    "detailed_feedback": "string explaining your evaluation"
}}"""
    
    def _get_code_evaluation_system_prompt(self) -> str:
        """Get system prompt for code evaluation."""
        return """You are an expert code reviewer for algorithmic trading systems.
Your task is to evaluate trading strategy code for quality, correctness, and constraint compliance.

Critical Constraints to Check:
1. No lookahead bias: Only use data.history() with proper lookback
2. Proper Zipline structure: initialize() and handle_data() functions
3. Commission and slippage: Must be set in initialize()
4. Position sizing: Orders must respect available cash
5. Error handling: Must handle exceptions gracefully
6. Hypothesis alignment: Code must implement the stated hypothesis

Score each aspect from 0.0 to 1.0 and list any issues found."""
    
    def _format_code_evaluation_prompt(
        self,
        code: str,
        hypothesis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Format user prompt for code evaluation."""
        return f"""Evaluate the following trading strategy code:

HYPOTHESIS TO IMPLEMENT:
{hypothesis.get('hypothesis', 'N/A')}

CODE:
```python
{code}
```

CONSTRAINTS:
- Max position size: {constraints.get('max_position_size', '100%')}
- Max gross exposure: {constraints.get('max_gross_exposure', '100%')}
- No lookahead bias: Use only data.history()
- Commission: {constraints.get('commission', '$0.0075/share')}
- Slippage: {constraints.get('slippage', 'VolumeShareSlippage')}

Provide your evaluation in JSON format with these fields:
{{
    "program_alignment_evaluation_score": 0.0-1.0,
    "code_quality_score": 0.0-1.0,
    "constraint_compliance": {{
        "no_lookahead": true/false,
        "proper_structure": true/false,
        "commission_set": true/false,
        "position_limits": true/false
    }},
    "code_issues": ["list of identified problems"]
}}"""
    
    def _get_results_analysis_system_prompt(self) -> str:
        """Get system prompt for results analysis."""
        return """You are an expert quantitative analyst evaluating backtest results.
Your task is to analyze performance metrics, identify patterns, and extract actionable insights.

Key Metrics to Analyze:
1. Sharpe Ratio: Risk-adjusted returns
2. Sortino Ratio: Downside risk-adjusted returns
3. Information Ratio: Excess returns vs benchmark
4. Max Drawdown: Largest peak-to-trough decline
5. Total Return: Cumulative performance
6. Win Rate: Percentage of profitable periods
7. Profit Factor: Gross profits / Gross losses

Look for:
- Consistency of returns
- Performance in different market regimes
- Risk characteristics
- Comparison to benchmark
- Areas for improvement"""
    
    def _format_results_analysis_prompt(
        self,
        metrics: Dict[str, float],
        returns_series: List[float],
        benchmark_returns: List[float],
        strategy_code: str,
        hypothesis: Dict[str, Any]
    ) -> str:
        """Format user prompt for results analysis."""
        # Summarize returns series
        positive_returns = sum(1 for r in returns_series if r > 0)
        win_rate = positive_returns / len(returns_series) if returns_series else 0
        
        return f"""Analyze the following backtest results:

HYPOTHESIS:
{hypothesis.get('hypothesis', 'N/A')}

PERFORMANCE METRICS:
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
- Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}
- Information Ratio: {metrics.get('information_ratio', 0):.4f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.4f}
- Total Return: {metrics.get('total_return', 0):.4f}
- Win Rate: {win_rate:.2%}

RETURNS SUMMARY:
- Number of periods: {len(returns_series)}
- Average return: {sum(returns_series)/len(returns_series) if returns_series else 0:.4f}
- Return volatility: {(sum((r - sum(returns_series)/len(returns_series))**2 for r in returns_series)/len(returns_series))**0.5 if returns_series else 0:.4f}

Provide your analysis in JSON format with these fields:
{{
    "results_analysis_score": 0.0-1.0,
    "performance_insights": ["list of key observations"],
    "risk_analysis": {{
        "drawdown_severity": "low/medium/high",
        "volatility_level": "low/medium/high",
        "tail_risk": "low/medium/high"
    }},
    "improvement_suggestions": ["list of recommendations"]
}}"""
    
    def _get_categorization_system_prompt(self) -> str:
        """Get system prompt for strategy categorization."""
        return """You are an expert at classifying trading strategies into predefined categories.

Strategy Categories (8 families):
1. momentum: Strategies based on price momentum and trend continuation
2. mean_reversion: Strategies betting on price reverting to mean
3. trend_following: Strategies following established trends
4. breakout: Strategies trading price breakouts from ranges
5. statistical_arb: Relative value, spread, and volatility/dispersion style logic
6. machine_learning: Learned/predictive pattern models
7. sentiment_analysis: Flow/sentiment/event-driven signal logic
8. multi_factor: Composite models combining multiple factor sleeves

Analyze the code structure, indicators used, and trading logic to determine the primary category."""
    
    def _format_categorization_prompt(
        self,
        code: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Format user prompt for strategy categorization."""
        return f"""Categorize the following trading strategy:

CODE:
```python
{code[:3000]}  # Truncate for token limits
```

PERFORMANCE CHARACTERISTICS:
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.4f}
- Total Return: {metrics.get('total_return', 0):.4f}

Provide your categorization in JSON format with these fields:
{{
    "primary_category": "one of: momentum/mean_reversion/trend_following/breakout/statistical_arb/machine_learning/sentiment_analysis/multi_factor",
    "confidence": 0.0-1.0,
    "secondary_categories": ["list of other relevant categories"],
    "reasoning": "explanation of categorization decision"
}}"""
    
    def _get_insight_extraction_system_prompt(self) -> str:
        """Get system prompt for insight extraction."""
        return """You are an expert at extracting actionable insights from trading strategy experiments.

Your task is to distill key learnings that can guide future strategy development.

Each insight should be:
1. Specific: Clearly state what was learned
2. Actionable: Provide guidance for future iterations
3. Evidence-based: Supported by the data
4. Concise: One clear finding per insight

Format insights as structured JSON with all required fields."""
    
    def _format_insight_extraction_prompt(
        self,
        hypothesis: Dict[str, Any],
        code: str,
        metrics: Dict[str, float],
        analysis: Dict[str, Any],
        generation: int
    ) -> str:
        """Format user prompt for insight extraction."""
        return f"""Extract insights from this strategy experiment:

GENERATION: {generation}

HYPOTHESIS: {hypothesis.get('hypothesis', 'N/A')}

KEY METRICS:
- Sharpe: {metrics.get('sharpe_ratio', 0):.4f}
- Sortino: {metrics.get('sortino_ratio', 0):.4f}
- Max DD: {metrics.get('max_drawdown', 0):.4f}
- Return: {metrics.get('total_return', 0):.4f}

ANALYSIS INSIGHTS: {analysis.get('performance_insights', [])}

IMPROVEMENT SUGGESTIONS: {analysis.get('improvement_suggestions', [])}

Provide your extracted insight in JSON format with these fields:
{{
    "insight_id": "unique identifier",
    "generation": {generation},
    "category": "strategy category",
    "key_finding": "main insight",
    "evidence": "supporting data",
    "actionability": "how to use this insight",
    "confidence": 0.0-1.0,
    "timestamp": "ISO format timestamp"
}}"""
    
    # =========================================================================
    # Parsing Methods
    # =========================================================================
    
    def _parse_hypothesis_evaluation(self, response: Any) -> Dict[str, Any]:
        """Parse hypothesis evaluation response."""
        if isinstance(response, dict):
            return {
                'hypothesis_evaluation_score': float(response.get('hypothesis_evaluation_score', 0.5)),
                'novelty_score': float(response.get('novelty_score', 0.5)),
                'feasibility_score': float(response.get('feasibility_score', 0.5)),
                'detailed_feedback': str(response.get('detailed_feedback', 'No feedback provided'))
            }
        return self._create_fallback_hypothesis_evaluation()
    
    def _parse_code_evaluation(self, response: Any) -> Dict[str, Any]:
        """Parse code evaluation response."""
        if isinstance(response, dict):
            return {
                'program_alignment_evaluation_score': float(response.get('program_alignment_evaluation_score', 0.5)),
                'code_quality_score': float(response.get('code_quality_score', 0.5)),
                'constraint_compliance': response.get('constraint_compliance', {}),
                'code_issues': response.get('code_issues', [])
            }
        return self._create_fallback_code_evaluation()
    
    def _parse_results_analysis(self, response: Any) -> Dict[str, Any]:
        """Parse results analysis response."""
        if isinstance(response, dict):
            return {
                'results_analysis_score': float(response.get('results_analysis_score', 0.5)),
                'performance_insights': response.get('performance_insights', []),
                'risk_analysis': response.get('risk_analysis', {}),
                'improvement_suggestions': response.get('improvement_suggestions', [])
            }
        return self._create_fallback_results_analysis({})
    
    def _parse_categorization(self, response: Any) -> Dict[str, Any]:
        """Parse categorization response."""
        if isinstance(response, dict):
            primary = str(response.get('primary_category', 'multi_factor')).lower()
            alias_map = {
                'volatility': 'statistical_arb',
                'volume': 'sentiment_analysis',
                'fundamental': 'multi_factor',
                'hybrid': 'multi_factor',
                'market_microstructure': 'sentiment_analysis',
                'fundamental_value': 'multi_factor',
            }
            primary = alias_map.get(primary, primary)
            # Validate category
            if primary not in self.strategy_categories:
                primary = 'multi_factor'
            
            return {
                'primary_category': primary,
                'confidence': float(response.get('confidence', 0.5)),
                'secondary_categories': response.get('secondary_categories', []),
                'reasoning': response.get('reasoning', '')
            }
        return self._create_fallback_categorization()
    
    def _parse_insight(self, response: Any, generation: int) -> Dict[str, Any]:
        """Parse insight extraction response."""
        if isinstance(response, dict):
            return {
                'insight_id': response.get('insight_id', f"insight_{generation}_{datetime.now().timestamp()}"),
                'generation': generation,
                'category': response.get('category', 'unknown'),
                'key_finding': response.get('key_finding', ''),
                'evidence': response.get('evidence', ''),
                'actionability': response.get('actionability', ''),
                'confidence': float(response.get('confidence', 0.5)),
                'timestamp': response.get('timestamp', datetime.now().isoformat())
            }
        return self._create_fallback_insight(generation)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _category_to_bits(self, category: str) -> str:
        """
        Convert category name to 8-bit binary string.
        
        Args:
            category: Category name from strategy_categories list
            
        Returns:
            8-bit binary string (e.g., "00000001" for momentum)
        """
        bits = ['0'] * 8
        normalized = str(category or '').lower()
        alias_map = {
            'volatility': 'statistical_arb',
            'volume': 'sentiment_analysis',
            'fundamental': 'multi_factor',
            'hybrid': 'multi_factor',
            'market_microstructure': 'sentiment_analysis',
            'fundamental_value': 'multi_factor',
        }
        normalized = alias_map.get(normalized, normalized)
        try:
            index = self.strategy_categories.index(normalized)
            bits[index] = '1'
        except ValueError:
            # Unknown category, use composite/fallback bucket.
            bits[7] = '1'
        
        return ''.join(bits)
    
    def _create_fallback_hypothesis_evaluation(self) -> Dict[str, Any]:
        """Create fallback hypothesis evaluation on error."""
        return {
            'hypothesis_evaluation_score': 0.5,
            'novelty_score': 0.5,
            'feasibility_score': 0.5,
            'detailed_feedback': 'Fallback evaluation due to parsing error'
        }
    
    def _create_fallback_code_evaluation(self) -> Dict[str, Any]:
        """Create fallback code evaluation on error."""
        return {
            'program_alignment_evaluation_score': 0.5,
            'code_quality_score': 0.5,
            'constraint_compliance': {
                'no_lookahead': True,
                'proper_structure': True,
                'commission_set': True,
                'position_limits': True
            },
            'code_issues': ['Fallback evaluation due to parsing error']
        }
    
    def _create_fallback_results_analysis(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create fallback results analysis on error."""
        return {
            'results_analysis_score': 0.5,
            'performance_insights': ['Fallback analysis due to parsing error'],
            'risk_analysis': {
                'drawdown_severity': 'medium',
                'volatility_level': 'medium',
                'tail_risk': 'medium'
            },
            'improvement_suggestions': ['Review strategy parameters']
        }
    
    def _create_fallback_categorization(self) -> Dict[str, Any]:
        """Create fallback categorization on error."""
        return {
            'primary_category': 'multi_factor',
            'confidence': 0.5,
            'secondary_categories': [],
            'reasoning': 'Fallback categorization due to parsing error'
        }
    
    def _create_fallback_insight(self, generation: int) -> Dict[str, Any]:
        """Create fallback insight on error."""
        return {
            'insight_id': f"fallback_{generation}_{datetime.now().timestamp()}",
            'generation': generation,
            'category': 'unknown',
            'key_finding': 'Fallback insight due to parsing error',
            'evidence': 'N/A',
            'actionability': 'Review strategy manually',
            'confidence': 0.3,
            'timestamp': datetime.now().isoformat()
        }
