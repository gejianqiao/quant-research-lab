"""
Research Agent for QuantEvolve Framework.

Generates XML-structured hypotheses with 6 components:
<hypothesis>, <rationale>, <objectives>, <expected_insights>, 
<risks_limitations>, <next_step_ideas>

Uses Qwen3-80B for deep reasoning tasks.
"""

import logging
from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research Agent responsible for generating trading strategy hypotheses.
    
    This agent analyzes parent and cousin strategies, extracts insights from
    the evaluation team, and generates novel hypotheses in XML format that
    guide the coding team in creating new trading strategies.
    
    Attributes:
        model_name: LLM model identifier (default: Qwen3-80B for reasoning)
        insights_repository: List of historical insights for context
    """
    
    def __init__(
        self,
        model_name: str = "Qwen3-Next-80B-A3B-Instruct",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the Research Agent.
        
        Args:
            model_name: LLM model to use for hypothesis generation
            api_key: API key for LLM service
            api_base: Base URL for LLM API endpoint
            temperature: Sampling temperature for generation
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in generated response
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
        self.insights_repository: List[Dict[str, Any]] = []
        logger.info(f"ResearchAgent initialized with model: {model_name}")
    
    def add_insight(self, insight: Dict[str, Any]) -> None:
        """
        Add an insight to the repository for future hypothesis generation.
        
        Args:
            insight: Dictionary containing insight data with keys:
                    - strategy_id: ID of the strategy that generated this insight
                    - category: Strategy category
                    - key_finding: Main finding or observation
                    - metric_impact: Impact on performance metrics
                    - suggestion: Suggested improvement or direction
        """
        self.insights_repository.append(insight)
        logger.debug(f"Added insight to repository. Total insights: {len(self.insights_repository)}")
    
    def curate_insights(self, max_insights: int = 500) -> None:
        """
        Curate the insights repository by removing duplicates and keeping most recent.
        
        Args:
            max_insights: Maximum number of insights to retain
        """
        if len(self.insights_repository) > max_insights:
            # Keep the most recent insights
            self.insights_repository = self.insights_repository[-max_insights:]
            logger.info(f"Curated insights repository to {max_insights} insights")
    
    def _format_parent_cousin_context(
        self,
        parent_strategy: Dict[str, Any],
        cousin_strategies: List[Dict[str, Any]]
    ) -> str:
        """
        Format the parent and cousin strategies context for the prompt.
        
        Args:
            parent_strategy: Dictionary containing parent strategy information
            cousin_strategies: List of cousin strategy dictionaries
            
        Returns:
            Formatted string describing parent and cousin strategies
        """
        context = "## Parent Strategy\n"
        context += f"ID: {parent_strategy.get('id', 'unknown')}\n"
        context += f"Category: {parent_strategy.get('category', 'unknown')}\n"
        context += f"Metrics:\n"
        metrics = parent_strategy.get('metrics', {})
        context += f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}\n"
        context += f"  - Sortino Ratio: {metrics.get('sortino_ratio', 'N/A')}\n"
        context += f"  - Max Drawdown: {metrics.get('max_drawdown', 'N/A')}\n"
        context += f"  - Total Return: {metrics.get('total_return', 'N/A')}\n"
        context += f"  - Information Ratio: {metrics.get('information_ratio', 'N/A')}\n"
        context += f"Combined Score: {parent_strategy.get('score', 'N/A')}\n\n"
        
        context += "## Cousin Strategies\n"
        for i, cousin in enumerate(cousin_strategies, 1):
            context += f"\n### Cousin {i}\n"
            context += f"ID: {cousin.get('id', 'unknown')}\n"
            context += f"Category: {cousin.get('category', 'unknown')}\n"
            cousin_metrics = cousin.get('metrics', {})
            context += f"  - Sharpe Ratio: {cousin_metrics.get('sharpe_ratio', 'N/A')}\n"
            context += f"  - Max Drawdown: {cousin_metrics.get('max_drawdown', 'N/A')}\n"
            context += f"  - Total Return: {cousin_metrics.get('total_return', 'N/A')}\n"
            context += f"Combined Score: {cousin.get('score', 'N/A')}\n"
        
        return context
    
    def _format_insights_context(self) -> str:
        """
        Format historical insights for the prompt.
        
        Returns:
            Formatted string containing curated insights
        """
        if not self.insights_repository:
            return "No historical insights available yet.\n"
        
        context = "## Historical Insights Repository\n"
        context += f"Total insights: {len(self.insights_repository)}\n\n"
        
        # Show most recent 20 insights
        recent_insights = self.insights_repository[-20:]
        for i, insight in enumerate(recent_insights, 1):
            context += f"{i}. [{insight.get('category', 'N/A')}] {insight.get('key_finding', 'N/A')}\n"
            if insight.get('suggestion'):
                context += f"   Suggestion: {insight.get('suggestion')}\n"
        
        return context
    
    def generate_hypothesis(
        self,
        parent_strategy: Dict[str, Any],
        cousin_strategies: List[Dict[str, Any]],
        generation: int,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a new hypothesis based on parent and cousin strategies.
        
        Args:
            parent_strategy: The selected parent strategy dictionary
            cousin_strategies: List of 7 cousin strategy dictionaries
            generation: Current generation number
            market_context: Optional market analysis context from Data Agent
            
        Returns:
            Dictionary containing the generated hypothesis with keys:
            - hypothesis: The core hypothesis statement
            - rationale: Reasoning behind the hypothesis
            - objectives: Specific objectives for the strategy
            - expected_insights: What we expect to learn
            - risks_limitations: Potential risks and limitations
            - next_step_ideas: Ideas for future iterations
            - raw_xml: The raw XML output from the LLM
        """
        # Build the prompt
        parent_cousin_context = self._format_parent_cousin_context(
            parent_strategy, cousin_strategies
        )
        insights_context = self._format_insights_context()
        
        system_prompt = """You are an expert quantitative research analyst specializing in trading strategy development. Your task is to generate novel, testable hypotheses for trading strategies based on analysis of existing strategies and market insights.

Your hypotheses must be:
1. Specific and testable through backtesting
2. Financially grounded in market theory or empirical observation
3. Innovative but not overly complex
4. Clear enough to be translated into executable code

Always output your response in valid XML format with the following structure:
<hypothesis>
    <hypothesis>[Your core hypothesis statement]</hypothesis>
    <rationale>[Detailed reasoning and theoretical foundation]</rationale>
    <objectives>[List of specific, measurable objectives]</objectives>
    <expected_insights>[What we expect to learn from testing this]</expected_insights>
    <risks_limitations>[Potential risks, limitations, and failure modes]</risks_limitations>
    <next_step_ideas>[Ideas for future iterations or variations]</next_step_ideas>
</hypothesis>"""

        user_prompt = f"""Generate a novel trading strategy hypothesis based on the following context:

## Generation Information
Current Generation: {generation}

## Market Context
{market_context if market_context else "Standard equity/futures market analysis applies."}

{parent_cousin_context}

{insights_context}

## Task
Analyze the parent strategy and its cousins to identify:
1. What works well in the parent strategy
2. What limitations or weaknesses exist
3. How cousin strategies differ and what we can learn from them
4. Opportunities for innovation based on historical insights

Generate a hypothesis that:
- Builds on the parent's strengths
- Addresses identified weaknesses
- Incorporates learnings from cousins
- Leverages historical insights where relevant
- Explores a new direction or variation

Output your hypothesis in the required XML format."""

        # Generate response from LLM
        try:
            response = self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_xml=True,
                max_retries=3
            )
            
            # Parse the XML response
            hypothesis_data = self._parse_hypothesis_xml(response)
            hypothesis_data['raw_xml'] = response
            hypothesis_data['generation'] = generation
            
            logger.info(f"Generated hypothesis for generation {generation}")
            return hypothesis_data
            
        except Exception as e:
            logger.error(f"Failed to generate hypothesis: {str(e)}")
            # Return a fallback hypothesis
            return self._create_fallback_hypothesis(parent_strategy, generation)
    
    def _parse_hypothesis_xml(self, xml_string: str) -> Dict[str, Any]:
        """
        Parse the XML hypothesis response into a dictionary.
        
        Args:
            xml_string: Raw XML string from LLM
            
        Returns:
            Dictionary with parsed hypothesis components
        """
        result = {
            'hypothesis': '',
            'rationale': '',
            'objectives': '',
            'expected_insights': '',
            'risks_limitations': '',
            'next_step_ideas': ''
        }
        
        # Define XML tags to extract
        tags = ['hypothesis', 'rationale', 'objectives', 'expected_insights', 
                'risks_limitations', 'next_step_ideas']
        
        for tag in tags:
            # Use regex to extract content between tags
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
            if match:
                result[tag] = match.group(1).strip()
            else:
                logger.warning(f"Could not find <{tag}> tag in hypothesis XML")
                result[tag] = f"[Missing {tag}]"
        
        return result
    
    def _create_fallback_hypothesis(
        self,
        parent_strategy: Dict[str, Any],
        generation: int
    ) -> Dict[str, Any]:
        """
        Create a fallback hypothesis if LLM generation fails.
        
        Args:
            parent_strategy: Parent strategy dictionary
            generation: Current generation number
            
        Returns:
            Dictionary with basic fallback hypothesis
        """
        category = parent_strategy.get('category', 'momentum')
        
        return {
            'hypothesis': f"Enhanced {category} strategy with adaptive parameters based on market volatility.",
            'rationale': "Adaptive parameter adjustment can improve strategy performance across different market regimes by responding to changing volatility conditions.",
            'objectives': "1. Implement volatility-based parameter scaling\n2. Maintain core strategy logic\n3. Improve risk-adjusted returns",
            'expected_insights': "Understanding how adaptive parameters affect performance in different volatility regimes.",
            'risks_limitations': "May underperform in stable markets; increased complexity may lead to overfitting.",
            'next_step_ideas': "Test different volatility lookback periods; explore machine learning-based parameter optimization.",
            'raw_xml': '',
            'generation': generation,
            'is_fallback': True
        }
    
    def execute(
        self,
        parent_strategy: Dict[str, Any],
        cousin_strategies: List[Dict[str, Any]],
        generation: int,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the research agent's main task: generate a hypothesis.
        
        This is the abstract method implementation required by BaseAgent.
        
        Args:
            parent_strategy: The selected parent strategy
            cousin_strategies: List of cousin strategies for context
            generation: Current generation number
            market_context: Optional market analysis context
            
        Returns:
            Dictionary containing the generated hypothesis
        """
        return self.generate_hypothesis(
            parent_strategy=parent_strategy,
            cousin_strategies=cousin_strategies,
            generation=generation,
            market_context=market_context
        )
