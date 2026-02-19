"""
Base Agent Module for QuantEvolve

This module defines the abstract base class for all LLM agents in the QuantEvolve framework.
It provides common functionality for LLM interaction, prompt handling, and response parsing.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents in the QuantEvolve system.
    
    This class provides the common interface for interacting with large language models,
    including initialization, prompt formatting, response generation, and output parsing.
    All specialized agents (DataAgent, ResearchAgent, CodingTeam, EvaluationTeam) 
    should inherit from this base class.
    
    Attributes:
        model_name (str): Name of the LLM model to use (e.g., 'Qwen3-30B-A3B-Instruct-2507')
        api_key (str): API key for LLM service authentication
        api_base (str): Base URL for LLM API endpoint
        temperature (float): Sampling temperature for generation (0.0-1.0)
        top_p (float): Nucleus sampling parameter (0.0-1.0)
        max_tokens (int): Maximum number of tokens to generate
        client: LLM client instance (vllm, openai, or other compatible client)
    
    Example:
        >>> class MyAgent(BaseAgent):
        ...     def generate_hypothesis(self, context: str) -> str:
        ...         prompt = self._format_prompt(context)
        ...         return self.generate_response(prompt)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen3-30B-A3B-Instruct-2507",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the base agent with LLM configuration.
        
        Args:
            model_name: Name/identifier of the LLM model to use
            api_key: API key for authentication (reads from env if not provided)
            api_base: Base URL for API endpoint (reads from env if not provided)
            temperature: Sampling temperature for generation diversity
            top_p: Nucleus sampling parameter for token selection
            max_tokens: Maximum tokens to generate in response
            **kwargs: Additional model-specific parameters
            
        Raises:
            ValueError: If required configuration is missing
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.additional_params = kwargs
        
        self.client = None
        self._initialize_llm_client()
        
        logger.info(f"BaseAgent initialized with model: {model_name}")
    
    def _initialize_llm_client(self):
        """
        Initialize the LLM client based on available libraries and configuration.
        
        Attempts to initialize in the following order:
        1. OpenAI-compatible client (for vllm, OpenAI API, etc.)
        2. Direct vllm client (for local inference)
        3. Mock client (for testing without LLM)
        
        Raises:
            ImportError: If no compatible LLM library is available
        """
        # Try OpenAI-compatible client first (works with vllm, OpenAI, etc.)
        try:
            from openai import OpenAI
            
            self.client = OpenAI(
                api_key=self.api_key if self.api_key else "not-needed",
                base_url=self.api_base
            )
            self.client_type = "openai"
            logger.debug("Initialized OpenAI-compatible client")
            return
        except ImportError:
            pass
        
        # Try vllm direct client
        try:
            from vllm import LLM
            
            self.client = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9
            )
            self.client_type = "vllm"
            logger.debug("Initialized vllm client")
            return
        except ImportError:
            pass
        
        # Fallback to mock client for testing
        logger.warning("No LLM client available. Using mock client for testing.")
        self.client_type = "mock"
    
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = False,
        parse_xml: bool = False,
        max_retries: int = 3
    ) -> Union[str, Dict, List]:
        """
        Generate a response from the LLM given system and user prompts.
        
        Args:
            system_prompt: System instruction prompt defining agent behavior
            user_prompt: User input prompt with specific task context
            parse_json: If True, attempt to parse response as JSON
            parse_xml: If True, attempt to parse response as XML structure
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            Response content as string, dict (if parse_json), or parsed XML structure
            
        Raises:
            ValueError: If response parsing fails after max_retries
            RuntimeError: If LLM client is not properly initialized
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                if self.client_type == "openai":
                    response = self._generate_openai(messages)
                elif self.client_type == "vllm":
                    response = self._generate_vllm(user_prompt)
                elif self.client_type == "mock":
                    response = self._generate_mock(system_prompt, user_prompt)
                else:
                    raise RuntimeError(f"Unknown client type: {self.client_type}")
                
                # Parse response if requested
                if parse_json:
                    return self._parse_json_response(response)
                elif parse_xml:
                    return self._parse_xml_response(response)
                else:
                    return response.strip()
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate response after {max_retries} attempts: {str(e)}")
        
        # Should not reach here, but just in case
        raise RuntimeError("Unexpected error in response generation")
    
    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using OpenAI-compatible client.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            **self.additional_params
        )
        return response.choices[0].message.content
    
    def _generate_vllm(self, prompt: str) -> str:
        """
        Generate response using vllm direct client.
        
        Args:
            prompt: Full prompt text (system + user combined)
            
        Returns:
            Generated response text
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        outputs = self.client.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_mock(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate mock response for testing without LLM.
        
        Args:
            system_prompt: System instruction prompt
            user_prompt: User input prompt
            
        Returns:
            Mock response text
        """
        logger.warning("Using mock response generator")
        
        # Return a placeholder based on the agent type
        if "hypothesis" in user_prompt.lower():
            return "<hypothesis>Mock hypothesis for testing</hypothesis>"
        elif "code" in user_prompt.lower() or "python" in user_prompt.lower():
            return "def initialize(context):\n    pass\n\ndef handle_data(context, data):\n    pass"
        elif "json" in user_prompt.lower() or "score" in user_prompt.lower():
            return '{"score": 0.5, "insights": []}'
        else:
            return "Mock response for testing purposes."
    
    def _parse_json_response(self, response: str) -> Union[Dict, List]:
        """
        Parse JSON from LLM response.
        
        Args:
            response: Raw response string potentially containing JSON
            
        Returns:
            Parsed JSON as dict or list
            
        Raises:
            ValueError: If JSON parsing fails
        """
        # Try to extract JSON from response (may be wrapped in markdown)
        import re
        
        # Look for JSON block in markdown
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object/array directly
            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.debug(f"Response content: {response}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
    
    def _parse_xml_response(self, response: str) -> Dict[str, str]:
        """
        Parse XML structure from LLM response.
        
        Extracts all XML tags and their content into a dictionary.
        
        Args:
            response: Raw response string containing XML tags
            
        Returns:
            Dictionary mapping tag names to their content
            
        Raises:
            ValueError: If XML parsing fails
        """
        import re
        
        # Extract all XML tags and their content
        xml_pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(xml_pattern, response, re.DOTALL)
        
        if not matches:
            logger.warning("No XML tags found in response")
            raise ValueError("Response does not contain expected XML structure")
        
        result = {}
        for tag_name, content in matches:
            result[tag_name] = content.strip()
        
        return result
    
    def _format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with provided variables.
        
        Args:
            template: Prompt template string with {variable} placeholders
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template: {str(e)}")
            raise
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the agent's primary task.
        
        This method must be implemented by all subclasses to define
        the specific behavior of each agent type.
        
        Args:
            *args: Positional arguments specific to the task
            **kwargs: Keyword arguments specific to the task
            
        Returns:
            Task-specific result
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dictionary containing model name, client type, and parameters
        """
        return {
            "model_name": self.model_name,
            "client_type": self.client_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }
    
    def set_generation_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update generation parameters dynamically.
        
        Args:
            temperature: New temperature value (0.0-1.0)
            top_p: New top_p value (0.0-1.0)
            max_tokens: New maximum token count
        """
        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, top_p))
        if max_tokens is not None:
            self.max_tokens = max(1, max_tokens)
        
        logger.debug(f"Updated generation params: temp={self.temperature}, top_p={self.top_p}, max_tokens={self.max_tokens}")
