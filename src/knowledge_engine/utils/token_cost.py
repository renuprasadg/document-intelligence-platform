"""
Token counting and cost estimation for OpenAI models (Enterprise Edition)
"""
import yaml
import tiktoken
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from functools import lru_cache

from knowledge_engine.core.exceptions import TokenCountError, ConfigurationError
from knowledge_engine.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TokenCount:
    """Result of token counting operation"""
    text: str
    token_count: int
    model: str
    
    def __str__(self) -> str:
        return f"TokenCount(tokens={self.token_count}, model={self.model})"


@dataclass
class CostEstimate:
    """Result of cost estimation operation"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    
    def __str__(self) -> str:
        return (
            f"CostEstimate(model={self.model}, tokens={self.total_tokens}, "
            f"cost=${self.total_cost:.6f})"
        )
    
    def format_summary(self) -> str:
        """Format cost information as human-readable string"""
        return (
            f"Model: {self.model}\n"
            f"Tokens: {self.input_tokens} input + "
            f"{self.output_tokens} output = "
            f"{self.total_tokens} total\n"
            f"Cost: ${self.input_cost:.6f} input + "
            f"${self.output_cost:.6f} output = "
            f"${self.total_cost:.6f} total"
        )


class TokenCostCalculator:
    """
    Token counting and cost estimation for OpenAI models
    
    Enterprise-grade design:
    - Configurable pricing (from YAML)
    - Cached encodings (performance)
    - Structured results (dataclasses)
    - Easy to test and extend
    """
    
    def __init__(self, pricing_config_path: Optional[str] = None):
        """
        Initialize calculator
        
        Args:
            pricing_config_path: Path to pricing.yaml (optional)
        """
        self.pricing_config_path = pricing_config_path or "./configs/pricing.yaml"
        self._pricing: Optional[Dict] = None
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}
    
    @property
    def pricing(self) -> Dict:
        """Load pricing config (lazy, cached)"""
        if self._pricing is None:
            self._pricing = self._load_pricing_config()
        return self._pricing
    
    def _load_pricing_config(self) -> Dict:
        """Load pricing from YAML config"""
        config_path = Path(self.pricing_config_path)
        
        if not config_path.exists():
            logger.warning(
                f"Pricing config not found at {config_path}, using fallback pricing"
            )
            return self._get_fallback_pricing()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded pricing config from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading pricing config: {e}")
            return self._get_fallback_pricing()
    
    def _get_fallback_pricing(self) -> Dict:
        """Fallback pricing if config file not available"""
        return {
            "models": {
                "gpt-4": {"input": 30.00, "output": 60.00},
                "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
                "text-embedding-3-small": {"input": 0.02, "output": 0.00},
            },
            "fallback_pricing": {"input": 30.00, "output": 60.00}
        }
    
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get tiktoken encoding for model (cached)"""
        if model not in self._encoding_cache:
            try:
                self._encoding_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.warning(
                    f"Encoding not found for {model}, using cl100k_base"
                )
                self._encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoding_cache[model]
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> TokenCount:
        """
        Count tokens in text for a specific model
        
        Args:
            text: Input text
            model: Model name (default: "gpt-4")
            
        Returns:
            TokenCount dataclass with results
            
        Raises:
            TokenCountError: If token counting fails
            
        Examples:
            >>> calc = TokenCostCalculator()
            >>> result = calc.count_tokens("Hello world", model="gpt-4")
            >>> print(result.token_count)
            2
        """
        if not isinstance(text, str):
            raise TokenCountError(f"Expected str, got {type(text).__name__}")
        
        if not text:
            return TokenCount(text=text, token_count=0, model=model)
        
        try:
            encoding = self._get_encoding(model)
            tokens = encoding.encode(text)
            token_count = len(tokens)
            
            return TokenCount(
                text=text,
                token_count=token_count,
                model=model
            )
        
        except Exception as e:
            raise TokenCountError(f"Error counting tokens: {e}")
    
    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a specific model"""
        models = self.pricing.get("models", {})
        
        if model in models:
            return models[model]
        
        # Try without version suffix (e.g., gpt-4-0613 -> gpt-4)
        base_model = model.split('-')[0:2]  # ["gpt", "4"]
        base_model_name = '-'.join(base_model)
        
        if base_model_name in models:
            logger.debug(f"Using pricing for {base_model_name} for model {model}")
            return models[base_model_name]
        
        # Use fallback
        logger.warning(f"Pricing not found for {model}, using fallback")
        return self.pricing.get("fallback_pricing", {"input": 30.00, "output": 60.00})
    
    def estimate_cost(
        self,
        input_text: str = "",
        output_text: str = "",
        model: str = "gpt-4",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> CostEstimate:
        """
        Estimate cost for OpenAI API call
        
        Args:
            input_text: Input text (prompt)
            output_text: Output text (completion)
            model: Model name
            input_tokens: Pre-counted input tokens (optional)
            output_tokens: Pre-counted output tokens (optional)
            
        Returns:
            CostEstimate dataclass with detailed breakdown
            
        Examples:
            >>> calc = TokenCostCalculator()
            >>> result = calc.estimate_cost(
            ...     input_text="What is 2+2?",
            ...     output_text="4",
            ...     model="gpt-4"
            ... )
            >>> result.total_cost  # doctest: +SKIP
            0.00042
        """
        # Count tokens if not provided
        if input_tokens is None and input_text:
            input_count = self.count_tokens(input_text, model)
            input_tokens = input_count.token_count
        elif input_tokens is None:
            input_tokens = 0
        
        if output_tokens is None and output_text:
            output_count = self.count_tokens(output_text, model)
            output_tokens = output_count.token_count
        elif output_tokens is None:
            output_tokens = 0
        
        total_tokens = input_tokens + output_tokens
        
        # Get pricing for model
        pricing = self._get_model_pricing(model)
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            model=model,
        )


# Convenience factory function
@lru_cache()
def get_token_calculator(pricing_config_path: Optional[str] = None) -> TokenCostCalculator:
    """
    Get cached TokenCostCalculator instance
    
    Args:
        pricing_config_path: Path to pricing config
        
    Returns:
        TokenCostCalculator instance
    """
    return TokenCostCalculator(pricing_config_path)
