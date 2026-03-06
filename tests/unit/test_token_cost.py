"""
Unit tests for token counting and cost estimation
"""
import pytest
from pathlib import Path

from knowledge_engine.utils.token_cost import (
    TokenCostCalculator,
    TokenCount,
    CostEstimate,
    get_token_calculator,
)
from knowledge_engine.core.exceptions import TokenCountError


class TestTokenCostCalculator:
    """Tests for TokenCostCalculator class"""
    
    def test_count_tokens_basic(self):
        """Test basic token counting"""
        calc = TokenCostCalculator()
        result = calc.count_tokens("Hello world", model="gpt-4")
        
        assert isinstance(result, TokenCount)
        assert result.token_count > 0
        assert result.model == "gpt-4"
        assert result.text == "Hello world"
    
    def test_count_tokens_empty(self):
        """Test empty string returns 0 tokens"""
        calc = TokenCostCalculator()
        result = calc.count_tokens("", model="gpt-4")
        
        assert result.token_count == 0
    
    def test_count_tokens_invalid_input(self):
        """Test error handling for invalid input"""
        calc = TokenCostCalculator()
        
        with pytest.raises(TokenCountError):
            calc.count_tokens(123, model="gpt-4")
        
        with pytest.raises(TokenCountError):
            calc.count_tokens(None, model="gpt-4")
    
    def test_estimate_cost_basic(self):
        """Test basic cost estimation"""
        calc = TokenCostCalculator()
        result = calc.estimate_cost(
            input_text="What is 2+2?",
            output_text="4",
            model="gpt-4"
        )
        
        assert isinstance(result, CostEstimate)
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens
        assert result.total_cost > 0
        assert result.model == "gpt-4"
    
    def test_estimate_cost_with_token_counts(self):
        """Test cost estimation with pre-counted tokens"""
        calc = TokenCostCalculator()
        result = calc.estimate_cost(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4"
        )
        
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.total_cost > 0
    
    def test_different_models_different_costs(self):
        """Test cost difference between models"""
        calc = TokenCostCalculator()
        text = "Hello world"
        
        gpt4_result = calc.estimate_cost(input_text=text, model="gpt-4")
        gpt35_result = calc.estimate_cost(input_text=text, model="gpt-3.5-turbo")
        
        # GPT-4 should be more expensive
        assert gpt4_result.total_cost > gpt35_result.total_cost
    
    def test_cost_estimate_format_summary(self):
        """Test cost summary formatting"""
        calc = TokenCostCalculator()
        result = calc.estimate_cost(
            input_text="Test",
            output_text="Response",
            model="gpt-4"
        )
        
        summary = result.format_summary()
        
        assert "gpt-4" in summary
        assert "input" in summary.lower()
        assert "output" in summary.lower()
        assert "total" in summary.lower()
    
    def test_pricing_config_loading(self):
        """Test pricing config loads correctly"""
        calc = TokenCostCalculator()
        
        assert "models" in calc.pricing
        assert "gpt-4" in calc.pricing["models"]
        assert "input" in calc.pricing["models"]["gpt-4"]
        assert "output" in calc.pricing["models"]["gpt-4"]
    
    def test_encoding_cache(self):
        """Test encoding is cached"""
        calc = TokenCostCalculator()
        
        # First call
        calc.count_tokens("Test", model="gpt-4")
        
        # Second call should use cache
        assert "gpt-4" in calc._encoding_cache
    
    def test_factory_function(self):
        """Test get_token_calculator factory"""
        calc1 = get_token_calculator()
        calc2 = get_token_calculator()
        
        # Should return same cached instance
        assert calc1 is calc2


class TestTokenCount:
    """Tests for TokenCount dataclass"""
    
    def test_token_count_str(self):
        """Test TokenCount string representation"""
        tc = TokenCount(text="Hello", token_count=2, model="gpt-4")
        s = str(tc)
        
        assert "tokens=2" in s
        assert "gpt-4" in s


class TestCostEstimate:
    """Tests for CostEstimate dataclass"""
    
    def test_cost_estimate_str(self):
        """Test CostEstimate string representation"""
        ce = CostEstimate(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_cost=0.0003,
            output_cost=0.0003,
            total_cost=0.0006,
            model="gpt-4"
        )
        s = str(ce)
        
        assert "gpt-4" in s
        assert "tokens=15" in s
