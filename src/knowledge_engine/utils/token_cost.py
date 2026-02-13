import tiktoken
from typing import Dict


# Pricing (per 1K tokens)
PRICING = {
    "gpt-3.5-turbo": {"input": 0.002, "cached": 0.0005, "output": 0.008},
    "gpt-4.1": {"input": 0.002, "cached": 0.0005, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "cached": 0.0001, "output": 0.0016},
}


def count_tokens(text: str, model: str) -> int:
    """Count tokens using tiktoken encoding."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        raise ValueError(f"Unsupported model: {model}")

    return len(encoding.encode(text))


def estimate_cost(text: str, model: str) -> Dict[str, float]:
    """Estimate input/output token cost for a given text."""
    if model not in PRICING:
        raise ValueError(f"Model '{model}' not supported")

    tokens = count_tokens(text, model)
    price = PRICING[model]

    input_cost = (tokens / 1000) * price["input"]
    cache_cost = (tokens / 1000) * price["cached"]
    output_cost = (tokens / 1000) * price["output"]

    return {
        "tokens": tokens,
        "input_cost": round(input_cost, 6),
        "cache_cost": round(cache_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + cache_cost + output_cost, 6),
    }
