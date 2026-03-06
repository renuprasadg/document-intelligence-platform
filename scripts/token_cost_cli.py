#!/usr/bin/env python3
"""
CLI tool for token counting and cost estimation

Usage:
    python -m scripts.token_cost_cli "Your text here"
    python -m scripts.token_cost_cli --file input.txt
    python -m scripts.token_cost_cli --list-models
"""
import argparse
from pathlib import Path

# No sys.path hacks! Rely on editable install
from knowledge_engine.utils.token_cost import get_token_calculator


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens and estimate costs for OpenAI API calls"
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or use --file)"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Read text from file"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gpt-4",
        help="Model name (default: gpt-4)"
    )
    
    parser.add_argument(
        "--output",
        help="Expected output text (for cost estimation)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and pricing"
    )
    
    parser.add_argument(
        "--pricing-config",
        help="Path to pricing config YAML"
    )
    
    args = parser.parse_args()
    
    # Get calculator
    calc = get_token_calculator(args.pricing_config)
    
    # List models
    if args.list_models:
        print("\nAvailable Models and Pricing (per 1M tokens):")
        print("=" * 70)
        for model_name, pricing in calc.pricing.get("models", {}).items():
            print(f"\n{model_name}:")
            print(f"  Input:  ${pricing['input']:.2f}")
            print(f"  Output: ${pricing['output']:.2f}")
        return
    
    # Get input text
    if args.file:
        try:
            text = Path(args.file).read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    elif args.text:
        text = args.text
    else:
        print("Error: Provide text or use --file")
        parser.print_help()
        return 1
    
    # Estimate cost
    try:
        result = calc.estimate_cost(
            input_text=text,
            output_text=args.output or "",
            model=args.model
        )
        
        print("\n" + result.format_summary())
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
