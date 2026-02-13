import argparse
from knowledge_engine.utils.token_cost import estimate_cost

def main():
    parser = argparse.ArgumentParser(description="Estimate token cost")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name")

    args = parser.parse_args()

    result = estimate_cost(args.text, args.model)

    print("\nToken Cost Estimate")
    print("-------------------")
    for k, v in result.items():
        print(f"{k:12}: {v}")

if __name__ == "__main__":
    main()
