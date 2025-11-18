"""
Prompt Experiment Tester for Code Verification
Based on Applied-AI 3 Verifiëren casus

This tool tests different prompt variations for code verification tasks.
"""

import os
from openai import OpenAI
from typing import List, Dict, Any
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

ROOT_PATH = Path().cwd()


class PromptExperiment:
    """Test different prompts for code verification tasks."""

    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.results = []

    def run_prompt(
        self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """Execute a single prompt and return the response."""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # Deterministic for testing
            )
            elapsed_time = time.time() - start_time

            return {
                "response": response.choices[0].message.content,
                "model": model,
                "tokens_input": response.usage.prompt_tokens,
                "tokens_output": response.usage.completion_tokens,
                "time_seconds": elapsed_time,
                "success": True,
            }
        except Exception as e:
            return {"response": None, "error": str(e), "success": False}

    def test_prompt_variation(
        self,
        prompt_name: str,
        system_prompt: str,
        test_cases: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """Test a prompt variation across multiple test cases."""
        print(f"\n{'=' * 60}")
        print(f"Testing: {prompt_name}")
        print(f"{'=' * 60}")

        results = {
            "prompt_name": prompt_name,
            "system_prompt": system_prompt,
            "model": model,
            "test_results": [],
        }

        for idx, test_case in enumerate(test_cases):
            print(f"\nTest Case {idx + 1}: {test_case['name']}")

            result = self.run_prompt(
                system_prompt=system_prompt,
                user_prompt=test_case["user_prompt"],
                model=model,
            )

            test_result = {
                "test_case": test_case["name"],
                "expected": test_case.get("expected"),
                "result": result,
            }

            results["test_results"].append(test_result)

            if result["success"]:
                print(
                    f"Response received ({result['tokens_output']} tokens, {result['time_seconds']:.2f}s)"
                )
                print(f"Response preview: {result['response'][:200]}...")
            else:
                print(f"✗ Error: {result['error']}")

        self.results.append(results)
        return results

    def save_results(self, filename: str = "experiment_results.json"):
        """Save all experiment results to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n\nResults saved to {filename}")


def main():
    """Run the prompt experiment."""
    print("Code Verification Prompt Experiment Tester")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not found in environment variables")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return

    # Load test cases and prompts from JSON files
    try:
        with open(ROOT_PATH / "data/test_cases.json", "r") as f:
            test_cases = json.load(f)
        with open(ROOT_PATH / "data/prompts.json", "r") as f:
            prompt_variations = json.load(f)
    except FileNotFoundError as e:
        print(f"\nError: Could not find data files - {e}")
        print("Make sure data/test_cases.json and data/prompts.json exist")
        return
    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON in data files - {e}")
        return

    # Initialize experiment
    experiment = PromptExperiment()

    # Test each prompt variation
    for prompt_name, system_prompt in prompt_variations.items():
        experiment.test_prompt_variation(
            prompt_name=prompt_name,
            system_prompt=system_prompt,
            test_cases=test_cases,
        model="deepseek-chat",
        # model="gpt-4.1-mini",
        )

    # Save results
    experiment.save_results("code_verification_results-deepseek.json")

    print("\n\nExperiment complete!")
    print(f"Tested {len(prompt_variations)} prompt variations")
    print(f"Across {len(test_cases)} test cases")


if __name__ == "__main__":
    main()
