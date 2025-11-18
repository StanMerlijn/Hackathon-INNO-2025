import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def evaluate_response_match(
    client: OpenAI, expected: str, actual_response: str, model: str = "gpt-4o-mini"
) -> bool:
    """
    Use an LLM to evaluate if the AI response matches the expected criteria.

    Args:
        expected: Expected criteria (e.g., "Should identify SQL injection vulnerability")
        actual_response: The actual AI response text
        model: The model to use for evaluation

    Returns:
        bool: True if the response matches the expected criteria
    """

    judge_prompt = f"""You are an evaluator for AI code review responses.

    Expected Criteria: {expected}

    AI Response:
    {actual_response}

    Does the AI response adequately address the expected criteria? Consider:
    - Does it identify the issue mentioned in the expected criteria?
    - Does it provide relevant explanations or solutions?
    - Does it miss the main point of what was expected?

    Respond with ONLY "true" or "false" (lowercase, no additional text)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise evaluator. Respond only with 'true' or 'false'.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        result = response.choices[0].message.content.strip().lower()
        return result == "true"

    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return False


def load_results(filename: str = "code_verification_results.json") -> list:
    """Load experiment results from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def analyze_prompt_performance(results: list):
    """Analyze and compare prompt variations."""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("Prompt Performance Analysis")
    print("=" * 80)

    for prompt_result in results:
        prompt_name = prompt_result["prompt_name"]
        test_results = prompt_result["test_results"]

        print(f"\n{prompt_name.upper()}")
        print("-" * 80)

        tokens_output = 0
        tokens_input = 0
        total_time = 0
        success_count = 0
        num_correct = 0 

        for test in test_results:
            result = test["result"]
            expected = test["expected"]

            if result["success"]:
                tokens_output += result["tokens_output"]
                tokens_input += result["tokens_input"]
                total_time += result["time_seconds"]
                success_count += 1

                if "correct" not in test:
                    test["correct"] = evaluate_response_match(
                        client, expected, result["response"]
                    )
                
                if test["correct"]:
                    num_correct += 1

        avg_tokens = (tokens_input + tokens_output) / success_count if success_count > 0 else 0
        avg_time = total_time / success_count if success_count > 0 else 0
        # OpenAI gpt-4.1-mini
        output_cost = (tokens_output / 1e6) * 1.6
        input_cost = (tokens_input / 1e6) * 0.4
        
        # Deepseek-chat 
        # output_cost = (tokens_output / 1e6) * 0.42
        # input_cost = (tokens_input / 1e6) * 0.28
        
        print(f"Success Rate: {success_count}/{len(test_results)}")
        print(f"correct responses: {num_correct}/{len(test_results)}")
        print(f"Avg Tokens: {avg_tokens:.0f}")
        print(f"Avg Time: {avg_time:.2f}s")
        print(
            f"Total Cost (est): ${output_cost + input_cost:.4f}"
        )  # Rough estimate

        print("\nTest Case Results:")
        print(f"  {'Succes':<8} {'Correct':<8} Test naam")

        for test in test_results:
            test_name = test["test_case"]
            expected = test["expected"]
            success = "✓" if test["result"]["success"] else "✗"

            # Check if response matches expected criteria
            matches = "✓" if test["result"]["success"] and test.get("correct", False) else "✗"

            print(f"  {success:<8} {matches:<8} {test_name}")


def compare_responses(results: list, test_case_name: str):
    """Compare how different prompts handled the same test case."""
    print(f"\n\n🔍 Comparing Responses for: {test_case_name}")
    print("=" * 80)

    for prompt_result in results:
        prompt_name = prompt_result["prompt_name"]

        # Find the specific test case
        for test in prompt_result["test_results"]:
            if test["test_case"] == test_case_name:
                print(f"\n{prompt_name.upper()}:")
                print("-" * 80)
                if test["result"]["success"]:
                    print(test["result"]["response"])
                else:
                    print(f"Error: {test['result']['error']}")
                break


def main():
    """Run the analysis."""
    try:
        results = load_results()

        # Overall performance analysis
        analyze_prompt_performance(results)

    except FileNotFoundError:
        print("❌ Results file not found. Please run prompt_tester.py first.")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


if __name__ == "__main__":
    main()
