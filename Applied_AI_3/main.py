from prompt_tester import run_tests
from analyze_results import analyze_results

if __name__ == "__main__":
    model_clients = ["openAI", "deepseek"]
    run_tests(model_clients)
    analyze_results(model_clients)    

