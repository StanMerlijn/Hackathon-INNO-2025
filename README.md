# Hackathon-INNO-2025

## Code Verification Prompt Experiment Tester

A tool for testing and comparing different LLM prompts for code verification tasks, based on the Applied-AI 3 Verifiëren casus.

### Features

- Test multiple prompt variations simultaneously
- Built-in test cases covering common code issues:
  - SQL injection vulnerabilities
  - Missing error handling
  - Hardcoded credentials
  - Code quality assessment
- Automatic result collection and analysis
- Performance metrics (tokens, time, cost)

### Setup

1. Create venv:

```bash
uv venv 
```

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file and add the API keys:
```bash
OPENAI_API_KEY=your-API-key
DEEPSEEK_API_KEY=your-API-key
```

### Usage

#### Run the experiment:
```bash
uv run prompt_tester.py
```

This will test 4 different prompt variations across 4 test cases and save results to `code_verification_results.json`.

#### Analyze results:
```bash
uv run  analyze_results.py
```

This will show:
- Performance comparison across prompts
- Token usage and timing

### Prompt Variations
The tester includes 4 prompt strategies:

1. **Basic**: Simple code review request
2. **Structured**: Organized output with sections
3. **Checklist**: Systematic verification checklist
4. **Security-focused**: Specialized security analysis

### Customization

You can easily add:
- New prompt variations in `prompts.json`
- Additional test cases in `test_cases.json`
- Different models (`gpt-4.1-mini`, `deepseek-chat`) 
> **Note** prices are only calculated for the above mentioned models, others wil display amount of tokens only

### Example Output

```
BASIC
--------------------------------------------------------------------------------
Success Rate: 6/6
correct responses: 6/6
Avg Tokens: 530
Avg Time: 17.43s
Total Cost (est): $0.0013

Test Case Results:
  Succes   Correct  Test naam
  ✓        ✓        SQL Injection Vulnerability
  ✓        ✓        Missing Error Handling
  ✓        ✓        Hardcoded Credentials
  ✓        ✓        Good Code Example
  ✓        ✓        Race Condition in Shared State
  ✓        ✓        Insecure Deserialization
```