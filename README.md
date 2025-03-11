# AutoCoder AI Agent

An AI agent that can code itself using Ollama for local language model inference.

## Overview

AutoCoder is a self-improving AI coding agent that can:
1. Generate code based on requirements
2. Evaluate its own code quality
3. Iteratively improve its implementations
4. Learn from its previous coding attempts
5. Improve its own codebase through self-modification

## Requirements

- Python 3.8+
- Ollama with a code-capable model installed (such as `codellama`)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install Ollama from [https://ollama.com/](https://ollama.com/)
3. Pull a coding-capable model: `ollama pull llama3.1:8b`
   *_Refer to https://github.com/ollama/ollama for specific models_
5. Install Python dependencies: `pip install -r requirements.txt`

## Usage

### Basic Usage

```bash
python autocoder.py --task "Create a function that calculates the factorial of a number"
```

### Advanced Options

```bash
python autocoder.py --task "Your coding task description" --model "llama3.1:8b" --temperature 0.2 --max-iterations 3 --output-dir "output" --test-file "your_test_cases.json"
```

### Self-Improvement Mode

AutoCoder can improve its own codebase:

```bash
python autocoder.py --self-improve all
```

You can also target specific components:

```bash
python autocoder.py --self-improve generator
```

Options include: `all`, `generator`, `evaluator`, `improver`, or `main`.

### Run the Example

Try the included example that generates a Fibonacci sequence function:

```bash
python example.py
```

## How It Works

### 1. Code Generation

The agent first uses Ollama's language model to generate an initial solution based on the provided task description. It leverages powerful coding-capable models like CodeLlama to create code that matches the requirements.

### 2. Code Evaluation

Once the initial code is generated, the agent evaluates it across multiple dimensions:
- Syntax checking
- Style analysis
- Code complexity assessment 
- Execution testing
- AI-powered code review

### 3. Code Improvement

Based on the evaluation results, the agent iteratively improves the code:
- Identifies specific issues to fix
- Generates improved versions
- Re-evaluates the improvements
- Continues until reaching a quality threshold or max iteration count

### 4. Self-Improvement

The agent can analyze and improve its own codebase using the same capabilities:
- Analyzes its own modules
- Generates improved versions
- Saves improved versions for review/adoption

## Project Structure

- `autocoder.py`: Main agent script
- `code_generator.py`: Handles code generation using Ollama
- `code_evaluator.py`: Evaluates code quality and correctness
- `code_improver.py`: Improves code based on evaluation
- `utils.py`: Utility functions for the agent
- `example.py`: Example script demonstrating AutoCoder in action
- `example_test_cases.json`: Example test cases format

## Test Cases Format

Test cases are specified in JSON format. Here's an example from `example_test_cases.json`:

```json
[
  {
    "name": "Test basic addition",
    "input": "5 7",
    "expected_output": "12"
  },
  {
    "name": "Test negative numbers",
    "input": "-3 -8",
    "expected_output": "-11"
  }
]
```

## Output Structure

Generated outputs are saved to the specified output directory (default: `output/`):
- `initial_solution.py`: The first generated code
- `improved_solution.py`: The final improved code
- `initial_evaluation.json`: Evaluation of the initial solution
- `improvement_history.json`: History of all improvement iterations

## Extending the Agent

You can extend AutoCoder by:
1. Adding new evaluation metrics to `code_evaluator.py`
2. Enhancing the improvement strategies in `code_improver.py`
3. Supporting additional programming languages
4. Adding more sophisticated test frameworks
5. Integrating with development tools and environments

## Limitations

- Requires Ollama to be installed and running
- Quality of generated code depends on the underlying model
- Currently focused on Python code generation
- Execution of generated code has safety limitations

## Contributing

Contributions are welcome! Please feel free to submit a pull request. 
