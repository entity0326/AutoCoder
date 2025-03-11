#!/usr/bin/env python3
"""
Example script demonstrating how to use AutoCoder.
"""
import os
import sys
import argparse
from autocoder import AutoCoder

def main():
    """Run a simple example of AutoCoder."""
    parser = argparse.ArgumentParser(description="AutoCoder Example")
    parser.add_argument("--model", type=str, default="codellama", help="Ollama model to use (default: codellama)")
    args = parser.parse_args()
    
    # Create the AutoCoder agent
    agent = AutoCoder(
        model=args.model,
        temperature=0.2,
        max_improvement_iterations=2,
        output_dir="example_output",
    )
    
    # Define a simple coding task
    task = """
    Create a Python function that calculates the Fibonacci sequence up to n terms.
    The function should:
    1. Take a single integer parameter 'n'
    2. Return a list containing the first n Fibonacci numbers
    3. Handle edge cases (n <= 0)
    4. Include proper documentation
    5. Be efficient for large values of n using memoization
    """
    
    # Create simple test cases
    test_cases = [
        {
            "name": "Test with n=1",
            "input": "1",
            "expected_output": "[0]"
        },
        {
            "name": "Test with n=5",
            "input": "5",
            "expected_output": "[0, 1, 1, 2, 3]"
        },
        {
            "name": "Test with n=10",
            "input": "10",
            "expected_output": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
        },
        {
            "name": "Test with n=0",
            "input": "0",
            "expected_output": "[]"
        },
        {
            "name": "Test with negative n",
            "input": "-5",
            "expected_output": "[]"
        }
    ]
    
    print("=" * 50)
    print("AutoCoder Example: Fibonacci Sequence Generator")
    print("=" * 50)
    print(f"Task: {task.strip()}")
    print(f"Model: {args.model}")
    print("=" * 50)
    
    # Generate the solution
    solution = agent.generate_solution(task, test_cases)
    
    # Print summary
    print("\nSolution generated successfully!")
    print(f"Initial code: {solution['file_paths']['initial_code']}")
    print(f"Improved code: {solution['file_paths']['improved_code']}")
    print(f"Improvement iterations: {solution['iterations_performed']}")
    print("\nFinal code:")
    print("-" * 50)
    
    # Print the final improved code
    with open(solution['file_paths']['improved_code'], 'r') as f:
        print(f.read())
    
    print("-" * 50)
    
    # Execute the final solution
    print("\nExecuting the solution...")
    print("-" * 50)
    
    try:
        sys.path.append(os.path.dirname(solution['file_paths']['improved_code']))
        exec(open(solution['file_paths']['improved_code']).read())
        
        # Try to call the fibonacci function with n=10
        import importlib.util
        spec = importlib.util.spec_from_file_location("solution", solution['file_paths']['improved_code'])
        solution_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_module)
        
        # Try to find a function that might be the fibonacci function
        fib_func = None
        for name in dir(solution_module):
            if name.lower().startswith(('fib', 'fibonacci', 'generate_fibonacci')):
                fib_func = getattr(solution_module, name)
                break
        
        if fib_func and callable(fib_func):
            result = fib_func(10)
            print(f"Result of fibonacci(10): {result}")
        else:
            print("Could not find the fibonacci function in the generated solution.")
            
    except Exception as e:
        print(f"Error executing solution: {str(e)}")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 