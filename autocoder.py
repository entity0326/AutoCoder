#!/usr/bin/env python3
"""
AutoCoder AI Agent

An AI agent that can code itself using Ollama for local language model inference.
"""
import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple

from code_generator import CodeGenerator
from code_evaluator import CodeEvaluator
from code_improver import CodeImprover
from utils import (
    setup_environment,
    save_to_file,
    load_from_file,
    save_history,
    check_ollama_availability,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("autocoder")

class AutoCoder:
    """
    AutoCoder is a self-improving AI agent that generates, evaluates,
    and iteratively improves code based on requirements.
    """
    
    def __init__(
        self,
        model: str = "codellama",
        temperature: float = 0.2,
        max_improvement_iterations: int = 3,
        output_dir: str = "output",
        save_history: bool = True,
    ):
        """
        Initialize the AutoCoder agent.
        
        Args:
            model: The Ollama model to use
            temperature: Temperature for code generation
            max_improvement_iterations: Maximum number of improvement iterations
            output_dir: Directory to save outputs
            save_history: Whether to save history of generations and improvements
        """
        self.model = model
        self.temperature = temperature
        self.max_improvement_iterations = max_improvement_iterations
        self.output_dir = output_dir
        self.save_history_flag = save_history
        
        # Initialize components
        self.generator = CodeGenerator(model=model, temperature=temperature)
        self.evaluator = CodeEvaluator(model=model)
        self.improver = CodeImprover(model=model, max_iterations=max_improvement_iterations)
        
        # Setup environment
        setup_environment()
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_solution(self, task: str, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a complete solution for a coding task.
        
        This method orchestrates the entire process:
        1. Generate initial code
        2. Evaluate the code
        3. Improve the code based on evaluation
        4. Return the final solution and its history
        
        Args:
            task: Description of the coding task
            test_cases: Optional list of test cases for validation
            
        Returns:
            Dictionary containing the solution and its history
        """
        logger.info(f"Starting solution generation for task: {task}")
        
        # Step 1: Generate initial code
        initial_code = self.generator.generate_with_context(task)
        logger.info(f"Initial code generated ({len(initial_code.split('\\n'))} lines)")
        
        # Save the initial code
        initial_code_path = os.path.join(self.output_dir, "initial_solution.py")
        save_to_file(initial_code, initial_code_path)
        logger.info(f"Initial solution saved to {initial_code_path}")
        
        # Step 2: Evaluate the code
        evaluation = self.evaluator.evaluate(initial_code, task, test_cases)
        logger.info(f"Initial code evaluation complete. Score: {evaluation.get('overall_score', 'N/A')}")
        
        # Save the evaluation
        evaluation_path = os.path.join(self.output_dir, "initial_evaluation.json")
        with open(evaluation_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Initial evaluation saved to {evaluation_path}")
        
        # Step 3: Improve the code based on evaluation
        improvement_result = self.improver.improve(initial_code, evaluation, task)
        improved_code = improvement_result["improved_code"]
        logger.info(f"Code improvement complete. Performed {improvement_result['iterations_performed']} iterations.")
        
        # Save the improved code
        improved_code_path = os.path.join(self.output_dir, "improved_solution.py")
        save_to_file(improved_code, improved_code_path)
        logger.info(f"Improved solution saved to {improved_code_path}")
        
        # Save the improvement history
        improvement_history_path = os.path.join(self.output_dir, "improvement_history.json")
        with open(improvement_history_path, "w", encoding="utf-8") as f:
            json.dump(improvement_result["improvement_history"], f, indent=2)
        
        # Step 4: Return the final solution and its history
        solution = {
            "task": task,
            "initial_code": initial_code,
            "initial_evaluation": evaluation,
            "improved_code": improved_code,
            "improvement_history": improvement_result["improvement_history"],
            "iterations_performed": improvement_result["iterations_performed"],
            "file_paths": {
                "initial_code": initial_code_path,
                "initial_evaluation": evaluation_path,
                "improved_code": improved_code_path,
                "improvement_history": improvement_history_path,
            }
        }
        
        # Save the full solution history if enabled
        if self.save_history_flag:
            timestamp = self._get_timestamp()
            history_path = f"history/solution_{timestamp}.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(solution, f, indent=2)
            logger.info(f"Solution history saved to {history_path}")
        
        return solution
    
    def self_improve(self, aspect: str = "all") -> Dict[str, Any]:
        """
        Improve the AutoCoder's own codebase.
        
        Args:
            aspect: Which aspect of the codebase to improve ('all', 'generator', 
                    'evaluator', 'improver', or 'main')
            
        Returns:
            Dictionary containing information about the self-improvement
        """
        logger.info(f"Starting self-improvement of AutoCoder codebase, aspect: {aspect}")
        
        # Determine which files to improve
        files_to_improve = []
        
        if aspect == "all" or aspect == "main":
            files_to_improve.append(("autocoder.py", "AutoCoder main module"))
        
        if aspect == "all" or aspect == "generator":
            files_to_improve.append(("code_generator.py", "Code generator module"))
        
        if aspect == "all" or aspect == "evaluator":
            files_to_improve.append(("code_evaluator.py", "Code evaluator module"))
        
        if aspect == "all" or aspect == "improver":
            files_to_improve.append(("code_improver.py", "Code improver module"))
        
        if aspect == "all":
            files_to_improve.append(("utils.py", "Utility functions module"))
        
        # Process each file
        improvements = {}
        
        for file_path, description in files_to_improve:
            logger.info(f"Self-improving {description} ({file_path})")
            
            # Load the current code
            current_code = load_from_file(file_path)
            if not current_code:
                logger.error(f"Could not load file {file_path}")
                continue
            
            # Create task for self-improvement
            task = f"Improve the {description} of the AutoCoder AI agent. " \
                   f"The code should be more efficient, readable, and maintainable. " \
                   f"Add better error handling and documentation where needed."
            
            # Generate improved code
            improvement_prompt = (
                f"Task: {task}\n\n"
                f"Current code:\n```python\n{current_code}\n```\n\n"
                f"Please provide an improved version of this code. Focus on:\n"
                f"1. Making the code more efficient\n"
                f"2. Enhancing readability and maintainability\n"
                f"3. Improving error handling\n"
                f"4. Better documentation\n"
                f"5. Adding new useful features if appropriate\n\n"
                f"Return only the improved code, without any explanations outside of code comments."
            )
            
            system_prompt = (
                "You are an expert Python developer specializing in AI agent development. "
                "Your task is to improve the given code for an AI agent. "
                "Focus on making the code more efficient, readable, and robust. "
                "Return only the improved code without any explanations outside of code comments."
            )
            
            # Generate the improved code
            improved_code = self.generator.generate(improvement_prompt, system_prompt=system_prompt)
            
            # Extract code block if needed
            if "```" in improved_code:
                code_blocks = []
                inside_code_block = False
                current_block = []
                
                for line in improved_code.split("\n"):
                    if line.startswith("```"):
                        if inside_code_block:
                            inside_code_block = False
                            if current_block:
                                code_blocks.append("\n".join(current_block))
                                current_block = []
                        else:
                            inside_code_block = True
                    elif inside_code_block:
                        current_block.append(line)
                
                if code_blocks:
                    improved_code = code_blocks[0]
            
            # Save to a new file with '_improved' suffix
            file_name, file_ext = os.path.splitext(file_path)
            improved_file_path = f"{file_name}_improved{file_ext}"
            save_to_file(improved_code, improved_file_path)
            
            logger.info(f"Improved version of {file_path} saved to {improved_file_path}")
            
            # Add to improvements dictionary
            improvements[file_path] = {
                "original_file": file_path,
                "improved_file": improved_file_path,
                "description": description,
            }
        
        return {
            "aspect": aspect,
            "improvements": improvements,
        }
    
    def _get_timestamp(self) -> str:
        """Get a timestamp string for file naming."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Main function to run the AutoCoder agent."""
    parser = argparse.ArgumentParser(description="AutoCoder AI Agent")
    parser.add_argument("--task", type=str, help="The coding task to solve")
    parser.add_argument("--model", type=str, default="codellama", help="Ollama model to use (default: codellama)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation (default: 0.2)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save outputs (default: output)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum improvement iterations (default: 3)")
    parser.add_argument("--self-improve", type=str, choices=["all", "generator", "evaluator", "improver", "main"], 
                        help="Self-improve the specified component of the agent")
    parser.add_argument("--test-file", type=str, help="JSON file containing test cases for the task")
    parser.add_argument("--no-history", action="store_true", help="Don't save solution history")
    
    args = parser.parse_args()
    
    # Check if Ollama is available
    if not check_ollama_availability():
        logger.error("Ollama is not available. Make sure it is installed and running.")
        logger.error("Install from: https://ollama.ai/")
        logger.error("After installation, pull a coding model: ollama pull codellama")
        sys.exit(1)
    
    # Initialize the agent
    agent = AutoCoder(
        model=args.model,
        temperature=args.temperature,
        max_improvement_iterations=args.max_iterations,
        output_dir=args.output_dir,
        save_history=not args.no_history,
    )
    
    # Handle self-improvement mode
    if args.self_improve:
        result = agent.self_improve(aspect=args.self_improve)
        logger.info(f"Self-improvement completed for aspect: {args.self_improve}")
        
        # Print information about the improvements
        for file_path, info in result["improvements"].items():
            print(f"Improved {file_path} -> {info['improved_file']}")
        
        sys.exit(0)
    
    # Check if a task was provided
    if not args.task:
        parser.print_help()
        sys.exit(1)
    
    # Load test cases if provided
    test_cases = None
    if args.test_file:
        try:
            with open(args.test_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
            logger.info(f"Loaded {len(test_cases)} test cases from {args.test_file}")
        except Exception as e:
            logger.error(f"Error loading test cases: {str(e)}")
            sys.exit(1)
    
    # Generate solution for the task
    solution = agent.generate_solution(args.task, test_cases)
    
    # Print summary
    print("\n" + "="*50)
    print(f"AutoCoder Solution Summary:")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Model used: {args.model}")
    print(f"Initial code: {solution['file_paths']['initial_code']}")
    print(f"Improved code: {solution['file_paths']['improved_code']}")
    print(f"Improvement iterations: {solution['iterations_performed']}")
    print("="*50)


if __name__ == "__main__":
    main() 