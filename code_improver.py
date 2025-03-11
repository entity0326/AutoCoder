"""
Code improver module for the AutoCoder AI agent.
This module improves code based on evaluation and feedback.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests

from utils import print_colored_diff

logger = logging.getLogger("autocoder.improver")

class CodeImprover:
    """Class for improving code based on feedback."""
    
    def __init__(self, model: str = "codellama", max_iterations: int = 3):
        """
        Initialize the code improver.
        
        Args:
            model: The Ollama model to use for code improvement
            max_iterations: Maximum number of improvement iterations
        """
        self.model = model
        self.max_iterations = max_iterations
        self.base_url = "http://localhost:11434/api"
    
    def improve(self, original_code: str, evaluation: Dict[str, Any], task: str) -> Dict[str, Any]:
        """
        Improve code based on evaluation results.
        
        Args:
            original_code: The original code to improve
            evaluation: Evaluation results containing feedback
            task: The original task description
            
        Returns:
            Dictionary containing improvement results
        """
        logger.info("Starting code improvement process...")
        
        current_code = original_code
        improvement_history = [{
            "iteration": 0,
            "code": original_code,
            "evaluation": evaluation,
            "improvements_made": []
        }]
        
        # Extract issues to fix from the evaluation
        issues_to_fix = self._extract_issues(evaluation)
        
        # Iteratively improve the code
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Improvement iteration {iteration}/{self.max_iterations}")
            
            if not issues_to_fix:
                logger.info("No issues to fix, code is already optimal")
                break
            
            # Generate improved code
            improved_code, improvements_made = self._generate_improved_code(
                current_code, issues_to_fix, task, iteration
            )
            
            # Check if code actually changed
            if improved_code == current_code:
                logger.info("No changes made in this iteration, stopping")
                break
            
            # Display diff between iterations
            logger.info(f"Improvements made in iteration {iteration}:")
            for improvement in improvements_made:
                logger.info(f"- {improvement}")
            
            # Print colored diff for visual comparison
            print_colored_diff(current_code, improved_code)
            
            # Update current code
            current_code = improved_code
            
            # Re-evaluate the improved code
            # Note: In a full implementation, you would call the evaluator here
            # For now, we'll just add a placeholder for the next evaluation
            next_evaluation = {
                "placeholder": "In a complete implementation, the code would be re-evaluated here"
            }
            
            # Record this iteration
            improvement_history.append({
                "iteration": iteration,
                "code": improved_code,
                "evaluation": next_evaluation,
                "improvements_made": improvements_made
            })
            
            # Update issues to fix
            issues_to_fix = self._extract_issues(next_evaluation)
        
        # Return the improvement results
        return {
            "original_code": original_code,
            "improved_code": current_code,
            "improvement_history": improvement_history,
            "iterations_performed": len(improvement_history) - 1
        }
    
    def _extract_issues(self, evaluation: Dict[str, Any]) -> List[str]:
        """Extract issues to fix from the evaluation results."""
        issues = []
        
        # Extract syntax issues
        if "syntax_check" in evaluation and not evaluation["syntax_check"].get("valid", True):
            issues.append(f"Fix syntax error: {evaluation['syntax_check'].get('error', 'Unknown syntax error')}")
        
        # Extract style issues
        if "style_check" in evaluation and "issues" in evaluation["style_check"]:
            issues.extend(evaluation["style_check"]["issues"])
        
        # Extract issues from AI feedback
        if "ai_feedback" in evaluation and isinstance(evaluation["ai_feedback"], dict):
            ai_feedback = evaluation["ai_feedback"]
            
            # Add suggestions from AI feedback
            if "suggestions" in ai_feedback and isinstance(ai_feedback["suggestions"], list):
                issues.extend(ai_feedback["suggestions"])
            
            # Add low-scoring areas as issues
            for aspect in ["correctness", "efficiency", "readability", "error_handling"]:
                if aspect in ai_feedback and isinstance(ai_feedback[aspect], dict):
                    score = ai_feedback[aspect].get("score", 10)
                    explanation = ai_feedback[aspect].get("explanation", "")
                    
                    if score < 7:  # Only consider as an issue if score is low
                        issues.append(f"Improve {aspect}: {explanation}")
        
        # Extract issues from execution
        if "execution" in evaluation and not evaluation["execution"].get("success", True):
            error = evaluation["execution"].get("error", "Unknown execution error")
            issues.append(f"Fix execution error: {error}")
        
        # Extract issues from test results
        if "test_results" in evaluation and evaluation["test_results"]:
            for i, test_result in enumerate(evaluation["test_results"]):
                if not test_result.get("passed", False):
                    test_case = test_result.get("test_case", {})
                    test_name = test_case.get("name", f"Test case {i+1}")
                    
                    if "error" in test_result and test_result["error"]:
                        issues.append(f"Fix error in {test_name}: {test_result['error']}")
                    else:
                        expected = test_case.get("expected_output", "unknown expected output")
                        actual = test_result.get("output", "unknown actual output")
                        issues.append(f"Fix {test_name}: Expected '{expected}', got '{actual}'")
        
        return issues
    
    def _generate_improved_code(self, code: str, issues: List[str], task: str, iteration: int) -> Tuple[str, List[str]]:
        """
        Generate improved code using Ollama based on the issues.
        
        Args:
            code: The current code to improve
            issues: List of issues to address
            task: The original task description
            iteration: The current improvement iteration
            
        Returns:
            Tuple of (improved code, list of improvements made)
        """
        # Prepare the prompt for Ollama
        prompt = (
            f"Task: {task}\n\n"
            f"Current Code (Iteration {iteration-1}):\n```\n{code}\n```\n\n"
            f"Issues to fix:\n"
        )
        
        for i, issue in enumerate(issues):
            prompt += f"{i+1}. {issue}\n"
        
        prompt += (
            f"\nYour task is to improve the code to fix these issues. "
            f"The improved code should still solve the original task correctly. "
            f"Return ONLY the improved code without any explanations outside of code comments. "
            f"Add comments within the code to explain important fixes you've made."
        )
        
        system_prompt = (
            f"You are an expert code improver for iteration {iteration}. "
            f"You are fixing code based on identified issues. "
            f"Focus on addressing the specific issues while preserving the original functionality. "
            f"Return only the improved code, properly formatted and ready to execute. "
            f"Do not include any explanation or markdown outside the code itself - only code and code comments."
        )
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": 0.2,
            "stream": False
        }
        
        try:
            logger.info(f"Requesting code improvement from Ollama...")
            response = requests.post(f"{self.base_url}/generate", json=request_data)
            
            if response.status_code != 200:
                logger.error(f"Failed to generate improved code: {response.text}")
                return code, [f"Error: Failed to generate improved code"]
            
            result = response.json()
            improved_code = result.get("response", "# Error: No response from Ollama")
            
            # Extract code block if present
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
                    improved_code = code_blocks[0]  # Use the first code block
            
            # Generate a list of improvements made
            improvements_made = self._identify_improvements(code, improved_code, issues)
            
            return improved_code, improvements_made
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {str(e)}")
            return code, [f"Error: {str(e)}"]
    
    def _identify_improvements(self, old_code: str, new_code: str, issues: List[str]) -> List[str]:
        """
        Identify what improvements were made in the new code.
        
        Args:
            old_code: The original code
            new_code: The improved code
            issues: The issues that needed to be fixed
            
        Returns:
            List of improvements that were likely made
        """
        improvements = []
        
        # Simple indicators of changes
        if old_code == new_code:
            return ["No changes were made"]
        
        # Check for specific patterns
        if "Fix syntax error" in " ".join(issues):
            if self._has_syntax_improvement(old_code, new_code):
                improvements.append("Fixed syntax errors")
        
        # Check for style improvements
        old_lines = old_code.split("\n")
        new_lines = new_code.split("\n")
        
        if any("line is too long" in issue for issue in issues):
            if self._count_long_lines(old_lines) > self._count_long_lines(new_lines):
                improvements.append("Shortened long lines")
        
        if "insufficient comments" in " ".join(issues).lower():
            old_comments = self._count_comments(old_lines)
            new_comments = self._count_comments(new_lines)
            if new_comments > old_comments:
                improvements.append(f"Added comments (from {old_comments} to {new_comments})")
        
        # Check for error handling
        if "error handling" in " ".join(issues).lower():
            if "try:" in new_code and "try:" not in old_code:
                improvements.append("Added try-except error handling")
            elif new_code.count("except") > old_code.count("except"):
                improvements.append("Improved error handling")
        
        # If we haven't identified specific improvements, add a generic one
        if not improvements:
            improvements.append("Made code improvements based on feedback")
        
        return improvements
    
    def _has_syntax_improvement(self, old_code: str, new_code: str) -> bool:
        """Check if syntax errors were likely fixed."""
        import ast
        
        # Try to parse the old code
        try:
            ast.parse(old_code)
            old_code_valid = True
        except SyntaxError:
            old_code_valid = False
        
        # Try to parse the new code
        try:
            ast.parse(new_code)
            new_code_valid = True
        except SyntaxError:
            new_code_valid = False
        
        # If old code had syntax errors but new code doesn't, that's an improvement
        return (not old_code_valid) and new_code_valid
    
    def _count_long_lines(self, lines: List[str], max_length: int = 100) -> int:
        """Count the number of lines longer than max_length."""
        return sum(1 for line in lines if len(line) > max_length)
    
    def _count_comments(self, lines: List[str]) -> int:
        """Count the number of comment lines."""
        return sum(1 for line in lines if line.strip().startswith("#") or '"""' in line or "'''" in line) 