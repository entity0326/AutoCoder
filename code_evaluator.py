"""
Code evaluator module for the AutoCoder AI agent.
This module evaluates generated code for correctness, efficiency, and quality.
"""
import re
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests

from utils import execute_code

logger = logging.getLogger("autocoder.evaluator")

class CodeEvaluator:
    """Class for evaluating generated code."""
    
    def __init__(self, model: str = "codellama"):
        """
        Initialize the code evaluator.
        
        Args:
            model: The Ollama model to use for evaluation
        """
        self.model = model
        self.base_url = "http://localhost:11434/api"
    
    def evaluate(self, code: str, task: str, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate code based on the task and optional test cases.
        
        Args:
            code: The code to evaluate
            task: The original task description
            test_cases: Optional list of test cases for functional testing
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Evaluating generated code...")
        
        # Initialize evaluation result
        evaluation = {
            "syntax_check": self._check_syntax(code),
            "style_check": self._check_style(code),
            "complexity": self._estimate_complexity(code),
            "ai_feedback": self._get_ai_feedback(code, task),
            "execution": None,
            "test_results": None,
        }
        
        # Execute the code to see if it runs
        if evaluation["syntax_check"]["valid"]:
            success, output, error = execute_code(code)
            evaluation["execution"] = {
                "success": success,
                "output": output,
                "error": error,
            }
        
        # Run test cases if provided
        if test_cases and evaluation["syntax_check"]["valid"]:
            evaluation["test_results"] = self._run_test_cases(code, test_cases)
        
        # Calculate overall score
        evaluation["overall_score"] = self._calculate_overall_score(evaluation)
        
        return evaluation
    
    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check if the code has valid syntax."""
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {str(e)}")
            return {"valid": False, "error": str(e)}
        except Exception as e:
            logger.warning(f"Error parsing generated code: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def _check_style(self, code: str) -> Dict[str, Any]:
        """Check code style and quality metrics."""
        style_issues = []
        
        # Check line length
        long_lines = 0
        for i, line in enumerate(code.split("\n")):
            if len(line) > 100:
                long_lines += 1
                if len(style_issues) < 5:  # Limit number of reported issues
                    style_issues.append(f"Line {i+1} is too long ({len(line)} > 100 characters)")
        
        # Check function length
        functions = self._extract_functions(code)
        long_functions = 0
        for func_name, func_body in functions.items():
            if func_body.count("\n") > 50:
                long_functions += 1
                if len(style_issues) < 10:
                    style_issues.append(f"Function '{func_name}' is too long ({func_body.count('\\n')} lines)")
        
        # Check comment ratio
        comment_lines = 0
        code_lines = 0
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("#") or line.startswith('"""') or line.startswith("'''"):
                comment_lines += 1
            elif line and not line.startswith("#"):
                code_lines += 1
        
        comment_ratio = comment_lines / max(1, code_lines + comment_lines)
        
        if comment_ratio < 0.1:
            style_issues.append("Code has insufficient comments")
        
        # Return style check results
        return {
            "issues": style_issues,
            "metrics": {
                "long_lines": long_lines,
                "long_functions": long_functions,
                "comment_ratio": comment_ratio,
            }
        }
    
    def _extract_functions(self, code: str) -> Dict[str, str]:
        """Extract functions from code."""
        functions = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function name
                    func_name = node.name
                    
                    # Get line numbers
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    # Extract function body from source code
                    code_lines = code.split("\n")
                    func_body = "\n".join(code_lines[start_line-1:end_line])
                    
                    functions[func_name] = func_body
                    
            return functions
        except Exception as e:
            logger.warning(f"Error extracting functions: {str(e)}")
            return {}
    
    def _estimate_complexity(self, code: str) -> Dict[str, Any]:
        """Estimate code complexity."""
        complexity = {
            "cyclomatic": 0,
            "nesting_depth": 0,
        }
        
        # Count control structures as a simple complexity estimate
        control_keywords = ["if", "for", "while", "except", "with"]
        for keyword in control_keywords:
            pattern = r'\b' + keyword + r'\b'
            complexity["cyclomatic"] += len(re.findall(pattern, code))
        
        # Estimate nesting depth
        max_depth = 0
        current_depth = 0
        for line in code.split("\n"):
            line = line.strip()
            
            # Increase depth for indentation blocks
            if line.endswith(":") and any(keyword in line for keyword in control_keywords):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            # Decrease depth when finding standalone 'return', 'break', etc.
            if current_depth > 0 and (line.startswith("return") or line.startswith("break") or line.startswith("continue")):
                current_depth -= 1
        
        complexity["nesting_depth"] = max_depth
        return complexity
    
    def _get_ai_feedback(self, code: str, task: str) -> Dict[str, Any]:
        """Get AI feedback on the code using Ollama."""
        prompt = (
            f"Task: {task}\n\n"
            f"Generated Code:\n```\n{code}\n```\n\n"
            f"Please evaluate this code in the following aspects:\n"
            f"1. Correctness: Does the code correctly solve the task?\n"
            f"2. Efficiency: Is the code efficient in terms of time and space complexity?\n"
            f"3. Readability: Is the code well-structured and easy to understand?\n"
            f"4. Error handling: Does the code handle potential errors appropriately?\n"
            f"5. Improvements: What specific improvements would you suggest?\n\n"
            f"Format your response as a structured JSON with the following fields:\n"
            f"\"correctness\": (0-10 score and brief explanation),\n"
            f"\"efficiency\": (0-10 score and brief explanation),\n"
            f"\"readability\": (0-10 score and brief explanation),\n"
            f"\"error_handling\": (0-10 score and brief explanation),\n"
            f"\"suggestions\": [array of specific improvement suggestions],\n"
            f"\"overall_feedback\": (brief overall assessment)"
        )
        
        system_prompt = (
            "You are an expert code reviewer. Provide detailed, constructive feedback on code. "
            "Your feedback should be honest but fair, highlighting both strengths and areas for improvement. "
            "Always respond in the exact JSON format requested in the prompt."
        )
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/generate", json=request_data)
            
            if response.status_code != 200:
                logger.error(f"Failed to get AI feedback: {response.text}")
                return {"error": f"Failed to get AI feedback: {response.text}"}
            
            result = response.json()
            feedback_text = result.get("response", "")
            
            # Try to extract JSON from the response
            try:
                # Find JSON-like content in the response
                json_match = re.search(r'({[\s\S]*})', feedback_text)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                else:
                    # Create a structured dictionary if JSON parsing fails
                    return {
                        "correctness": {"score": 5, "explanation": "Unable to parse AI feedback"},
                        "efficiency": {"score": 5, "explanation": "Unable to parse AI feedback"},
                        "readability": {"score": 5, "explanation": "Unable to parse AI feedback"},
                        "error_handling": {"score": 5, "explanation": "Unable to parse AI feedback"},
                        "suggestions": ["Unable to parse AI feedback"],
                        "overall_feedback": feedback_text[:200]  # Truncated feedback
                    }
            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI feedback as JSON: {feedback_text}")
                return {
                    "correctness": {"score": 5, "explanation": "Unable to parse AI feedback"},
                    "efficiency": {"score": 5, "explanation": "Unable to parse AI feedback"},
                    "readability": {"score": 5, "explanation": "Unable to parse AI feedback"},
                    "error_handling": {"score": 5, "explanation": "Unable to parse AI feedback"},
                    "suggestions": ["Unable to parse AI feedback"],
                    "overall_feedback": feedback_text[:200]  # Truncated feedback
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {str(e)}")
            return {"error": str(e)}
    
    def _run_test_cases(self, code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run test cases on the code."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i+1}")
            
            # Create a test file that imports the code and runs the test
            test_file = f"""
import sys
from io import StringIO
import json

# Original code to test
{code}

# Test case input
test_input = {repr(test_case.get('input', ''))}

# Capture stdout
original_stdout = sys.stdout
sys.stdout = StringIO()

# Redirect stdin if input is provided
if test_input:
    original_stdin = sys.stdin
    sys.stdin = StringIO(test_input)

try:
    # Run the main function if it exists
    if 'main' in globals():
        main()
    
    # Get the captured output
    output = sys.stdout.getvalue()
    print(json.dumps({{"status": "success", "output": output}}))
except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}))
finally:
    # Restore stdout and stdin
    sys.stdout = original_stdout
    if test_input:
        sys.stdin = original_stdin
"""
            
            # Execute the test file
            success, output, error = execute_code(test_file)
            
            # Process the result
            test_result = {
                "test_case": test_case,
                "success": success,
                "output": output.strip() if success else None,
                "error": error if not success else None,
                "passed": False
            }
            
            # Check if the output contains a JSON result
            if success:
                try:
                    result_json = json.loads(output.strip())
                    test_result["status"] = result_json.get("status")
                    
                    if result_json.get("status") == "success":
                        test_result["output"] = result_json.get("output")
                        
                        # Check against expected output if provided
                        if "expected_output" in test_case:
                            expected = str(test_case["expected_output"]).strip()
                            actual = result_json.get("output", "").strip()
                            test_result["passed"] = expected in actual
                            
                    else:
                        test_result["error"] = result_json.get("error")
                except json.JSONDecodeError:
                    test_result["output"] = output.strip()
                    
                    # Try simple string matching against expected output
                    if "expected_output" in test_case:
                        expected = str(test_case["expected_output"]).strip()
                        actual = output.strip()
                        test_result["passed"] = expected in actual
            
            results.append(test_result)
        
        return results
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate an overall score for the code based on all evaluations."""
        score = 0.0
        max_score = 0.0
        
        # Syntax check (25%)
        if evaluation["syntax_check"]["valid"]:
            score += 25
        max_score += 25
        
        # Style check (15%)
        style_issues = len(evaluation["style_check"]["issues"])
        style_score = max(0, 15 - style_issues)
        score += style_score
        max_score += 15
        
        # Execution (25%)
        if evaluation["execution"] and evaluation["execution"]["success"]:
            score += 25
        max_score += 25
        
        # AI feedback (35%)
        if "ai_feedback" in evaluation and not isinstance(evaluation["ai_feedback"], str):
            ai_feedback = evaluation["ai_feedback"]
            if "correctness" in ai_feedback and isinstance(ai_feedback["correctness"], dict):
                score += ai_feedback["correctness"].get("score", 0) * 1.0  # 10%
                max_score += 10
            
            if "efficiency" in ai_feedback and isinstance(ai_feedback["efficiency"], dict):
                score += ai_feedback["efficiency"].get("score", 0) * 0.75  # 7.5%
                max_score += 7.5
            
            if "readability" in ai_feedback and isinstance(ai_feedback["readability"], dict):
                score += ai_feedback["readability"].get("score", 0) * 1.0  # 10%
                max_score += 10
            
            if "error_handling" in ai_feedback and isinstance(ai_feedback["error_handling"], dict):
                score += ai_feedback["error_handling"].get("score", 0) * 0.75  # 7.5%
                max_score += 7.5
        
        # Test results
        if evaluation["test_results"]:
            test_count = len(evaluation["test_results"])
            if test_count > 0:
                passed_tests = sum(1 for test in evaluation["test_results"] if test.get("passed", False))
                test_score = (passed_tests / test_count) * 100
                score += test_score
                max_score += 100
        
        # Calculate final percentage
        if max_score > 0:
            return round((score / max_score) * 100, 2)
        else:
            return 0.0 