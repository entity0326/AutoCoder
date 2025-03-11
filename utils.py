"""
Utility functions for the AutoCoder AI agent.
"""
import os
import json
import logging
import tempfile
import subprocess
from termcolor import colored
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("autocoder")

def setup_environment() -> None:
    """Set up the environment for the agent."""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("history", exist_ok=True)

def save_to_file(content: str, file_path: str) -> None:
    """Save content to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Content saved to {file_path}")

def load_from_file(file_path: str) -> str:
    """Load content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        return ""

def save_history(task: str, prompt: str, code: str, evaluation: Dict[str, Any]) -> None:
    """Save the history of a code generation and evaluation."""
    timestamp = get_timestamp()
    history = {
        "timestamp": timestamp,
        "task": task,
        "prompt": prompt,
        "code": code,
        "evaluation": evaluation,
    }
    
    filename = f"history/history_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"History saved to {filename}")

def get_timestamp() -> str:
    """Get a timestamp string for file naming."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def execute_code(code: str, language: str = "python") -> Tuple[bool, str, str]:
    """
    Execute the generated code in a safe environment.
    
    Args:
        code: The code to execute
        language: The programming language of the code
        
    Returns:
        Tuple containing (success flag, output, error message)
    """
    if language != "python":
        logger.warning(f"Execution of {language} code is not yet supported")
        return False, "", f"Execution of {language} code is not supported"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode("utf-8"))
    
    try:
        # Execute the code with a timeout
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )
        
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        
        if success:
            logger.info("Code executed successfully")
        else:
            logger.error(f"Code execution failed: {error}")
        
        return success, output, error
    
    except subprocess.TimeoutExpired:
        logger.error("Code execution timed out")
        return False, "", "Execution timed out (30 seconds)"
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def print_colored_diff(old_code: str, new_code: str) -> None:
    """Print a colored diff between old and new code."""
    try:
        import difflib
        
        diff = difflib.unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            lineterm="",
        )
        
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                print(colored(line, "green"))
            elif line.startswith("-") and not line.startswith("---"):
                print(colored(line, "red"))
            else:
                print(line)
    except Exception as e:
        logger.error(f"Error generating diff: {str(e)}")
        print("Old code:\n", old_code)
        print("\nNew code:\n", new_code)

def check_ollama_availability() -> bool:
    """Check if Ollama is available and running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False 