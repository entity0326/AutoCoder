"""
Code generator module for the AutoCoder AI agent.
This module handles interaction with Ollama to generate code.
"""
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("autocoder.generator")

class CodeGenerator:
    """Class for generating code using Ollama."""
    
    def __init__(self, model: str = "codellama", temperature: float = 0.2):
        """
        Initialize the code generator.
        
        Args:
            model: The Ollama model to use for code generation
            temperature: The temperature parameter for generation (0.0-1.0)
        """
        self.model = model
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api"
        self.validate_model()
    
    def validate_model(self) -> bool:
        """Validate that the specified model exists in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code != 200:
                logger.error(f"Failed to get model list: {response.text}")
                return False
            
            models = response.json().get("models", [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found in Ollama. Available models: {', '.join(model_names)}")
                logger.warning(f"You may need to run: ollama pull {self.model}")
                return False
            
            logger.info(f"Using Ollama model: {self.model}")
            return True
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            logger.error("Make sure Ollama is installed and running")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 4000, system_prompt: Optional[str] = None) -> str:
        """
        Generate code using Ollama based on the provided prompt.
        
        Args:
            prompt: The prompt describing the code to generate
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the model
            
        Returns:
            Generated code as a string
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert programmer. Generate clean, efficient, and well-documented code. "
                "Focus only on producing working code that meets the requirements. "
                "Don't include explanations outside of code comments."
            )
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": self.temperature,
            "num_predict": max_tokens,
            "options": {
                "num_ctx": 4096,  # Context window size
                "top_k": 40,
                "top_p": 0.9,
            },
            "stream": False
        }
        
        try:
            logger.info(f"Generating code with model {self.model}...")
            response = requests.post(f"{self.base_url}/generate", json=request_data)
            
            if response.status_code != 200:
                logger.error(f"Failed to generate code: {response.text}")
                return f"# Error generating code: {response.text}"
            
            result = response.json()
            generated_code = result.get("response", "# No code generated")
            
            # Extract code blocks if needed
            if "```" in generated_code:
                code_blocks = []
                inside_code_block = False
                current_block = []
                
                for line in generated_code.split("\n"):
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
                    return code_blocks[0]  # Return the first code block
            
            return generated_code
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {str(e)}")
            return f"# Error: {str(e)}"
    
    def generate_with_context(self, task: str, existing_code: Optional[str] = None) -> str:
        """
        Generate code for a task with optional context from existing code.
        
        Args:
            task: The task description
            existing_code: Optional existing code to provide context
            
        Returns:
            Generated code as a string
        """
        prompt = f"Task: {task}\n\n"
        
        if existing_code:
            prompt += f"Existing code:\n```\n{existing_code}\n```\n\n"
        
        prompt += "Generate the code to solve this task:"
        
        system_prompt = (
            "You are an expert programmer tasked with writing high-quality code. "
            "Your output should be ready to use, well-documented, and follow best practices. "
            "Provide only the code without additional explanations. "
            "Focus on correctness, efficiency, and readability."
        )
        
        return self.generate(prompt, system_prompt=system_prompt)
    
    def improve_code(self, code: str, feedback: str) -> str:
        """
        Improve code based on feedback.
        
        Args:
            code: The original code to improve
            feedback: Feedback or evaluation about the code
            
        Returns:
            Improved code as a string
        """
        prompt = (
            f"I have the following code that needs improvement:\n\n"
            f"```\n{code}\n```\n\n"
            f"Here is the feedback about the code:\n{feedback}\n\n"
            f"Please provide an improved version of the code addressing the feedback."
        )
        
        system_prompt = (
            "You are an expert code reviewer and improver. "
            "Your task is to enhance the given code based on the provided feedback. "
            "Return only the improved code without explanations outside of code comments."
        )
        
        return self.generate(prompt, system_prompt=system_prompt) 