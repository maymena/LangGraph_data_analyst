import os
import re
from typing import Optional
from openai import OpenAI

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize OpenAI client with API key from environment
api_key = os.environ.get("NEBIUS_API_KEY")
if not api_key:
    raise ValueError("NEBIUS_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=api_key
)

def call_llm(prompt: str, llm_client: Optional[object] = None, model: str = "Qwen/Qwen3-30B-A3B", temperature: float = 0) -> str:
    """
    Call the LLM with the given prompt. If llm_client is None, use the default client from this file.
    Returns the LLM's response as a string.
    """
    if llm_client is None:
        _client = client
    else:
        _client = llm_client
    response = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing <think></think> tags and their content
    
    Args:
        response: Raw LLM response that may contain think tags
        
    Returns:
        Cleaned response without think tags
    """
    if not response:
        return response
    
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Clean up any extra whitespace that might be left
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple newlines to double
    cleaned = cleaned.strip()
    
    return cleaned