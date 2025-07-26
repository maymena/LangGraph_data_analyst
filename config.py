"""
Configuration for the LangGraph Customer Service Q&A Agent
"""

import os
from typing import Dict, Any


class Config:
    """Configuration settings for the application"""
    
    # API Keys
    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Dataset settings
    DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    
    # Memory settings
    MEMORY_LIMIT = 100  # Maximum number of interactions to store
    MEMORY_SEARCH_LIMIT = 5  # Maximum number of memory results to return
    
    # LLM settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.1
    
    # Workflow settings
    MAX_PROCESSING_STEPS = 10
    ENABLE_DEBUG_LOGGING = True
    
    # Classification thresholds
    OUT_OF_SCOPE_THRESHOLD = 0.3
    MEMORY_QUERY_THRESHOLD = 0.5
    
    # Tool settings
    DEFAULT_EXAMPLES_COUNT = 5
    DEFAULT_DISTRIBUTION_TOP_N = 10
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.NEBIUS_API_KEY and not cls.OPENAI_API_KEY:
            raise ValueError("Either NEBIUS_API_KEY or OPENAI_API_KEY must be set")
        return True
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model": cls.DEFAULT_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "api_key": cls.NEBIUS_API_KEY or cls.OPENAI_API_KEY
        }


# Workflow node configurations
NODE_CONFIGS = {
    "question_classification": {
        "timeout": 30,
        "retry_count": 3
    },
    "structured_processing": {
        "max_results": 100,
        "timeout": 60
    },
    "unstructured_processing": {
        "max_context_length": 4000,
        "timeout": 90
    },
    "memory": {
        "search_limit": Config.MEMORY_SEARCH_LIMIT,
        "timeout": 30
    },
    "summarization": {
        "max_summary_length": 500,
        "timeout": 45
    }
}

# Question classification patterns
CLASSIFICATION_PATTERNS = {
    "out_of_scope": [
        "weather", "sports", "politics", "personal", "unrelated"
    ],
    "memory_keywords": [
        "remember", "previous", "before", "earlier", "last time", 
        "you said", "we discussed", "from before"
    ],
    "structured_keywords": [
        "how many", "count", "number of", "distribution", "examples", 
        "show me", "list", "what are", "which", "top"
    ],
    "unstructured_keywords": [
        "summarize", "analyze", "insights", "explain", "describe", 
        "tell me about", "what do you think", "overview"
    ]
}

# Dataset field mappings
DATASET_FIELDS = {
    "instruction": "User instruction/query",
    "category": "Service category", 
    "intent": "User intent",
    "response": "Agent response"
}

# Error messages
ERROR_MESSAGES = {
    "out_of_scope": "I'm sorry, but I can only answer questions about the customer service dataset. Please ask questions related to customer service categories, intents, or data analysis.",
    "processing_error": "I encountered an error processing your request. Please try again or rephrase your question.",
    "memory_error": "I had trouble accessing previous conversations. Please try your question again.",
    "timeout_error": "Your request took too long to process. Please try a simpler question.",
    "no_results": "I couldn't find any relevant information for your query. Please try rephrasing your question."
}
