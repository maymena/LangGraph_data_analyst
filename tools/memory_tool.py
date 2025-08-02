"""
Memory tool for accessing previous conversation history within the ReAct loop
"""

from typing import List, Dict, Any, Optional
from tools.session_memory import access_session_memory

def memory_tool(query: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Access previous conversation history to answer memory-related queries
    
    Args:
        query: The memory query (e.g., "what number did you tell me before?")
        context: Optional session history context (will be provided by workflow)
        
    Returns:
        Dictionary with memory results
    """
    if not context:
        return {
            "error": "No conversation history available",
            "result": "I don't have access to previous conversation history."
        }
    
    try:
        # Use the existing session memory access function
        memory_response = access_session_memory(query, context)
        
        return {
            "query": query,
            "result": memory_response,
            "history_count": len(context),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "result": f"Error accessing memory: {str(e)}",
            "success": False
        }


# Tool metadata for registration
MEMORY_TOOL_INFO = {
    "name": "memory",
    "description": "Access previous conversation history to retrieve information from past interactions",
    "parameters": {
        "query": {
            "type": "string", 
            "description": "The memory query describing what information to retrieve from previous conversations"
        }
    },
    "required": ["query"]
}
