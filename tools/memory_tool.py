"""
Memory tool for accessing previous conversation history within the ReAct loop
"""

from typing import List, Dict, Any, Optional
from tools.session_memory import access_session_memory

def memory_tool(query: str, context: Optional[List[Dict[str, Any]]] = None, 
                persistent_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Access previous conversation history to answer memory-related queries
    Combines both session memory (current session) and persistent memory (across sessions)
    
    Args:
        query: The memory query (e.g., "what number did you tell me before?")
        context: Optional session history context (current session)
        persistent_context: Optional persistent memory context (across sessions)
        
    Returns:
        Dictionary with memory results
    """
    if not context and not persistent_context:
        return {
            "error": "No conversation history available",
            "result": "I don't have access to previous conversation history.",
            "source": "none"
        }
    
    try:
        # Combine session and persistent memory
        combined_context = []
        
        # Add persistent memory first (older interactions)
        if persistent_context:
            combined_context.extend(persistent_context)
        
        # Add session memory last (most recent interactions)
        if context:
            combined_context.extend(context)
        
        # Use the existing session memory access function with combined context
        memory_response = access_session_memory(query, combined_context)
        
        # Determine the source of information
        source = "combined"
        if context and not persistent_context:
            source = "session"
        elif persistent_context and not context:
            source = "persistent"
        
        return {
            "query": query,
            "result": memory_response,
            "source": source,
            "session_count": len(context) if context else 0,
            "persistent_count": len(persistent_context) if persistent_context else 0,
            "total_history": len(combined_context),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "result": f"Error accessing memory: {str(e)}",
            "source": "error",
            "success": False
        }


# Tool metadata for registration
MEMORY_TOOL_INFO = {
    "name": "memory",
    "description": "Access previous conversation history to retrieve information from past interactions. Combines both current session and persistent memory across sessions.",
    "parameters": {
        "query": {
            "type": "string", 
            "description": "The memory query describing what information to retrieve from previous conversations"
        }
    },
    "required": ["query"]
}
