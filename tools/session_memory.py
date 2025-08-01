"""
Session Memory Tool Interface for LangGraph Workflow with Checkpoint System
"""

from typing import List, Dict, Any
import json
import re
from tools.llm_utils import call_llm


def access_session_memory(user_input: str, session_history: List[Dict[str, Any]]) -> str:
    """
    Access session memory for questions that reference current session information
    
    Args:
        user_input: The current user question
        session_history: List of previous interactions from checkpoint system
        
    Returns:
        Response string based on session memory
    """
    if not session_history:
        return "I don't have any previous interactions to reference in this session."
    
    # Get the most recent interactions (last 5)
    recent_history = session_history[-5:]
    
    # Create a prompt to analyze the user's memory-related question
    memory_prompt = f"""
    Based on the following conversation history, answer the user's question about previous interactions.
    
    Conversation History:
    {json.dumps(recent_history, indent=2)}
    
    User Question: {user_input}
    
    Please provide a helpful response that references the relevant information from our previous conversation.
    If the user is asking for "more examples" or similar, try to identify what they're referring to from the context.
    If the user is asking about previous results, categories, intents, or data mentioned, provide that information.
    
    Be conversational and helpful. Reference specific details from the conversation history when relevant.
    """
    
    try:
        response = call_llm(memory_prompt)
        return response
    except Exception as e:
        return f"I had trouble accessing the conversation history. Error: {str(e)}"


def should_use_session_memory(user_input: str, has_session_history: bool = False) -> bool:
    """
    Determine if session memory should be used for this question using LLM
    
    Args:
        user_input: The user's question
        has_session_history: Whether there is session history available
        
    Returns:
        True if session memory should be used directly (bypass workflow), False otherwise
    """
    if not has_session_history:
        return False
    
    # Use LLM to determine if this is a direct memory question
    memory_detection_prompt = f"""
    Analyze the following user question and determine if it's asking for information from previous conversation history.

    User question: "{user_input}"

    A question should be classified as a DIRECT MEMORY question if it:
    1. Explicitly asks about what was said before (e.g., "What did you tell me before?", "You mentioned something about...")
    2. Asks to repeat or recall previous responses (e.g., "Tell me again", "Repeat what you said")
    3. References previous conversation directly (e.g., "From our conversation", "You said earlier")

    A question should NOT be classified as direct memory if it:
    1. Asks for examples or details about a topic mentioned before (these should go through workflow with memory context)
    2. Asks for analysis or recommendations based on previous data (these should go through workflow)
    3. Is a follow-up question that needs new processing (e.g., "Show me examples from that category")

    Respond with only "YES" if this is a direct memory question that should bypass the workflow entirely.
    Respond with only "NO" if this should go through the workflow (possibly with memory context).
    """
    
    try:
        llm_response = call_llm(memory_detection_prompt)
        
        # Extract the final answer from the response (handle <think> tags)
        if "</think>" in llm_response:
            # Extract text after the last </think> tag
            final_answer = llm_response.split("</think>")[-1].strip()
        else:
            final_answer = llm_response.strip()
        
        is_direct_memory = final_answer.upper() == "YES"
        return is_direct_memory
    except Exception as e:
        print(f"Error in LLM memory detection: {e}")
        # Fallback to simple keyword check for critical cases
        direct_memory_keywords = ["what did you tell me", "what did you say", "you said", "tell me again"]
        return any(keyword in user_input.lower() for keyword in direct_memory_keywords)


def get_memory_context(session_history: List[Dict[str, Any]], context_type: str = "recent") -> str:
    """
    Get formatted memory context for use in prompts
    
    Args:
        session_history: List of session interactions from checkpoint system
        context_type: Type of context to retrieve ("recent", "all", "tools_used")
        
    Returns:
        Formatted string of memory context
    """
    if not session_history:
        return "No previous interactions in this session."
    
    if context_type == "recent":
        # Get last 3 interactions
        recent = session_history[-3:]
        context = "Recent interactions:\n"
        for i, interaction in enumerate(recent, 1):
            context += f"{i}. Q: {interaction['user_query']}\n"
            context += f"   A: {interaction['response'][:200]}...\n"
            if interaction.get('tools_used'):
                context += f"   Tools used: {', '.join(interaction['tools_used'])}\n"
            context += "\n"
        return context
    
    elif context_type == "tools_used":
        # Get summary of tools used
        all_tools = []
        for interaction in session_history:
            all_tools.extend(interaction.get('tools_used', []))
        unique_tools = list(set(all_tools))
        return f"Tools used in this session: {', '.join(unique_tools)}"
    
    elif context_type == "all":
        # Get all interactions (truncated)
        context = f"All {len(session_history)} interactions in this session:\n"
        for i, interaction in enumerate(session_history, 1):
            context += f"{i}. {interaction['user_query']} -> {interaction['response'][:100]}...\n"
        return context
    
    return "Invalid context type requested."


def summarize_session_memory(session_history: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the entire session memory
    
    Args:
        session_history: List of session interactions from checkpoint system
        
    Returns:
        Summary string of the session
    """
    if not session_history:
        return "No interactions in this session yet."
    
    # Create summary prompt
    summary_prompt = f"""
    Please provide a concise summary of this conversation session with a customer service dataset Q&A agent.
    
    Session Data:
    - Total interactions: {len(session_history)}
    - Session started: {session_history[0].get('timestamp', 'Unknown')}
    - Last interaction: {session_history[-1].get('timestamp', 'Unknown')}
    
    Interactions:
    {json.dumps(session_history, indent=2)}
    
    Please summarize:
    1. Main topics discussed
    2. Types of questions asked (structured vs unstructured)
    3. Key findings or insights discovered
    4. Tools most frequently used
    5. Any patterns in the user's interests
    
    Keep the summary concise but informative.
    """
    
    try:
        summary = call_llm(summary_prompt)
        return summary
    except Exception as e:
        return f"Could not generate session summary. Error: {str(e)}"


def extract_memory_relevant_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract memory-relevant information from the current state for checkpoint storage
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with memory-relevant information
    """
    return {
        "user_query": state.get("user_input", ""),
        "response": state.get("final_response", ""),
        "tools_used": state.get("tools_used", []),
        "processing_results": state.get("processing_results", []),
        "question_type": state.get("question_type", ""),
        "structure_type": state.get("structure_type", "")
    }
