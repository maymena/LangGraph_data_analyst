"""
Workflow Manager for LangGraph Customer Service Q&A Agent with Memory
"""

from typing import Dict, Any, List, Optional, Tuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph_workflow import create_workflow, WorkflowState
from tools.session_memory import (
    should_use_session_memory, 
    access_session_memory, 
    summarize_session_memory,
    get_memory_context
)
import uuid
import datetime


class WorkflowManager:
    """
    Manages the LangGraph workflow with checkpoint-based memory system
    """
    
    def __init__(self):
        """Initialize the workflow manager with memory checkpointer"""
        self.workflow = create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def run_query(self, user_input: str, thread_id: Optional[str] = None, user_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Run a query with memory support
        
        Args:
            user_input: The user's question
            thread_id: Optional thread ID for session continuity
            user_name: Optional username for persistent memory
            
        Returns:
            Tuple of (response, thread_id)
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if this is a memory-related query
        session_history = self.get_session_history(thread_id)
        is_memory_query = should_use_session_memory(user_input, bool(session_history))
        
        if is_memory_query and session_history:
            # Handle memory query directly
            response = access_session_memory(user_input, session_history)
            
            # Still run through workflow to maintain state consistency
            initial_state = self._create_initial_state(user_input, thread_id, user_name)
            initial_state["final_response"] = response
            initial_state["tools_used"] = ["session_memory"]
            
            # Save this interaction to checkpoint
            result = self.app.invoke(initial_state, config)
            
            # Save to persistent memory if user_name provided
            if user_name:
                self._save_to_persistent_memory(user_name, user_input, response, ["session_memory"], "memory", None)
            
            return result["final_response"], thread_id
        else:
            # Run normal workflow
            initial_state = self._create_initial_state(user_input, thread_id, user_name)
            
            # Add memory context if available
            if session_history:
                initial_state["contains_memory_query"] = True
            
            result = self.app.invoke(initial_state, config)
            return result["final_response"], thread_id
    
    def _create_initial_state(self, user_input: str, thread_id: str = None, user_name: str = None) -> Dict[str, Any]:
        """Create initial state for workflow with memory context"""
        initial_state = {
            "user_input": user_input,
            "question_type": None,
            "structure_type": None,
            "contains_memory_query": False,
            "memory_results": [],
            "processing_results": [],
            "final_response": "",
            "tools_used": [],
            "error": "",
            "user_name": user_name or "",
            "is_identified": bool(user_name),
            "persistent_context": [],
            "user_record": None
        }
        
        # Add memory context if thread_id is available
        if thread_id:
            initial_state["thread_id"] = thread_id
            # Pre-load session history for access_memory_node
            session_history = self.get_session_history(thread_id)
            initial_state["session_history"] = session_history
        
        # Load persistent memory context if user_name provided
        if user_name:
            try:
                from tools.persistent_memory import get_persistent_memory_context, get_or_create_user
                persistent_context = get_persistent_memory_context(user_name, limit=20)
                user_record = get_or_create_user(user_name)
                initial_state["persistent_context"] = persistent_context
                initial_state["user_record"] = user_record
            except Exception as e:
                print(f"Error loading persistent memory for {user_name}: {e}")
        
        return initial_state
    
    def _save_to_persistent_memory(self, user_name: str, user_input: str, response: str, 
                                 tools_used: List[str], question_type: str = None, 
                                 structure_type: str = None, processing_results: List[Dict] = None) -> bool:
        """
        Save interaction to persistent memory
        
        Args:
            user_name: Username
            user_input: User's question
            response: Agent's response
            tools_used: List of tools used
            question_type: Type of question
            structure_type: Structure type
            processing_results: Processing results
            
        Returns:
            True if successful
        """
        try:
            from tools.persistent_memory import save_interaction_to_persistent_memory
            interaction = {
                "user_query": user_input,
                "response": response,
                "tools_used": tools_used,
                "question_type": question_type,
                "structure_type": structure_type,
                "processing_results": processing_results or []
            }
            return save_interaction_to_persistent_memory(user_name, interaction)
        except Exception as e:
            print(f"Error saving to persistent memory: {e}")
            return False
    
    def get_session_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get the session history for a given thread ID
        
        Args:
            thread_id: The thread ID to get history for
            
        Returns:
            List of interactions in the session
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            history = []
            for checkpoint in self.app.get_state_history(config):
                state = checkpoint.values
                if state.get("user_input") and state.get("final_response"):
                    history.append({
                        "timestamp": checkpoint.metadata.get("timestamp", datetime.datetime.now().isoformat()),
                        "user_query": state["user_input"],
                        "response": state["final_response"],
                        "tools_used": state.get("tools_used", []),
                        "processing_results": state.get("processing_results", []),
                        "question_type": state.get("question_type", ""),
                        "structure_type": state.get("structure_type", "")
                    })
            
            # Return in chronological order (oldest first)
            return list(reversed(history))
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []
    
    def get_session_summary(self, thread_id: str) -> str:
        """
        Get a summary of the session
        
        Args:
            thread_id: The thread ID to summarize
            
        Returns:
            Summary string of the session
        """
        history = self.get_session_history(thread_id)
        return summarize_session_memory(history)
    
    def get_memory_context_for_prompt(self, thread_id: str, context_type: str = "recent") -> str:
        """
        Get memory context formatted for use in prompts
        
        Args:
            thread_id: The thread ID
            context_type: Type of context ("recent", "all", "tools_used")
            
        Returns:
            Formatted memory context string
        """
        history = self.get_session_history(thread_id)
        return get_memory_context(history, context_type)
    
    def clear_session(self, thread_id: str) -> bool:
        """
        Clear session memory (note: MemorySaver doesn't support this directly)
        
        Args:
            thread_id: The thread ID to clear
            
        Returns:
            True if successful (always True for MemorySaver)
        """
        # MemorySaver stores in memory, so clearing would require restarting
        # In production, you might want to use a different checkpointer
        return True
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session thread IDs
        Note: MemorySaver doesn't provide this functionality directly
        
        Returns:
            List of thread IDs (empty for MemorySaver)
        """
        # MemorySaver doesn't track all sessions
        return []


# Global instance for easy access
workflow_manager = WorkflowManager()


def run_workflow_with_memory(user_input: str, thread_id: Optional[str] = None, user_name: str = None) -> Tuple[str, str]:
    """
    Convenience function to run workflow with memory
    
    Args:
        user_input: The user's question
        thread_id: Optional thread ID for session continuity
        
    Returns:
        Tuple of (response, thread_id)
    """
    return workflow_manager.run_query(user_input, thread_id, user_name)


def get_session_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get session history
    
    Args:
        thread_id: The thread ID
        
    Returns:
        List of session interactions
    """
    return workflow_manager.get_session_history(thread_id)


def get_session_summary(thread_id: str) -> str:
    """
    Convenience function to get session summary
    
    Args:
        thread_id: The thread ID
        
    Returns:
        Session summary string
    """
    return workflow_manager.get_session_summary(thread_id)
