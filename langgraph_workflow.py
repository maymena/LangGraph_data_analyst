"""
LangGraph Workflow for Customer Service Dataset Q&A Agent
Refactored from ReAct agent to LangGraph workflow
"""

from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json


class WorkflowState(TypedDict):
    """State object that flows through the workflow"""
    user_input: str
    question_type: Literal["out_of_scope", "memory", "standard"]
    structure_type: Literal["structured", "unstructured"] 
    contains_memory_query: bool
    memory_results: List[Dict[str, Any]]
    processing_results: List[Dict[str, Any]]
    final_response: str
    tools_used: List[str]
    error: str


def question_classification_node(state: WorkflowState) -> WorkflowState:
    """
    Classify the incoming question into one of three types:
    - out_of_scope: Questions not related to the dataset
    - memory: Questions that can be answered from memory alone
    - standard: Questions requiring dataset analysis
    """
    user_input = state["user_input"]
    
    # TODO: Implement classification logic
    # This could use LLM classification or rule-based approach
    
    # Placeholder logic
    if "dataset" not in user_input.lower() and "customer" not in user_input.lower():
        question_type = "out_of_scope"
    elif "remember" in user_input.lower() or "previous" in user_input.lower():
        question_type = "memory"
    else:
        question_type = "standard"
    
    return {
        **state,
        "question_type": question_type
    }


def out_of_scope_node(state: WorkflowState) -> WorkflowState:
    """Handle out-of-scope questions with polite decline"""
    return {
        **state,
        "final_response": "I'm sorry, but I can only answer questions about the customer service dataset. Please ask questions related to customer service categories, intents, or data analysis."
    }


def memory_node(state: WorkflowState) -> WorkflowState:
    """
    Query memory system and provide response based on past interactions
    """
    user_input = state["user_input"]
    
    # TODO: Implement memory querying
    # - Search past interactions
    # - Find relevant context
    # - Generate response based on memory
    
    memory_results = []  # Placeholder
    response = "Based on our previous conversations..."  # Placeholder
    
    return {
        **state,
        "memory_results": memory_results,
        "final_response": response
    }


def question_structure_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    Analyze whether the question requires structured or unstructured processing
    Also determine if memory access is needed
    """
    user_input = state["user_input"]
    
    # TODO: Implement structure analysis
    # Structured: specific queries (counts, distributions, examples)
    # Unstructured: summaries, insights, analysis
    
    # Placeholder logic
    structured_keywords = ["how many", "count", "distribution", "examples", "show me"]
    unstructured_keywords = ["summarize", "analyze", "insights", "explain"]
    
    if any(keyword in user_input.lower() for keyword in structured_keywords):
        structure_type = "structured"
    else:
        structure_type = "unstructured"
    
    # Check if memory query is needed
    contains_memory_query = "previous" in user_input.lower() or "remember" in user_input.lower()
    
    return {
        **state,
        "structure_type": structure_type,
        "contains_memory_query": contains_memory_query
    }


def structured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle structured questions using dataset tools
    - select_semantic_intent
    - select_semantic_category  
    - count_category
    - count_intent
    - get_intent_distribution
    - get_category_distribution
    - show_examples
    """
    user_input = state["user_input"]
    
    # TODO: Implement structured processing
    # - Parse the question to identify required tools
    # - Call appropriate dataset functions
    # - Format results
    
    processing_results = []  # Placeholder for tool results
    tools_used = []  # Track which tools were used
    
    return {
        **state,
        "processing_results": processing_results,
        "tools_used": tools_used
    }


def unstructured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle unstructured questions requiring analysis and summarization
    - Use LLM for analysis
    - Generate insights
    - Create summaries
    """
    user_input = state["user_input"]
    
    # TODO: Implement unstructured processing
    # - Use summarize tool
    # - Generate insights using LLM
    # - Combine with dataset queries if needed
    
    processing_results = []  # Placeholder
    tools_used = ["summarize"]  # Placeholder
    
    return {
        **state,
        "processing_results": processing_results,
        "tools_used": tools_used
    }


def access_memory_node(state: WorkflowState) -> WorkflowState:
    """
    Access memory system when processing nodes need historical context
    """
    # TODO: Implement memory access
    # - Query relevant past interactions
    # - Add context to processing results
    
    memory_results = state.get("memory_results", [])
    # Add new memory results
    
    return {
        **state,
        "memory_results": memory_results
    }


def summarization_node(state: WorkflowState) -> WorkflowState:
    """
    Always called before output:
    1. Summarize the results into a coherent response
    2. Save the user query and response to memory
    """
    user_input = state["user_input"]
    processing_results = state.get("processing_results", [])
    memory_results = state.get("memory_results", [])
    
    # If we already have a final response (from out_of_scope or memory nodes), use it
    if state.get("final_response"):
        final_response = state["final_response"]
    else:
        # TODO: Implement response generation
        # - Combine processing results
        # - Format for user consumption
        # - Generate coherent narrative
        final_response = "Generated response based on analysis..."  # Placeholder
    
    # TODO: Save to memory
    # - Store user query
    # - Store response
    # - Store tools used
    # - Store timestamp
    
    return {
        **state,
        "final_response": final_response
    }


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("question_classification", question_classification_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("question_structure_analysis", question_structure_analysis_node)
    workflow.add_node("structured_processing", structured_processing_node)
    workflow.add_node("unstructured_processing", unstructured_processing_node)
    workflow.add_node("access_memory", access_memory_node)
    workflow.add_node("summarization", summarization_node)
    
    # Set entry point
    workflow.set_entry_point("question_classification")
    
    # Add conditional edges based on question type
    workflow.add_conditional_edges(
        "question_classification",
        lambda state: state["question_type"],
        {
            "out_of_scope": "out_of_scope",
            "memory": "memory", 
            "standard": "question_structure_analysis"
        }
    )
    
    # Add conditional edges based on structure type
    workflow.add_conditional_edges(
        "question_structure_analysis",
        lambda state: state["structure_type"],
        {
            "structured": "structured_processing",
            "unstructured": "unstructured_processing"
        }
    )
    
    # Add conditional edges for memory access
    workflow.add_conditional_edges(
        "structured_processing",
        lambda state: "access_memory" if state["contains_memory_query"] else "summarization",
        {
            "access_memory": "access_memory",
            "summarization": "summarization"
        }
    )
    
    workflow.add_conditional_edges(
        "unstructured_processing", 
        lambda state: "access_memory" if state["contains_memory_query"] else "summarization",
        {
            "access_memory": "access_memory",
            "summarization": "summarization"
        }
    )
    
    # All paths lead to summarization before ending
    workflow.add_edge("out_of_scope", "summarization")
    workflow.add_edge("memory", "summarization")
    workflow.add_edge("access_memory", "summarization")
    workflow.add_edge("summarization", END)
    
    return workflow


def run_workflow(user_input: str) -> str:
    """Run the workflow with user input and return the response"""
    
    workflow = create_workflow()
    app = workflow.compile()
    
    # Initial state
    initial_state = {
        "user_input": user_input,
        "question_type": None,
        "structure_type": None,
        "contains_memory_query": False,
        "memory_results": [],
        "processing_results": [],
        "final_response": "",
        "tools_used": [],
        "error": ""
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    return result["final_response"]


if __name__ == "__main__":
    # Test the workflow
    test_input = "How many refund requests did we get?"
    response = run_workflow(test_input)
    print(f"Input: {test_input}")
    print(f"Response: {response}")
