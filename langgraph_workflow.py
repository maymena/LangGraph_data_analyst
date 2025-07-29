"""
LangGraph Workflow for Customer Service Dataset Q&A Agent
Refactored from ReAct agent to LangGraph workflow
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
from tools.llm_utils import call_llm
from tools.session_memory import access_session_memory, should_use_session_memory
from data.download_dataset import load_dataset_df
from tools.tool_functions import TOOL_FUNCTIONS
import datetime

df = load_dataset_df()


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
    session_memory: List[Dict[str, Any]]  # Current session interactions


def user_input_node(state: WorkflowState) -> WorkflowState:
    user_input = state.get("user_input", "")
    if not user_input or user_input.strip() == "":
        return {
            **state,
            "final_response": "This is not a valid input. Do you have any question?"
        }
    return state


def question_classification_node(state: WorkflowState) -> WorkflowState:
    """
    Classify the incoming question into one of three types:
    - out_of_scope: Questions not related to the dataset
    - memory: Questions that can be answered from memory alone
    - standard: Questions requiring dataset analysis
    """
    user_input = state["user_input"]
    
    # Create classification prompt for LLM
    classification_prompt = f"""You are a question classifier for a customer service dataset Q&A system. 

The dataset contains customer service conversations with the following columns:
- instruction: Customer queries/requests
- category: Service categories (like 'account', 'order', 'delivery', 'payment', etc.)
- intent: Specific customer intents (like 'cancel_order', 'get_refund', 'track_order', etc.)
- response: Agent responses to customer queries

Classify the following user question into exactly one of these three categories:

1. "out_of_scope" - Questions that are completely unrelated to customer service data analysis:
   - General knowledge questions (weather, cooking, movies, sports, politics)
   - Personal questions not about the dataset
   - Questions about topics outside customer service domain
   Examples: "What's the weather?", "How do I cook pasta?", "Tell me about sports"

2. "memory" - Questions that explicitly reference previous interactions or conversations:
   - Questions using words like "remember", "previous", "before", "earlier", "last time"
   - Questions asking "what did you tell me", "from our conversation", "you mentioned"
   - Follow-up questions referencing prior specific answers or results
   Examples: "What did you tell me before?", "Remember the categories we discussed?", "Show me more examples from the previous result"

3. "standard" - Questions that require analyzing the customer service dataset:
   - Questions about categories, intents, distributions, counts, examples
   - Questions asking for summaries, insights, or analysis of customer service data
   - Questions about customer service patterns, agent responses, or dataset content
   - Questions that mention dataset-related terms even if phrased generally
   Examples: "How many refund requests?", "What are common categories?", "Summarize delivery issues", "What was the last intent you mentioned?" (this refers to dataset intents)

Important: If a question mentions dataset-related terms like "intent", "category", "refund", "order", etc., it should be classified as "standard" even if it uses words like "last" or "previous".

User question: "{user_input}"

Respond with ONLY one word: "out_of_scope", "memory", or "standard"."""

    try:
        # Call LLM for classification
        llm_response = call_llm(classification_prompt)
        question_type = llm_response.strip().lower()
        
        # Validate response and fallback to rule-based if needed
        if question_type not in ["out_of_scope", "memory", "standard"]:
            # Fallback to simple rule-based classification
            question_type = _fallback_classification(user_input)
            
    except Exception as e:
        # If LLM call fails, use fallback classification
        question_type = _fallback_classification(user_input)
    
    return {
        **state,
        "question_type": question_type
    }


def _fallback_classification(user_input: str) -> str:
    """Fallback rule-based classification if LLM fails"""
    user_input_lower = user_input.lower()
    
    # Check for dataset-related keywords first (higher priority)
    dataset_keywords = ["category", "intent", "customer", "service", "agent", "response", 
                       "refund", "order", "delivery", "account", "dataset", "data",
                       "cancel", "track", "payment", "billing", "support"]
    has_dataset_keywords = any(keyword in user_input_lower for keyword in dataset_keywords)
    
    # Check for memory-related keywords
    memory_keywords = ["remember", "previous", "before", "earlier", "last time", 
                      "you said", "we discussed", "from before", "what did you tell me",
                      "you mentioned", "from our conversation"]
    has_memory_keywords = any(keyword in user_input_lower for keyword in memory_keywords)
    
    # If it has dataset keywords, it's standard (even if it has memory keywords)
    if has_dataset_keywords:
        return "standard"
    
    # If it has memory keywords but no dataset keywords, it's memory
    if has_memory_keywords:
        return "memory"
    
    # Check for clearly out-of-scope topics
    out_of_scope_keywords = ["weather", "sports", "politics", "news", "movie", "music", 
                            "recipe", "cook", "travel", "health", "personal", "pasta",
                            "favorite", "hobby"]
    if any(keyword in user_input_lower for keyword in out_of_scope_keywords):
        return "out_of_scope"
    
    # Default to standard if unclear
    return "standard"


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

    # LLM-based structure analysis
    prompt = f"""Decide whether the {user_input} is a structured or unstructured question.\nThe schema of the dataset is: {str(df.dtypes.to_dict())}.\nThe unique values of the category column are: {str(list(df['category'].unique()))}.\nThe unique values of the intent column are: {str(list(df['intent'].unique()))}.\n\nGiven a user question, respond with a string 'structured' or 'unstructured'.\nStructured questions are questions that inquire about the frequency or the values of the categories in the category column or the intent column. For example:\n• 'what are all the categories'\n• 'What categories exist?' \n• 'What are all the values in the category column?'\n• 'What are all the values in the intent column?'\n• 'Show examples of category'\n• 'provide 10 examples of Category order'\n• 'which intents exist when category is account?'\n• 'do we have category order with intent other than cancel order?'\n• 'What are the most frequent categories?' \n• 'What are the most frequent intents?' \n• 'Which intents are most frequent?'\n•  'Which categories are most frequent?'\nunstructured questions are questions that answering them require using examples, insights, analysis or summary of the 'instruction' or 'response' column even if the word instruction or response is not mentioned literally in the user question. For example:\n• 'Summarize Category invoice'\n• 'Show 5 examples of intent View invoice'\n• 'Summarize how agent respond to Intent Delivery options'\n• 'what customers ask or request regarding Newsletter subscription'\n• 'give 6 examples of customer questions about contact'\n• 'can you find requests that have replies which are inadequate' \nRespond with ONLY the word 'structured' or 'unstructured' and nothing else."""
    llm_response = call_llm(prompt)
    structure_type = llm_response.strip().lower()
    if structure_type not in ["structured", "unstructured"]:
        structure_type = "unstructured"

    # LLM-based check for follow-up/memory query
    memory_prompt = f"""Does the following user question refer to previous answers, examples, results, or is it a follow-up to a
     previous question?\nQuestion: {user_input}\n\nRespond with ONLY 'yes' or 'no'. 
     For example, if the user question is "Show me more examples", "What is the total count of the last two intents?", or "what are the intents of the category I ask about", 
     the answer is 'yes' because it is a follow-up to a previous answer or question. 
     If the user question is 'what are the most frequent categories?', the answer is 'no' because it is not a follow-up to a previous question or answer."""
    memory_response = call_llm(memory_prompt)
    contains_memory_query = memory_response.strip().lower() == "yes"

    # TODO: check if we need to also use:
    # followup_phrases = [
    #     "show me more examples", "more examples", "last two", "previous answer", "previous result", "previous examples", "previous intent", "previous category", "previous question", "previous response", "previous data", "previous output", "previously shown", "continue", "as before", "as above", "as previously", "another example", "another result", "another answer"
    # ]
    # contains_memory_query = any(phrase in user_input.lower() for phrase in followup_phrases)


    return {
        **state,
        "structure_type": structure_type,
        "contains_memory_query": contains_memory_query
    }


def react_loop(system_prompt: str, user_input: str, allowed_tools: List[str]) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Execute a ReAct loop with the given system prompt, user input, and allowed tools.
    
    Args:
        system_prompt: The system prompt for the LLM
        user_input: The user's question
        allowed_tools: List of tool names that are allowed to be used
        
    Returns:
        tuple: (final_response, processing_results, tools_used)
    """
    # ReAct loop
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    tools_used = []
    max_steps = 8
    step = 0
    final_response = None
    processing_results = []

    while step < max_steps:
        step += 1
        # Call the LLM
        response = call_llm(prompt=json.dumps(messages))
        try:
            response_message = json.loads(response) if isinstance(response, str) and response.strip().startswith('{') else response
        except Exception:
            response_message = response

        # If the LLM returns a function call (tool use)
        if isinstance(response_message, dict) and "tool_call" in response_message:
            tool_call = response_message["tool_call"]
            function_name = tool_call["function_name"]
            function_args = tool_call.get("function_args", {})
            
            # Check if the tool is allowed
            if function_name not in allowed_tools:
                tool_result = {"error": f"Tool {function_name} is not allowed. Allowed tools: {allowed_tools}"}
            else:
                if function_name not in tools_used:
                    tools_used.append(function_name)
                if function_name == "finish":
                    final_response = tool_call.get("answer", "No answer provided.")
                    break
                # Execute the tool
                if function_name in TOOL_FUNCTIONS:
                    try:
                        tool_result = TOOL_FUNCTIONS[function_name](**function_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}
                else:
                    tool_result = {"error": f"Tool {function_name} not implemented."}
            
            processing_results.append({"tool": function_name, "args": function_args, "result": tool_result})
            # Add tool result to messages for next LLM step
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_result)
            })
        else:
            # If no tool call, treat response as final answer
            final_response = response_message if isinstance(response_message, str) else str(response_message)
            break

    if final_response is None:
        final_response = "Reached maximum number of steps without a final answer."

    return final_response, processing_results, tools_used


def structured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle structured questions using dataset tools via LLM ReAct approach.
    The LLM can use only the following tools:
    - select_semantic_intent(intent_name: List[str])
    - select_semantic_category(category_name: List[str])
    - sum_numbers(a: float, b: float)
    - count_category(category: str)
    - count_intent(intent: str)
    - get_intent_distribution(top_n: int = 10, category: Optional[str] = None)
    - get_category_distribution(top_n: int = 10)
    """
    user_input = state["user_input"]
    session_memory = state.get("session_memory", [])

    # Check if this question should use session memory
    if should_use_session_memory(user_input, session_memory):
        response = access_session_memory(user_input, session_memory)
        return {
            **state,
            "final_response": response,
            "tools_used": ["session_memory"]
        }

    system_prompt = (
        "You are an AI assistant that helps answer questions about a customer service dataset.\n"
        "You have access to the following tools that can query and analyze the dataset:\n"
        "- select_semantic_intent(intent_name: List[str]): Select conversations with specific intents.\n"
        "- select_semantic_category(category_name: List[str]): Select conversations with specific categories.\n"
        "- sum_numbers(a: float, b: float): Add two numbers.\n"
        "- count_category(category: str): Count conversations in a category.\n"
        "- count_intent(intent: str): Count conversations with an intent.\n"
        "- get_intent_distribution(top_n: int = 10, category: Optional[str] = None): Get the distribution of intents.\n"
        "- get_category_distribution(top_n: int = 10): Get the distribution of categories.\n"
        "\n"
        "Follow these steps:\n"
        "1. Understand the user's question.\n"
        "2. Determine which tools you need to use to answer the question.\n"
        "3. Call the appropriate tools with the right parameters.\n"
        "4. Synthesize the results into a clear answer.\n"
        "5. When you have the final answer, call the finish tool.\n"
        "\n"
        "The dataset contains customer service conversations with intents and categories.\n"
        "The dataset includes various intents like edit_account, switch_account, check_invoice, etc.\n"
        "Categories include ACCOUNT, ORDER, REFUND, INVOICE, etc.\n"
        "\n"
        "For out-of-scope questions not related to the dataset, politely explain that you can only answer questions about the customer service dataset.\n"
        "You are only allowed to use the tools listed above."
    )

    allowed_tools = ["select_semantic_intent", "select_semantic_category", "sum_numbers", "count_category", "count_intent", "get_intent_distribution", "get_category_distribution"]
    final_response, processing_results, tools_used = react_loop(system_prompt, user_input, allowed_tools)

    return {
        **state,
        "processing_results": processing_results,
        "final_response": final_response,
        "tools_used": tools_used
    }


def unstructured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle unstructured questions requiring analysis and summarization using LLM ReAct approach.
    The LLM can use only the following tools:
    - select_semantic_intent(intent_name: List[str])
    - select_semantic_category(category_name: List[str])
    - show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None)
    - summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None)
    - show_dataframe(data_type: str = "all", limit: int = 20)
    """
    user_input = state["user_input"]
    session_memory = state.get("session_memory", [])

    # Check if this question should use session memory
    if should_use_session_memory(user_input, session_memory):
        response = access_session_memory(user_input, session_memory)
        return {
            **state,
            "final_response": response,
            "tools_used": ["session_memory"]
        }

    system_prompt = (
        "You are an AI assistant that helps answer unstructured questions about a customer service dataset.\n"
        "You have access to the following tools that can query and analyze the dataset:\n"
        "- select_semantic_intent(intent_name: List[str]): Select conversations with specific intents.\n"
        "- select_semantic_category(category_name: List[str]): Select conversations with specific categories.\n"
        "- show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None): Show n example conversations.\n"
        "- summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None): Generate a summary based on user request.\n"
        "- show_dataframe(data_type: str = \"all\", limit: int = 20): Show the dataset as a pandas dataframe.\n"
        "\n"
        "Follow these steps:\n"
        "1. Understand the user's question (summarize, analyze, show examples, etc.).\n"
        "2. Determine which tools you need to use to answer the question.\n"
        "3. Call the appropriate tools with the right parameters.\n"
        "4. Synthesize the results into a clear, comprehensive answer.\n"
        "5. When you have the final answer, call the finish tool.\n"
        "\n"
        "The dataset contains customer service conversations with intents and categories.\n"
        "The dataset includes various intents like edit_account, switch_account, check_invoice, etc.\n"
        "Categories include ACCOUNT, ORDER, REFUND, INVOICE, etc.\n"
        "\n"
        "For unstructured questions, focus on providing insights, summaries, and examples rather than just counts or distributions.\n"
        "You are only allowed to use the tools listed above."
    )

    allowed_tools = ["select_semantic_intent", "select_semantic_category", "show_examples", "summarize", "show_dataframe"]
    final_response, processing_results, tools_used = react_loop(system_prompt, user_input, allowed_tools)

    return {
        **state,
        "processing_results": processing_results,
        "final_response": final_response,
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
    session_memory = state.get("session_memory", [])
    tools_used = state.get("tools_used", [])
    
    # If we already have a final response (from out_of_scope or memory nodes), use it
    if state.get("final_response"):
        final_response = state["final_response"]
    else:
        # Use LLM to generate a coherent response from all available information
        summary_prompt = f"""
        Generate a coherent, user-friendly response based on the following information:

        User Question: {user_input}
        
        Processing Results: {json.dumps(processing_results, indent=2)}
        
        Memory Results: {json.dumps(memory_results, indent=2)}
        
        Session Memory (recent interactions): {json.dumps(session_memory[-3:] if session_memory else [], indent=2)}
        
        Tools Used: {tools_used}

        Please create a clear, comprehensive response that:
        1. Directly answers the user's question
        2. Incorporates relevant information from processing results
        3. References previous context if available in session memory
        4. Is conversational and helpful
        5. Avoids technical jargon unless necessary
        
        """
        
        final_response = call_llm(prompt=summary_prompt)
    
    # Save interaction to session memory
    interaction = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_query": user_input,
        "response": final_response,
        "tools_used": tools_used,
        "processing_results": processing_results,
        "memory_results": memory_results
    }
    session_memory.append(interaction)
    
    return {
        **state,
        "final_response": final_response,
        "session_memory": session_memory
    }


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("question_classification", question_classification_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("question_structure_analysis", question_structure_analysis_node)
    workflow.add_node("structured_processing", structured_processing_node)
    workflow.add_node("unstructured_processing", unstructured_processing_node)
    workflow.add_node("access_memory", access_memory_node)
    workflow.add_node("summarization", summarization_node)
    
    # Set entry point to user_input node
    workflow.set_entry_point("user_input")
    
    # Add conditional edge: if input is valid, go to question_classification; else, go to summarization
    workflow.add_conditional_edges(
        "user_input",
        lambda state: "summarization" if state.get("final_response") else "question_classification",
        {
            "summarization": "summarization",
            "question_classification": "question_classification"
        }
    )
    
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


def run_workflow(user_input: str, session_memory: List[Dict[str, Any]] = None) -> str:
    """Run the workflow with user input and return the response"""
    
    if session_memory is None:
        session_memory = []
    
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
        "error": "",
        "session_memory": session_memory
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
