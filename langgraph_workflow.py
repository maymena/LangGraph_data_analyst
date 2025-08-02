"""
LangGraph Workflow for Customer Service Dataset Q&A Agent
Refactored from ReAct agent to LangGraph workflow with checkpoint memory
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError
import json
import re
from tools.llm_utils import call_llm
from tools.session_memory import access_session_memory, should_use_session_memory, get_memory_context
from data.download_dataset import load_dataset_df
from tools.tool_functions import TOOL_FUNCTIONS
import datetime
import uuid

df = load_dataset_df()


class ToolCall(BaseModel):
    function_name: str
    function_args: Dict[str, Any] = {}
    answer: Optional[str] = None  # For finish tool


class ToolCallResponse(BaseModel):
    tool_call: ToolCall


class WorkflowState(TypedDict):
    """State object that flows through the workflow"""
    user_name: str  # User identification
    is_identified: bool  # Whether user has been identified
    user_input: str
    question_type: Literal["out_of_scope", "memory", "standard"]
    structure_type: Literal["structured", "unstructured"] 
    contains_memory_query: bool
    memory_results: List[Dict[str, Any]]
    processing_results: List[Dict[str, Any]]
    final_response: str
    tools_used: List[str]
    error: str
    # Memory-related fields
    thread_id: Optional[str]  # Thread ID for memory access
    session_history: Optional[List[Dict[str, Any]]]  # Pre-loaded session history
    # Persistent memory fields
    persistent_context: Optional[List[Dict[str, Any]]]  # Persistent memory context
    user_record: Optional[Dict[str, Any]]  # User record from persistent memory


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

2. "memory" - Questions that ONLY need to recall what was said before (pure memory recall):
   - Direct questions asking to repeat previous responses: "What did you tell me before?", "Repeat what you said"
   - Questions asking about conversation history: "What did we discuss?", "You mentioned something earlier"
   - Simple recall requests: "Tell me again", "What was your last response?"
   Examples: "What did you tell me before about categories?", "You said something about refunds earlier"

3. "standard" - Questions that require analyzing the dataset OR combining memory with other operations:
   - Questions about categories, intents, distributions, counts, examples from the dataset
   - Questions asking for summaries, insights, or analysis of customer service data
   - Questions that combine memory with calculations, analysis, or processing
   - Questions about customer service patterns, agent responses, or dataset content
   - Questions that mention dataset-related terms even if phrased generally
   - Memory + math questions: "What is the previous answer plus 100?"
   - Memory + analysis questions: "Based on what you told me before, show me examples"
   Examples: "How many refund requests?", "What are common categories?", "Summarize delivery issues", "What is the answer of the previous question plus 100?", "Based on the previous analysis, what would you recommend?"

Important: 
- If a question combines memory with ANY other operation (math, analysis, examples), classify as "standard"
- If a question mentions dataset-related terms like "intent", "category", "refund", "order", etc., it should be classified as "standard" even if it uses words like "last" or "previous"
- Only classify as "memory" if the question ONLY wants to recall what was previously said without any processing

User question: "{user_input}"

Respond with ONLY one word: "out_of_scope", "memory", or "standard"."""

    try:
        # Call LLM for classification
        llm_response = call_llm(classification_prompt)
        
        # Extract the final answer from the response (handle <think> tags)
        if "</think>" in llm_response:
            # Extract text after the last </think> tag
            final_answer = llm_response.split("</think>")[-1].strip()
        else:
            final_answer = llm_response.strip()
        
        question_type = final_answer.lower()
        
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
    This node is called when the question is classified as a "memory" type question
    """
    user_input = state["user_input"]
    session_history = state.get("session_history", [])
    persistent_context = state.get("persistent_context", [])
    
    # Combine session and persistent memory
    combined_history = []
    
    # Add persistent memory first (older interactions)
    if persistent_context:
        combined_history.extend(persistent_context)
    
    # Add session memory last (most recent interactions)  
    if session_history:
        combined_history.extend(session_history)
    
    if combined_history:
        # Use the combined history to generate a memory-based response
        response = access_session_memory(user_input, combined_history)
        tools_used = ["session_memory", "persistent_memory"]
        memory_results = [
            {
                "type": "memory_query", 
                "query": user_input,
                "session_history_count": len(session_history),
                "persistent_history_count": len(persistent_context),
                "total_history_count": len(combined_history),
                "response_generated": True
            }
        ]
    else:
        response = "I don't have any previous conversation history to reference."
        tools_used = []
        memory_results = [{"type": "no_memory", "message": "No session or persistent history available"}]
    
    return {
        **state,
        "memory_results": memory_results,
        "final_response": response,
        "tools_used": tools_used
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



def extract_tool_call_from_response(response: str) -> Optional[ToolCallResponse]:
    """
    Extract and validate tool call from LLM response using Pydantic
    """
    try:
        # First try to parse the entire response as JSON
        if response.strip().startswith('{'):
            try:
                data = json.loads(response)
                return ToolCallResponse(**data)
            except (json.JSONDecodeError, ValidationError):
                pass
        
        # Try to find JSON within the response text using regex
        json_patterns = [
            # Pattern 1: Look for complete tool_call JSON objects
            r'\{[^{}]*"tool_call"[^{}]*\{[^{}]*\}[^{}]*\}',
            # Pattern 2: More flexible nested JSON
            r'\{(?:[^{}]|\{[^{}]*\})*"tool_call"(?:[^{}]|\{[^{}]*\})*\}',
            # Pattern 3: Even more flexible with multiple nesting levels
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"tool_call"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    return ToolCallResponse(**data)
                except (json.JSONDecodeError, ValidationError):
                    continue
        
        return None
    except Exception as e:
        print(f"Error extracting tool call: {e}")
        return None


def react_loop(system_prompt: str, user_input: str, allowed_tools: List[str], 
               session_history: List[Dict[str, Any]] = None, 
               persistent_context: List[Dict[str, Any]] = None) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Execute a ReAct loop with the given system prompt, user input, and allowed tools.
    
    Args:
        system_prompt: The system prompt for the LLM
        user_input: The user's question
        allowed_tools: List of tool names that are allowed to be used
        session_history: Session history for memory tool
        
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
        
        # Try to extract tool call using Pydantic
        tool_call_response = extract_tool_call_from_response(response)
        
        if tool_call_response:
            tool_call = tool_call_response.tool_call
            function_name = tool_call.function_name
            function_args = tool_call.function_args
            
            # Check if the tool is allowed
            if function_name not in allowed_tools:
                tool_result = {"error": f"Tool {function_name} is not allowed. Allowed tools: {allowed_tools}"}
            else:
                if function_name not in tools_used:
                    tools_used.append(function_name)
                if function_name == "finish":
                    final_response = tool_call.answer or "No answer provided."
                    break
                # Execute the tool
                if function_name in TOOL_FUNCTIONS:
                    try:
                        # Special handling for memory tool - pass both session and persistent context
                        if function_name == "memory":
                            tool_result = TOOL_FUNCTIONS[function_name](
                                context=session_history, 
                                persistent_context=persistent_context,
                                **function_args
                            )
                        else:
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
            # If no tool call found, treat response as final answer
            final_response = response if isinstance(response, str) else str(response)
            break

    if final_response is None:
        final_response = "Reached maximum number of steps without a final answer."

    return final_response, processing_results, tools_used


def structured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle structured questions using dataset tools via LLM ReAct approach.
    The LLM can use the following tools:
    - select_semantic_intent(intent_name: List[str])
    - select_semantic_category(category_name: List[str])
    - sum_numbers(a: float, b: float)
    - count_category(category: str)
    - count_intent(intent: str)
    - get_intent_distribution(top_n: int = 10, category: Optional[str] = None)
    - get_category_distribution(top_n: int = 10)
    - memory(query: str): Access previous conversation history
    """
    user_input = state["user_input"]
    session_history = state.get("session_history", [])
    persistent_context = state.get("persistent_context", [])

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
        "- memory(query: str): Access previous conversation history to retrieve information from past interactions.\n"
        "\n"
        "CRITICAL: You MUST respond with ONLY a JSON object. Do not include any other text, thinking, or explanations.\n"
        "\n"
        "When you want to use a tool, respond with this exact JSON format:\n"
        "{\n"
        "  \"tool_call\": {\n"
        "    \"function_name\": \"tool_name\",\n"
        "    \"function_args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}\n"
        "  }\n"
        "}\n"
        "\n"
        "When you have the final answer, use the finish tool:\n"
        "{\n"
        "  \"tool_call\": {\n"
        "    \"function_name\": \"finish\",\n"
        "    \"answer\": \"Your final answer here\"\n"
        "  }\n"
        "}\n"
        "\n"
        "Follow these steps:\n"
        "1. Understand the user's question.\n"
        "2. If the question references previous interactions, use the memory tool first to get the relevant information.\n"
        "3. Extract the specific data you need from the memory result (e.g., numbers, categories).\n"
        "4. Use other tools as needed (e.g., sum_numbers for calculations, select_semantic_category for data queries).\n"
        "5. Synthesize the results into a clear answer.\n"
        "6. When you have the final answer, call the finish tool.\n"
        "\n"
        "Examples:\n"
        "- For 'What is the previous answer plus 100?': \n"
        "  Step 1: Call memory('what was the previous answer') to get the number\n"
        "  Step 2: Call sum_numbers(previous_number, 100) to calculate\n"
        "  Step 3: Call finish() with the result\n"
        "- For 'How many refund requests?': Call count_category('REFUND') or select_semantic_category(['REFUND']).\n"
        "\n"
        "The dataset contains customer service conversations with intents and categories.\n"
        "Categories include ACCOUNT, ORDER, REFUND, INVOICE, etc.\n"
        "\n"
        "You are only allowed to use the tools listed above."
    )

    allowed_tools = ["select_semantic_intent", "select_semantic_category", "sum_numbers", "count_category", "count_intent", "get_intent_distribution", "get_category_distribution", "memory", "finish"]
    final_response, processing_results, tools_used = react_loop(system_prompt, user_input, allowed_tools, session_history, persistent_context)

    return {
        **state,
        "processing_results": processing_results,
        "final_response": final_response,
        "tools_used": tools_used
    }


def unstructured_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Handle unstructured questions requiring analysis and summarization using LLM ReAct approach.
    The LLM can use the following tools:
    - select_semantic_intent(intent_name: List[str])
    - select_semantic_category(category_name: List[str])
    - show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None)
    - summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None)
    - show_dataframe(data_type: str = "all", limit: int = 20)
    - memory(query: str): Access previous conversation history
    """
    user_input = state["user_input"]
    session_history = state.get("session_history", [])
    persistent_context = state.get("persistent_context", [])

    system_prompt = (
        "You are an AI assistant that helps answer unstructured questions about a customer service dataset.\n"
        "You have access to the following tools that can query and analyze the dataset:\n"
        "- select_semantic_intent(intent_name: List[str]): Select conversations with specific intents.\n"
        "- select_semantic_category(category_name: List[str]): Select conversations with specific categories.\n"
        "- show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None): Show n example conversations.\n"
        "- summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None): Generate a summary based on user request.\n"
        "- show_dataframe(data_type: str = \"all\", limit: int = 20): Show the dataset as a pandas dataframe.\n"
        "- memory(query: str): Access previous conversation history to retrieve information from past interactions.\n"
        "\n"
        "CRITICAL: You MUST respond with ONLY a JSON object. Do not include any other text, thinking, or explanations.\n"
        "\n"
        "When you want to use a tool, respond with this exact JSON format:\n"
        "{\n"
        "  \"tool_call\": {\n"
        "    \"function_name\": \"tool_name\",\n"
        "    \"function_args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}\n"
        "  }\n"
        "}\n"
        "\n"
        "When you have the final answer, use the finish tool:\n"
        "{\n"
        "  \"tool_call\": {\n"
        "    \"function_name\": \"finish\",\n"
        "    \"answer\": \"Your final answer here\"\n"
        "  }\n"
        "}\n"
        "\n"
        "Follow these steps:\n"
        "1. Understand the user's question (summarize, analyze, show examples, etc.).\n"
        "2. If the question references previous interactions, use the memory tool first.\n"
        "3. Use other tools as needed to gather data and analysis.\n"
        "4. Synthesize the results into a clear, comprehensive answer.\n"
        "5. When you have the final answer, call the finish tool.\n"
        "\n"
        "The dataset contains customer service conversations with intents and categories.\n"
        "Categories include ACCOUNT, ORDER, REFUND, INVOICE, etc.\n"
        "\n"
        "For unstructured questions, focus on providing insights, summaries, and examples rather than just counts or distributions.\n"
        "You are only allowed to use the tools listed above."
    )

    allowed_tools = ["select_semantic_intent", "select_semantic_category", "show_examples", "summarize", "show_dataframe", "memory", "finish"]
    final_response, processing_results, tools_used = react_loop(system_prompt, user_input, allowed_tools, session_history, persistent_context)

    return {
        **state,
        "processing_results": processing_results,
        "final_response": final_response,
        "tools_used": tools_used
    }


def access_memory_node(state: WorkflowState) -> WorkflowState:
    """
    Access memory system when processing nodes need historical context
    This node uses the session_history pre-loaded in the state by WorkflowManager
    """
    user_input = state["user_input"]
    session_history = state.get("session_history", [])
    
    if session_history:
        try:
            # Use the session history to provide memory context
            memory_context = get_memory_context(session_history, "recent")
            
            # Create memory results with relevant context
            memory_results = [
                {
                    "type": "checkpoint_memory",
                    "history_count": len(session_history),
                    "recent_interactions": session_history[-3:] if len(session_history) >= 3 else session_history,
                    "memory_context": memory_context,
                    "user_query": user_input
                }
            ]
            
            # If the user query seems to reference previous interactions,
            # try to find relevant context
            if any(keyword in user_input.lower() for keyword in ["previous", "before", "last", "earlier"]):
                relevant_interactions = []
                for interaction in session_history[-5:]:  # Check last 5 interactions
                    if any(word in interaction.get("user_query", "").lower() for word in user_input.lower().split()):
                        relevant_interactions.append(interaction)
                
                if relevant_interactions:
                    memory_results.append({
                        "type": "relevant_context",
                        "interactions": relevant_interactions
                    })
                    
        except Exception as e:
            memory_results = [{"type": "memory_error", "error": str(e)}]
    else:
        memory_results = [{"type": "no_memory", "message": "No previous interactions found in this session"}]
    
    return {
        **state,
        "memory_results": memory_results
    }


def summarization_node(state: WorkflowState, llm_tools=None) -> WorkflowState:
    """
    Always called before output:
    1. Summarize the results into a coherent response
    2. The interaction is automatically saved by LangGraph checkpoint system
    """
    user_input = state["user_input"]
    processing_results = state.get("processing_results", [])
    memory_results = state.get("memory_results", [])
    tools_used = state.get("tools_used", [])
    
    # If we already have a final response (from out_of_scope or memory nodes), use it
    if state.get("final_response"):
        final_response = state["final_response"]
    else:
        # Use LLMTools to generate a coherent response from all available information
        if llm_tools:
            final_response = llm_tools.summarize(
                user_input=user_input,
                processing_results=processing_results,
                memory_results=memory_results,
                session_memory=session_memory,
                tools_used=tools_used
            )
        else:
            # Fallback to simple response if LLMTools not available
            final_response = f"Analysis complete for: {user_input}. Tools used: {', '.join(tools_used)}"
    
    # The interaction will be automatically saved by the checkpoint system
    # No need to manually manage session_memory here
    
    # Save to persistent user memory
    user_name = state.get("user_name", "")
    if user_name:
        # TODO: change to be compatible with the new memory code
        # from tools.user_memory import save_user_conversation_history
        # save_user_conversation_history(user_name, session_memory)
        pass
    
    return {
        **state,
        "final_response": final_response
    }


def identification_input_node(state: WorkflowState) -> WorkflowState:
    """
    First node in the workflow - handles user identification and persistent memory setup.
    This node runs only once at the start of each session.
    """
    from tools.persistent_memory import get_or_create_user, get_persistent_memory_context
    
    # Get user name from state (will be set by Streamlit app)
    user_name = state.get("user_name", "")
    
    if not user_name or user_name.strip() == "":
        # If no user name provided, ask for identification
        return {
            **state,
            "final_response": "Please enter your name to continue.",
            "is_identified": False
        }
    
    try:
        # Get or create user record in persistent memory
        user_record = get_or_create_user(user_name)
        
        # Load persistent memory context (last 20 interactions)
        persistent_context = get_persistent_memory_context(user_name, limit=20)
        
        # Don't set final_response here - let the workflow continue processing
        # The welcome message logic should be handled elsewhere if needed
        
        return {
            **state,
            "user_name": user_name,
            "persistent_context": persistent_context,
            "user_record": user_record,
            "is_identified": True
        }
        
    except Exception as e:
        return {
            **state,
            "final_response": f"Error setting up your profile: {str(e)}. Please try again.",
            "is_identified": False,
            "error": str(e)
        }


def route_after_identification(state: WorkflowState) -> str:
    """Route after user identification"""
    if state.get("is_identified", False):
        return "input_validation"
    else:
        return "end"


def create_workflow(llm_tools=None) -> StateGraph:
    """Create and configure the LangGraph workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes (renamed to avoid conflicts with state keys)
    workflow.add_node("input_validation", user_input_node)
    workflow.add_node("classify_question", question_classification_node)
    workflow.add_node("handle_out_of_scope", out_of_scope_node)
    workflow.add_node("handle_memory", memory_node)
    workflow.add_node("analyze_structure", question_structure_analysis_node)
    workflow.add_node("process_structured", structured_processing_node)
    workflow.add_node("process_unstructured", unstructured_processing_node)
    workflow.add_node("retrieve_memory", access_memory_node)
    workflow.add_node("identification_input", identification_input_node)
    
    # Create a closure to pass llm_tools to summarization_node
    def summarization_node_with_tools(state: WorkflowState) -> WorkflowState:
        return summarization_node(state, llm_tools)
    
    workflow.add_node("generate_response", summarization_node_with_tools)
    
    # Set entry point to identification node for user identification
    workflow.set_entry_point("identification_input")
    
    # Route from identification to input validation if user is identified
    workflow.add_conditional_edges(
        "identification_input",
        route_after_identification,
        {
            "input_validation": "input_validation",
            "end": END
        }
    )
    
    # Add conditional edge: if identification successful, go to input_validation; else, go to summarization
    workflow.add_conditional_edges(
        "identification_input",
        lambda state: "input_validation" if state.get("is_identified") else "generate_response",
        {
            "input_validation": "input_validation",
            "generate_response": "generate_response" # TODO: think if we need to save this failed interaction
        }
    )
    
    # Add conditional edge: if input is valid, go to question_classification; else, go to summarization
    workflow.add_conditional_edges(
        "input_validation",
        lambda state: "generate_response" if state.get("final_response") else "classify_question",
        {
            "generate_response": "generate_response",
            "classify_question": "classify_question"
        }
    )
    
    # Add conditional edges based on question type
    workflow.add_conditional_edges(
        "classify_question",
        lambda state: state["question_type"],
        {
            "out_of_scope": "handle_out_of_scope",
            "memory": "handle_memory", 
            "standard": "analyze_structure"
        }
    )
    
    # Add conditional edges based on structure type
    workflow.add_conditional_edges(
        "analyze_structure",
        lambda state: state["structure_type"],
        {
            "structured": "process_structured",
            "unstructured": "process_unstructured"
        }
    )
    
    # Add conditional edges for memory access
    workflow.add_conditional_edges(
        "process_structured",
        lambda state: "retrieve_memory" if state["contains_memory_query"] else "generate_response",
        {
            "retrieve_memory": "retrieve_memory",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_conditional_edges(
        "process_unstructured", 
        lambda state: "retrieve_memory" if state["contains_memory_query"] else "generate_response",
        {
            "retrieve_memory": "retrieve_memory",
            "generate_response": "generate_response"
        }
    )
    
    # All paths lead to response generation before ending
    workflow.add_edge("handle_out_of_scope", "generate_response")
    workflow.add_edge("handle_memory", "generate_response")
    workflow.add_edge("retrieve_memory", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow


# Import the existing WorkflowManager

# Global workflow manager instance for persistent memory
_workflow_manager = None

def get_workflow_manager():
    """Get or create the global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager


def run_workflow(user_input: str, thread_id: str = None) -> tuple[str, str]: # TODO: check if `, user_name: str = "", llm_tools=None` is needed
    """
    Run the workflow with user input and return the response
    
    Args:
        user_input: The user's question
        thread_id: Optional thread ID for session continuity. If None, generates a new one.
        
    Returns:
        tuple: (response, thread_id) - The response and the thread ID for session continuity
    """
    manager = get_workflow_manager()
    return manager.run_query(user_input, thread_id)


def get_session_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Get the session history for a given thread ID
    
    Args:
        thread_id: The thread ID to get history for
        
    Returns:
        List of interactions in the session
    """
    manager = get_workflow_manager()
    return manager.get_session_history(thread_id)
    
    try:
        # Get the checkpoint history
        history = []
        for checkpoint in app.get_state_history(config):
            state = checkpoint.values
            if state.get("user_input") and state.get("final_response"):
                history.append({
                    "timestamp": checkpoint.metadata.get("timestamp", ""),
                    "user_query": state["user_input"],
                    "response": state["final_response"],
                    "tools_used": state.get("tools_used", []),
                    "processing_results": state.get("processing_results", [])
                })
        
        # Return in chronological order (oldest first)
        return list(reversed(history))
    except Exception as e:
        print(f"Error getting session history: {e}")
        return []


def clear_session_memory(thread_id: str) -> bool:
    """
    Clear the session memory for a given thread ID
    
    Args:
        thread_id: The thread ID to clear memory for
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Note: MemorySaver doesn't have a direct clear method
        # In a production environment, you might want to use a different checkpointer
        # that supports clearing specific threads
        return True
    except Exception as e:
        print(f"Error clearing session memory: {e}")
        return False


if __name__ == "__main__":
    # Test the workflow with memory
    print("Testing LangGraph workflow with checkpoint memory...")
    
    # Test 1: Initial question
    test_input1 = "How many refund requests did we get?"
    response1, thread_id = run_workflow(test_input1)
    print(f"Input: {test_input1}")
    print(f"Response: {response1}")
    print(f"Thread ID: {thread_id}")
    print("-" * 50)
    
    # Test 2: Follow-up question using the same thread
    test_input2 = "Show me more examples from the previous result"
    response2, _ = run_workflow(test_input2, thread_id)
    print(f"Input: {test_input2}")
    print(f"Response: {response2}")
    print("-" * 50)
    
    # Test 3: Get session history
    history = get_session_history(thread_id)
    print(f"Session history ({len(history)} interactions):")
    for i, interaction in enumerate(history, 1):
        print(f"{i}. Q: {interaction['user_query']}")
        print(f"   A: {interaction['response'][:100]}...")
        print(f"   Tools: {interaction['tools_used']}")
        print()
