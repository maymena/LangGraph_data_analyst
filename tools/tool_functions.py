from typing import List, Dict, Any, Optional, Union
import pandas as pd
import openai
import os
from data.download_dataset import load_dataset_df
from tools.memory_tool import memory_tool

# Load the dataset
df = load_dataset_df()

def select_semantic_intent(intent_name: List[str]) -> Dict[str, Any]:
    """
    Select conversations with specific intents.
    
    Args:
        intent_name: List of intent names to select
        
    Returns:
        Dictionary with selected intents, count, and examples
    """
    filtered_df = df[df['intent'].isin(intent_name)]
    
    return {
        "selected_intents": intent_name,
        "count": len(filtered_df),
        "examples": filtered_df.head(3)[['instruction', 'intent', 'response']].to_dict('records')
    }

def select_semantic_category(category_name: List[str]) -> Dict[str, Any]:
    """
    Select conversations with specific categories.
    
    Args:
        category_name: List of category names to select
        
    Returns:
        Dictionary with selected categories, count, and examples
    """
    filtered_df = df[df['category'].isin(category_name)]
    
    return {
        "selected_categories": category_name,
        "count": len(filtered_df),
        "examples": filtered_df.head(3)[['instruction', 'category', 'response']].to_dict('records')
    }

def sum_numbers(a: float, b: float) -> Dict[str, float]:
    """
    Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with the result
    """
    return {"result": a + b}

def count_category(category: str) -> Dict[str, int]:
    """
    Count conversations in a category.
    
    Args:
        category: Category name to count
        
    Returns:
        Dictionary with the count
    """
    count = len(df[df['category'] == category])
    return {"count": count}

def count_intent(intent: str) -> Dict[str, int]:
    """
    Count conversations with an intent.
    
    Args:
        intent: Intent name to count
        
    Returns:
        Dictionary with the count
    """
    count = len(df[df['intent'] == intent])
    return {"count": count}

def show_examples(n: int = 3, intent: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Show n example conversations.
    
    Args:
        n: Number of examples to show
        intent: Optional intent to filter by
        category: Optional category to filter by
        
    Returns:
        Dictionary with formatted examples
    """
    filtered_df = df.copy()
    
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    examples = filtered_df.head(n)[['instruction', 'intent', 'category', 'response']].to_dict('records')
    
    # Format the examples into a readable string
    formatted_examples = f"Found {len(filtered_df)} total conversations"
    if intent:
        formatted_examples += f" with intent '{intent}'"
    if category:
        formatted_examples += f" in category '{category}'"
    formatted_examples += f". Showing {min(n, len(filtered_df))} examples:\n\n"
    
    for i, example in enumerate(examples, 1):
        formatted_examples += f"**Example {i}:**\n"
        formatted_examples += f"**Customer Question:** {example['instruction']}\n"
        formatted_examples += f"**Intent:** {example['intent']}\n"
        formatted_examples += f"**Category:** {example['category']}\n"
        formatted_examples += f"**Agent Response:** {example['response']}\n\n"
    
    return {
        "examples": formatted_examples,
        "total_matching": len(filtered_df),
        "shown": min(n, len(filtered_df))
    }

def summarize(user_request: str, intent: Optional[str] = None, category: Optional[str] = None) -> Dict[str, str]:
    """
    Generate a summary based on the user request using an LLM.
    
    Args:
        user_request: User request to summarize
        intent: Optional intent to summarize
        category: Optional category to summarize
        
    Returns:
        Dictionary with the summary
    """
    from tools.llm_utils import call_llm
    
    # Filter the dataset based on intent and category if provided
    filtered_df = df.copy()
    
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    # Extract relevant data for summarization
    total_count = len(filtered_df)
    
    if total_count == 0:
        return {"summary": "No data found matching the specified criteria."}
    
    # For count requests, return the count directly
    if "count" in user_request.lower():
        return {
            "summary": f"There are {total_count} conversations matching the criteria.",
            "count": total_count,
            "details": f"Found {total_count} conversations" + 
                      (f" with intent '{intent}'" if intent else "") +
                      (f" in category '{category}'" if category else "")
        }
    
    # Sample conversations to send to the LLM for other requests
    # Limit to a reasonable number to avoid token limits
    sample_size = min(20, total_count)
    sample_data = filtered_df.sample(sample_size)[['instruction', 'intent', 'category', 'response']]
    
    # Format the data for the LLM
    formatted_data = ""
    for _, row in sample_data.iterrows():
        formatted_data += f"Customer: {row['instruction']}\n"
        formatted_data += f"Intent: {row['intent']}, Category: {row['category']}\n"
        formatted_data += f"Agent: {row['response']}\n\n"
    
    # Create a prompt for the LLM
    prompt = f"""Based on the following {sample_size} customer service conversations 
{f"with intent '{intent}'" if intent else ""} 
{f"in category '{category}'" if category else ""}
please provide a detailed summary addressing: "{user_request}"

The summary should include:
1. Common patterns in customer queries
2. Typical agent response strategies
3. Key phrases or approaches used by agents
4. Any notable insights about how these conversations are handled

Total conversations matching criteria: {total_count}

Conversations:
{formatted_data}
"""
    
    try:
        # Use the same LLM client as everywhere else
        summary_text = call_llm(prompt)
        
        return {
            "summary": summary_text,
            "total_conversations": total_count,
            "sample_size": sample_size
        }
    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}", "error": str(e)}
def get_intent_distribution(top_n: int = 10, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the distribution of intents in the dataset.
    
    Args:
        top_n: Number of top intents to show
        category: Optional category to filter by
        
    Returns:
        Dictionary with the intent distribution
    """
    filtered_df = df.copy()
    
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    intent_counts = filtered_df['intent'].value_counts().head(top_n).to_dict()
    
    return {
        "intent_distribution": intent_counts,
        "total_conversations": len(filtered_df),
        "filter_category": category
    }

def get_category_distribution(top_n: int = 10) -> Dict[str, Any]:
    """
    Get the distribution of categories in the dataset.
    
    Args:
        top_n: Number of top categories to show
        
    Returns:
        Dictionary with the category distribution
    """
    category_counts = df['category'].value_counts().head(top_n).to_dict()
    
    return {
        "category_distribution": category_counts,
        "total_conversations": len(df)
    }

def show_dataframe(data_type: str = "all", limit: int = 20) -> Dict[str, Any]:
    """
    Show the dataset as a pandas dataframe.
    
    Args:
        data_type: Type of data to show ('all', 'category', 'intent', 'instruction', 'response')
        limit: Maximum number of rows to show
        
    Returns:
        Dictionary with the dataframe data
    """
    if data_type == "all":
        result_df = df.head(limit)
    elif data_type in df.columns:
        result_df = df[[data_type]].head(limit)
    else:
        return {"error": f"Invalid data_type: {data_type}. Valid options are 'all', 'category', 'intent', 'instruction', 'response'"}
    
    # Convert to dict for JSON serialization
    return {
        "dataframe": result_df.to_dict('records'),
        "columns": result_df.columns.tolist(),
        "shape": result_df.shape,
        "data_type": data_type,
        "limit": limit
    }

# Map function names to their implementations
TOOL_FUNCTIONS = {
    "select_semantic_intent": select_semantic_intent,
    "select_semantic_category": select_semantic_category,
    "sum_numbers": sum_numbers,
    "count_category": count_category,
    "count_intent": count_intent,
    "show_examples": show_examples,
    "summarize": summarize,
    "get_intent_distribution": get_intent_distribution,
    "get_category_distribution": get_category_distribution,
    "show_dataframe": show_dataframe,
    "memory": memory_tool,  # Add memory tool
    "finish": lambda answer: {"final_answer": answer}  # Special finish function
}
