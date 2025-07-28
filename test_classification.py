#!/usr/bin/env python3
"""
Test script for the question_classification_node implementation
"""

from langgraph_workflow import question_classification_node, WorkflowState

def test_classification():
    """Test the question classification with various inputs"""
    
    test_cases = [
        # Standard questions
        ("How many refund requests did we get?", "standard"),
        ("What are the most common categories?", "standard"),
        ("Show me examples of delivery issues", "standard"),
        ("Summarize the account category", "standard"),
        
        # Memory questions
        ("What did you tell me before?", "memory"),
        ("Remember the categories we discussed?", "memory"),
        ("Show me more examples from the previous result", "memory"),
        ("What was the last intent you mentioned?", "memory"),
        
        # Out of scope questions
        ("What's the weather like today?", "out_of_scope"),
        ("Tell me about sports news", "out_of_scope"),
        ("How do I cook pasta?", "out_of_scope"),
        ("What's your favorite movie?", "out_of_scope"),
    ]
    
    print("Testing question classification...")
    print("=" * 50)
    
    for question, expected in test_cases:
        # Create test state
        test_state = {
            "user_input": question,
            "question_type": None,
            "structure_type": None,
            "contains_memory_query": False,
            "memory_results": [],
            "processing_results": [],
            "final_response": "",
            "tools_used": [],
            "error": ""
        }
        
        # Run classification
        result_state = question_classification_node(test_state)
        actual = result_state["question_type"]
        
        # Check result
        status = "✓" if actual == expected else "✗"
        print(f"{status} Question: {question}")
        print(f"  Expected: {expected}, Got: {actual}")
        print()

if __name__ == "__main__":
    test_classification()
