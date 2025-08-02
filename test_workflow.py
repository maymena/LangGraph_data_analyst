#!/usr/bin/env python3
"""
Test script for the complete workflow with question classification
"""

from workflow_manager import WorkflowManager

def test_workflow():
    """Test the complete workflow with various question types"""
    
    test_questions = [
        "How many refund requests did we get?",
        "What's the weather like today?",
        "Remember what you told me before?",
        "What are the most common categories in the dataset?"
    ]
    
    print("Testing complete workflow...")
    print("=" * 60)
    
    manager = WorkflowManager()
    test_username = "workflow_test_user"
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        try:
            response, thread_id = manager.run_query(question, thread_id="test___", user_name=test_username)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    test_workflow()
