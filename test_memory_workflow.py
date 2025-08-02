#!/usr/bin/env python3
"""
Test script for LangGraph workflow with checkpoint memory system
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from workflow_manager import run_workflow_with_memory, get_session_history, get_session_summary

def test_memory_workflow():
    """Test the memory-enabled workflow"""
    print("=" * 60)
    print("Testing LangGraph Workflow with Checkpoint Memory")
    print("=" * 60)
    
    # Test 1: Initial question
    print("\n1. Initial Question:")
    question1 = "How many refund requests did we get?"
    print(f"Q: {question1}")
    
    response1, thread_id = run_workflow_with_memory(question1)
    print(f"A: {response1}")
    print(f"Thread ID: {thread_id}")
    
    # Test 2: Follow-up question (should use memory)
    print("\n" + "-" * 40)
    print("2. Follow-up Question (Memory Test):")
    question2 = "Show me more examples from the previous result"
    print(f"Q: {question2}")
    
    response2, _ = run_workflow_with_memory(question2, thread_id)
    print(f"A: {response2}")
    
    # Test 3: Another memory question
    print("\n" + "-" * 40)
    print("3. Another Memory Question:")
    question3 = "What did you tell me before about refunds?"
    print(f"Q: {question3}")
    
    response3, _ = run_workflow_with_memory(question3, thread_id)
    print(f"A: {response3}")
    
    # Test 4: Regular question about order category
    print("\n" + "-" * 40)
    print("4. Regular Question - Summarize Category Order:")
    question4 = "summarize category order"
    print(f"Q: {question4}")
    
    response4, _ = run_workflow_with_memory(question4, thread_id)
    print(f"A: {response4}")
    
    # Test 5: Follow-up question about order category
    print("\n" + "-" * 40)
    print("5. Follow-up Question - Additional Order Sammary:")
    question5 = "display a different summary"
    print(f"Q: {question5}")
    
    response5, _ = run_workflow_with_memory(question5, thread_id)
    print(f"A: {response5}")
    
    # Test 6: Another follow-up question about order category
    print("\n" + "-" * 40)
    print("6. Follow-up Question - Order Analysis:")
    question6 = "What are the main issues customers face?"
    print(f"Q: {question6}")
    
    response6, _ = run_workflow_with_memory(question6, thread_id)
    print(f"A: {response6}")
    
    # Test 7: Another follow-up question about order category
    print("\n" + "-" * 40)
    print("7. Follow-up Question -Elaborate Previous Order Discussion:")
    question7 = "elaborate on that"
    print(f"Q: {question7}")
    
    response7, _ = run_workflow_with_memory(question7, thread_id)
    print(f"A: {response7}")
    
    # Test 8: Another memory question about order summary
    print("\n" + "-" * 40)
    print("8. Memory Question - Order Summary Recall:")
    question8 = "Summarize all the things you told me about orders"
    print(f"Q: {question8}")
    
    response8, _ = run_workflow_with_memory(question8, thread_id)
    print(f"A: {response8}")
    
    # Test 9: Regular question (should work normally)
    print("\n" + "-" * 40)
    print("9. Regular Question:")
    question9 = "What are the most frequent categories?"
    print(f"Q: {question9}")
    
    response9, _ = run_workflow_with_memory(question9, thread_id)
    print(f"A: {response9}")
    
    # Test 10: Show session history
    print("\n" + "-" * 40)
    print("10. Session History:")
    history = get_session_history(thread_id)
    print(f"Total interactions: {len(history)}")
    
    for i, interaction in enumerate(history, 1):
        print(f"\n{i}. Q: {interaction['user_query']}")
        print(f"   A: {interaction['response'][:100]}...")
        print(f"   Tools: {', '.join(interaction['tools_used'])}")
        print(f"   Type: {interaction.get('question_type', 'N/A')} / {interaction.get('structure_type', 'N/A')}")
    
    # Test 11: Session summary
    print("\n" + "-" * 40)
    print("11. Session Summary:")
    summary = get_session_summary(thread_id)
    print(summary)
    
    print("\n" + "=" * 60)
    print("Memory workflow test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_workflow()
