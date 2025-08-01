#!/usr/bin/env python3
"""
Test case for access_memory_node integration with WorkflowManager
This test demonstrates how the workflow nodes can access MemorySaver checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from workflow_manager import WorkflowManager

def test_access_memory_node_integration():
    """
    Test that access_memory_node can access MemorySaver checkpoints
    when questions go through the full workflow but need memory context
    """
    print("=" * 70)
    print("Testing access_memory_node Integration with MemorySaver")
    print("=" * 70)
    
    manager = WorkflowManager()
    
    # Test 1: Initial question (establishes context)
    print("\n1. Initial Question (Establishes Context):")
    question1 = "What are the most frequent categories?"
    print(f"Q: {question1}")
    
    response1, thread_id = manager.run_query(question1)
    print(f"A: {response1}")
    print(f"Thread ID: {thread_id}")
    print(f"This should go through: classification → structure_analysis → structured_processing")
    
    # Test 2: Follow-up question that should trigger access_memory_node
    print("\n" + "-" * 50)
    print("2. Follow-up Question (Should Trigger access_memory_node):")
    question2 = "Can you show me examples from the categories you mentioned before?"
    print(f"Q: {question2}")
    
    response2, _ = manager.run_query(question2, thread_id)
    print(f"A: {response2}")
    print(f"This should go through: classification → structure_analysis → unstructured_processing → access_memory_node")
    
    # Test 3: Another memory-context question
    print("\n" + "-" * 50)
    print("3. Another Memory-Context Question:")
    question3 = "Based on the previous analysis, what would you recommend?"
    print(f"Q: {question3}")
    
    response3, _ = manager.run_query(question3, thread_id)
    print(f"A: {response3}")
    print(f"This should also use memory context from access_memory_node")
    
    # Test 4: Direct memory question (bypasses workflow)
    print("\n" + "-" * 50)
    print("4. Direct Memory Question (Bypasses Workflow):")
    question4 = "What did you tell me before about categories?"
    print(f"Q: {question4}")
    
    response4, _ = manager.run_query(question4, thread_id)
    print(f"A: {response4}")
    print(f"This should bypass workflow and use session_memory directly")
    
    # Test 5: Show session history and tools used
    print("\n" + "-" * 50)
    print("5. Session Analysis:")
    history = manager.get_session_history(thread_id)
    print(f"Total interactions: {len(history)}")
    
    for i, interaction in enumerate(history, 1):
        print(f"\n{i}. Q: {interaction['user_query']}")
        print(f"   A: {interaction['response'][:80]}...")
        print(f"   Tools: {interaction['tools_used']}")
        print(f"   Type: {interaction.get('question_type', 'N/A')} / {interaction.get('structure_type', 'N/A')}")
    
    # Test 6: Session summary
    print("\n" + "-" * 50)
    print("6. Session Summary:")
    try:
        from workflow_manager import get_session_summary
        summary = get_session_summary(thread_id)
        print(summary)
    except ImportError:
        print("Session summary function not available")
    
    print("\n" + "=" * 70)
    print("Test completed! Check the tools used to see workflow paths.")
    print("=" * 70)


def test_memory_context_flow():
    """
    Test the specific flow of memory context through the workflow
    """
    print("\n" + "=" * 70)
    print("Testing Memory Context Flow Through Workflow Nodes")
    print("=" * 70)
    
    manager = WorkflowManager()
    
    # Step 1: Create initial context
    print("\nStep 1: Creating Initial Context")
    response1, thread_id = manager.run_query("How many ACCOUNT category requests are there?")
    print(f"Response: {response1}")
    
    # Step 2: Question that should use memory context in workflow
    print("\nStep 2: Question That Should Access Memory in Workflow")
    # This question should:
    # 1. NOT be detected as direct memory query by WorkflowManager
    # 2. Go through the full workflow
    # 3. Have contains_memory_query=True
    # 4. Trigger access_memory_node
    # 5. Use memory context in processing
    
    response2, _ = manager.run_query("Show me detailed examples for that category", thread_id)
    print(f"Response: {response2}")
    
    # Step 3: Check the workflow path
    print("\nStep 3: Analyzing Workflow Path")
    history = manager.get_session_history(thread_id)
    
    for i, interaction in enumerate(history, 1):
        print(f"\nInteraction {i}:")
        print(f"  Query: {interaction['user_query']}")
        print(f"  Tools: {interaction['tools_used']}")
        
        # Check if access_memory was used
        if 'access_memory' in str(interaction.get('processing_results', [])):
            print(f"  ✅ access_memory_node was triggered!")
        elif 'session_memory' in interaction['tools_used']:
            print(f"  ✅ Direct session_memory was used (bypassed workflow)")
        else:
            print(f"  ℹ️  Standard workflow path")


if __name__ == "__main__":
    test_access_memory_node_integration()
    test_memory_context_flow()
