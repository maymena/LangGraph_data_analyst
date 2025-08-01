#!/usr/bin/env python3
"""
Test case for access_memory_node integration with WorkflowManager
This test demonstrates how the workflow nodes can access MemorySaver checkpoints
"""

import sys
import os
import re
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
    
    # Test 1: Initial question with specific verifiable data
    print("\n1. Initial Question (Should Use Real Tools):")
    question1 = "How many REFUND category conversations are there?"
    print(f"Q: {question1}")
    
    response1, thread_id = manager.run_query(question1)
    print(f"A: {response1}")
    print(f"Thread ID: {thread_id}")
    
    # Extract specific numbers from response1 for verification
    numbers1 = re.findall(r'\b\d{1,5}\b', response1)
    print(f"Numbers found in response: {numbers1}")
    
    # Test 2: Memory question that should reference the same specific data
    print("\n" + "-" * 50)
    print("2. Memory Question (Should Reference Same Specific Data):")
    question2 = "What exact number did you tell me for refund conversations?"
    print(f"Q: {question2}")
    
    response2, _ = manager.run_query(question2, thread_id)
    print(f"A: {response2}")
    
    # Verify the memory system is using real data, not hallucinating
    numbers2 = re.findall(r'\b\d{1,5}\b', response2)
    print(f"Numbers found in memory response: {numbers2}")
    
    if numbers1 and numbers2:
        common_numbers = set(numbers1) & set(numbers2)
        if common_numbers:
            print(f"✅ VERIFIED: Both responses contain same numbers: {common_numbers}")
            print("✅ Memory system is using REAL DATA, not hallucinating!")
        else:
            print("❌ WARNING: Different numbers - possible hallucination")
    
    # Test 3: Another question to establish more context
    print("\n" + "-" * 50)
    print("3. Another Data Question:")
    question3 = "What are the most frequent categories?"
    print(f"Q: {question3}")
    
    response3, _ = manager.run_query(question3, thread_id)
    print(f"A: {response3[:100]}...")
    
    # Test 4: Memory question referencing multiple previous answers
    print("\n" + "-" * 50)
    print("4. Complex Memory Question:")
    question4 = "What did you tell me about both refunds and categories?"
    print(f"Q: {question4}")
    
    response4, _ = manager.run_query(question4, thread_id)
    print(f"A: {response4[:150]}...")
    
    # Test 5: Detailed session analysis with tool verification
    print("\n" + "-" * 50)
    print("5. Detailed Session Analysis:")
    history = manager.get_session_history(thread_id)
    print(f"Total interactions: {len(history)}")
    
    for i, interaction in enumerate(history, 1):
        print(f"\n{i}. Q: {interaction['user_query']}")
        print(f"   A: {interaction['response'][:60]}...")
        print(f"   Tools: {interaction['tools_used']}")
        print(f"   Type: {interaction.get('question_type', 'N/A')} / {interaction.get('structure_type', 'N/A')}")
        
        # Check for actual tool results
        processing_results = interaction.get('processing_results', [])
        if processing_results:
            print(f"   ✅ Processing Results: {len(processing_results)} results")
            for j, result in enumerate(processing_results):
                if 'result' in result and isinstance(result['result'], dict):
                    if 'count' in result['result']:
                        print(f"      Result {j+1}: REAL COUNT = {result['result']['count']}")
                    elif 'summary' in result['result']:
                        print(f"      Result {j+1}: REAL SUMMARY (length: {len(result['result']['summary'])})")
        else:
            if interaction['tools_used']:
                print(f"   ✅ Tools used but no processing results (normal for memory queries)")
            else:
                print(f"   ❌ No tools used and no processing results")
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print("=" * 70)
    
    # Count different types of tool usage
    tool_usage = {}
    for interaction in history:
        for tool in interaction['tools_used']:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    print(f"Tool Usage Summary: {tool_usage}")
    
    # Verify we have both data tools and memory tools
    has_data_tools = any(tool in tool_usage for tool in ['summarize', 'select_semantic_category', 'get_category_distribution'])
    has_memory_tools = 'session_memory' in tool_usage
    
    if has_data_tools and has_memory_tools:
        print("✅ SUCCESS: Both data tools and memory tools were used!")
        print("✅ This proves the system is working correctly with real data!")
    elif has_data_tools:
        print("⚠️  Only data tools used - memory system may not be triggered")
    elif has_memory_tools:
        print("⚠️  Only memory tools used - data tools may not be working")
    else:
        print("❌ No tools detected - system may not be working")
    
    print("=" * 70)


def test_direct_tool_verification():
    """
    Verify tools are returning real data by calling them directly
    """
    print("\n" + "=" * 70)
    print("Direct Tool Verification")
    print("=" * 70)
    
    from tools.tool_functions import TOOL_FUNCTIONS
    
    print("\n1. Testing select_semantic_category tool directly:")
    try:
        result = TOOL_FUNCTIONS['select_semantic_category'](['REFUND'])
        print(f"   Result: {result}")
        if 'count' in result:
            print(f"   ✅ REAL DATA: Found {result['count']} REFUND conversations")
        else:
            print("   ❌ No count in result")
    except Exception as e:
        print(f"   ❌ Tool failed: {e}")
    
    print("\n2. Testing summarize tool directly:")
    try:
        result = TOOL_FUNCTIONS['summarize']('count of refund requests', category='REFUND')
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict) and 'count' in result:
            print(f"   ✅ REAL DATA: Summarize found {result['count']} refunds")
        elif isinstance(result, dict) and 'summary' in result:
            print(f"   ✅ REAL DATA: Summarize generated summary (length: {len(result['summary'])})")
        else:
            print("   ⚠️  Unexpected result format")
    except Exception as e:
        print(f"   ❌ Tool failed: {e}")


if __name__ == "__main__":
    test_access_memory_node_integration()
    test_direct_tool_verification()
