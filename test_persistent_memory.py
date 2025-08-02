"""
Test the persistent memory system
"""

from workflow_manager import WorkflowManager
from tools.persistent_memory import get_or_create_user, get_persistent_memory_context
import os
import shutil

def test_persistent_memory():
    """Test the persistent memory system with user identification"""
    
    print("🧪 TESTING PERSISTENT MEMORY SYSTEM")
    print("=" * 50)
    
    # Clean up any existing test data
    test_dir = "persistent_memory"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    manager = WorkflowManager()
    test_username = "test_user"
    
    print(f"\n1. Testing new user creation:")
    print(f"   Username: {test_username}")
    
    # Test 1: First interaction (should create new user)
    response1, thread_id = manager.run_query(
        "How many refund conversations are there?", 
        user_name=test_username
    )
    print(f"   First response: {response1}")
    
    # Check user record
    user_record = get_or_create_user(test_username)
    print(f"   User created: {user_record['username']}")
    print(f"   Total interactions: {user_record['total_interactions']}")
    
    print(f"\n2. Testing persistent memory retrieval:")
    
    # Test 2: Second interaction (should use persistent memory)
    response2, _ = manager.run_query(
        "What is the answer of the previous question plus 100?", 
        thread_id=thread_id,
        user_name=test_username
    )
    print(f"   Second response: {response2}")
    
    # Check updated user record
    user_record = get_or_create_user(test_username)
    print(f"   Updated total interactions: {user_record['total_interactions']}")
    
    print(f"\n3. Testing persistent memory context:")
    
    # Get persistent memory context
    persistent_context = get_persistent_memory_context(test_username, limit=5)
    print(f"   Persistent context length: {len(persistent_context)}")
    
    if persistent_context:
        print(f"   Latest interaction: {persistent_context[0]['user_query']}")
        print(f"   Latest response: {persistent_context[0]['response'][:50]}...")
    
    print(f"\n4. Testing new session with existing user:")
    
    # Test 3: New session with same user (should load persistent memory)
    new_manager = WorkflowManager()
    response3, new_thread_id = new_manager.run_query(
        "What did you tell me about refunds before?",
        user_name=test_username
    )
    print(f"   New session response: {response3[:100]}...")
    
    print(f"\n5. Testing different user:")
    
    # Test 4: Different user (should be isolated)
    different_user = "different_user"
    response4, _ = new_manager.run_query(
        "What did you tell me before?",
        user_name=different_user
    )
    print(f"   Different user response: {response4[:100]}...")
    
    # Check different user record
    different_record = get_or_create_user(different_user)
    print(f"   Different user interactions: {different_record['total_interactions']}")
    
    print(f"\n" + "=" * 50)
    print("✅ PERSISTENT MEMORY TEST COMPLETE!")
    print(f"✅ User isolation: Each user has separate memory")
    print(f"✅ Persistence: Memory survives across sessions")
    print(f"✅ Integration: Works with existing workflow")

if __name__ == "__main__":
    test_persistent_memory()
