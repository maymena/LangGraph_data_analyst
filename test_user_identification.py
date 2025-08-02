#!/usr/bin/env python3
"""
Test script for user identification logic
"""

from tools.persistent_memory import persistent_memory, get_or_create_user

def test_user_identification():
    """Test the user identification logic"""
    
    print("Testing User Identification Logic")
    print("=" * 50)
    
    # Test existing user
    print("\n1. Testing existing user 'mmbern':")
    user_exists = persistent_memory.user_exists('mmbern')
    print(f"   User exists: {user_exists}")
    
    if user_exists:
        user_record = persistent_memory.load_user_record('mmbern')
        print(f"   Total interactions: {user_record['total_interactions']}")
        print(f"   Created at: {user_record['created_at']}")
        print(f"   Status: Returning User")
    
    # Test non-existing user
    print("\n2. Testing non-existing user 'new_test_user':")
    user_exists = persistent_memory.user_exists('new_test_user')
    print(f"   User exists: {user_exists}")
    
    if not user_exists:
        print("   Status: New User")
        print("   Would create new user record...")
        
        # Actually create the user to test
        user_record = get_or_create_user('new_test_user')
        print(f"   Created user with {user_record['total_interactions']} interactions")
    
    # Test the created user now exists
    print("\n3. Testing newly created user 'new_test_user':")
    user_exists = persistent_memory.user_exists('new_test_user')
    print(f"   User exists: {user_exists}")
    
    if user_exists:
        user_record = persistent_memory.load_user_record('new_test_user')
        print(f"   Total interactions: {user_record['total_interactions']}")
        print(f"   Status: Returning User (on next visit)")
    
    print("\n" + "=" * 50)
    print("User identification logic test completed!")

if __name__ == "__main__":
    test_user_identification()
