"""
User Memory Management System
Handles persistent storage of user conversation history
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle


class UserMemoryManager:
    """Manages persistent storage of user conversation history"""
    
    def __init__(self, storage_dir: str = "user_data"):
        """Initialize with storage directory"""
        self.storage_dir = storage_dir
        self.users_file = os.path.join(storage_dir, "users.json")
        self.conversations_dir = os.path.join(storage_dir, "conversations")
        
        # Create directories if they don't exist
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(self.conversations_dir, exist_ok=True)
        
        # Initialize users file if it doesn't exist
        if not os.path.exists(self.users_file):
            self._save_users({})
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users dictionary from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: Dict[str, Dict[str, Any]]) -> None:
        """Save users dictionary to file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def _get_conversation_file(self, user_name: str) -> str:
        """Get the conversation file path for a user"""
        safe_name = user_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return os.path.join(self.conversations_dir, f"{safe_name}.pkl")
    
    def user_exists(self, user_name: str) -> bool:
        """Check if a user exists"""
        users = self._load_users()
        return user_name in users
    
    def create_new_user(self, user_name: str) -> None:
        """Create a new user entry"""
        users = self._load_users()
        
        if user_name not in users:
            users[user_name] = {
                "created_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "conversation_count": 0
            }
            self._save_users(users)
            
            # Initialize empty conversation history
            self.save_user_conversation_history(user_name, [])
    
    def get_user_conversation_history(self, user_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a user"""
        if not self.user_exists(user_name):
            return None
        
        conversation_file = self._get_conversation_file(user_name)
        
        try:
            with open(conversation_file, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return []
    
    def save_user_conversation_history(self, user_name: str, conversation_history: List[Dict[str, Any]]) -> None:
        """Save conversation history for a user"""
        conversation_file = self._get_conversation_file(user_name)
        
        with open(conversation_file, 'wb') as f:
            pickle.dump(conversation_history, f)
        
        # Update user metadata
        users = self._load_users()
        if user_name in users:
            users[user_name]["last_seen"] = datetime.now().isoformat()
            users[user_name]["conversation_count"] = len(conversation_history)
            self._save_users(users)
    
    def add_conversation_to_history(self, user_name: str, conversation: Dict[str, Any]) -> None:
        """Add a new conversation to user's history"""
        history = self.get_user_conversation_history(user_name) or []
        history.append(conversation)
        self.save_user_conversation_history(user_name, history)
    
    def get_all_users(self) -> List[str]:
        """Get list of all registered users"""
        users = self._load_users()
        return list(users.keys())
    
    def get_user_stats(self, user_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a user"""
        users = self._load_users()
        if user_name not in users:
            return None
        
        history = self.get_user_conversation_history(user_name) or []
        stats = users[user_name].copy()
        stats["total_conversations"] = len(history)
        
        return stats


# Global instance
_memory_manager = UserMemoryManager()


# Convenience functions
def user_exists(user_name: str) -> bool:
    """Check if a user exists"""
    return _memory_manager.user_exists(user_name)


def create_new_user(user_name: str) -> None:
    """Create a new user entry"""
    _memory_manager.create_new_user(user_name)


def get_user_conversation_history(user_name: str) -> Optional[List[Dict[str, Any]]]:
    """Get conversation history for a user"""
    return _memory_manager.get_user_conversation_history(user_name)


def save_user_conversation_history(user_name: str, conversation_history: List[Dict[str, Any]]) -> None:
    """Save conversation history for a user"""
    _memory_manager.save_user_conversation_history(user_name, conversation_history)


def add_conversation_to_history(user_name: str, conversation: Dict[str, Any]) -> None:
    """Add a new conversation to user's history"""
    _memory_manager.add_conversation_to_history(user_name, conversation)


def get_all_users() -> List[str]:
    """Get list of all registered users"""
    return _memory_manager.get_all_users()


def get_user_stats(user_name: str) -> Optional[Dict[str, Any]]:
    """Get statistics for a user"""
    return _memory_manager.get_user_stats(user_name) 