"""
Persistent Memory System for storing user interactions across sessions
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class PersistentMemory:
    """
    Manages persistent storage of user interactions across sessions
    """
    
    def __init__(self, storage_dir: str = "persistent_memory"):
        """
        Initialize persistent memory system
        
        Args:
            storage_dir: Directory to store user memory files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def get_user_file_path(self, username: str) -> Path:
        """Get the file path for a user's persistent memory"""
        # Sanitize username for filename
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
        return self.storage_dir / f"{safe_username}_memory.json"
    
    def user_exists(self, username: str) -> bool:
        """Check if a user has existing persistent memory"""
        return self.get_user_file_path(username).exists()
    
    def create_user(self, username: str) -> Dict[str, Any]:
        """
        Create a new user record
        
        Args:
            username: The username
            
        Returns:
            New user record
        """
        user_record = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "total_interactions": 0,
            "sessions": [],
            "interactions": []
        }
        
        self.save_user_record(username, user_record)
        return user_record
    
    def load_user_record(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Load user's persistent memory record
        
        Args:
            username: The username
            
        Returns:
            User record or None if doesn't exist
        """
        file_path = self.get_user_file_path(username)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                user_record = json.load(f)
            
            # Update last accessed
            user_record["last_accessed"] = datetime.now().isoformat()
            self.save_user_record(username, user_record)
            
            return user_record
            
        except Exception as e:
            print(f"Error loading user record for {username}: {e}")
            return None
    
    def save_user_record(self, username: str, user_record: Dict[str, Any]) -> bool:
        """
        Save user's persistent memory record
        
        Args:
            username: The username
            user_record: The user record to save
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.get_user_file_path(username)
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(user_record, f, indent=2, ensure_ascii=False)
            return True
            
        except Exception as e:
            print(f"Error saving user record for {username}: {e}")
            return False
    
    def add_interaction(self, username: str, interaction: Dict[str, Any]) -> bool:
        """
        Add a new interaction to user's persistent memory
        
        Args:
            username: The username
            interaction: The interaction to add
            
        Returns:
            True if successful, False otherwise
        """
        user_record = self.load_user_record(username)
        if not user_record:
            user_record = self.create_user(username)
        
        # Add timestamp if not present
        if "timestamp" not in interaction:
            interaction["timestamp"] = datetime.now().isoformat()
        
        # Add to interactions list
        user_record["interactions"].append(interaction)
        user_record["total_interactions"] = len(user_record["interactions"])
        user_record["last_accessed"] = datetime.now().isoformat()
        
        return self.save_user_record(username, user_record)
    
    def get_user_interactions(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get user's interaction history
        
        Args:
            username: The username
            limit: Maximum number of interactions to return (most recent first)
            
        Returns:
            List of interactions
        """
        user_record = self.load_user_record(username)
        if not user_record:
            return []
        
        interactions = user_record.get("interactions", [])
        
        # Return most recent first
        interactions = list(reversed(interactions))
        
        if limit:
            interactions = interactions[:limit]
        
        return interactions


# Global instance
persistent_memory = PersistentMemory()


def get_or_create_user(username: str) -> Dict[str, Any]:
    """
    Get existing user or create new one
    
    Args:
        username: The username
        
    Returns:
        User record
    """
    if persistent_memory.user_exists(username):
        return persistent_memory.load_user_record(username)
    else:
        return persistent_memory.create_user(username)


def save_interaction_to_persistent_memory(username: str, interaction: Dict[str, Any]) -> bool:
    """
    Save an interaction to persistent memory
    
    Args:
        username: The username
        interaction: The interaction to save
        
    Returns:
        True if successful
    """
    return persistent_memory.add_interaction(username, interaction)


def get_persistent_memory_context(username: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get persistent memory context for a user
    
    Args:
        username: The username
        limit: Maximum interactions to return
        
    Returns:
        List of interactions
    """
    return persistent_memory.get_user_interactions(username, limit)
