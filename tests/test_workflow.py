#!/usr/bin/env python3
"""
Test script for the complete workflow with question classification
"""

import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent directory
os.chdir(parent_dir)

from workflow_manager import WorkflowManager

def test_workflow():
    """Test the complete workflow with various question types"""
    
    test_questions = [
        # Standard questions (dataset analysis)
        # "How many refund requests did we get?",
        # "What are the most common categories?",
        "Show me examples of delivery issues",
        # "what are all the intents?",
        # "what are all the categories?",
        # "Show examples of Category account", 
        # "Show intent distributions",
        # "can you give me an example of canceled order?",
        # "how many people asked to get a refund?",
        # "what customers ask or request regarding Newsletter subscription",
        # "whats the complaint that is getting solved least times",
        # "which delivery options customers asked for",
        # "what are the categories and intents?",
        # "How do I track the status of my order?",
        
        # # Memory questions
        # "What did you tell me before?",  
        # "Remember the categories we discussed?",
        # "Show me more examples from the previous result", 
        # "What was the last intent you mentioned?",
        # "Can you repeat what you said earlier?",
        # "What did we talk about previously?",
        # "You mentioned something about refunds before, what was it?",
        # "From our earlier conversation, what did you say?",
        # "What was your previous response?",
        # "Tell me again what you told me before",
        # "What did you say in our last interaction?",
        # "Can you recall our previous discussion?",
        # "What was the last thing you mentioned?",
        # "From before, what did you tell me?",
        # "What did you explain earlier?",
        # "Can you remind me what you said before?",
        # "What was your earlier response?",
        # "From our conversation history, what did you say?",
        # "What did you mention in the previous answer?",
        # "Can you repeat your earlier explanation?",
        # "What did you tell me in the last response?",
        # "From what you said before, what was it?",
        # "What did you mention earlier about this?",
        
        # # # Out of scope questions
        # "What's the weather like today?",
        # "Tell me about sports news",
        # "How do I cook pasta?",
        # "What's your favorite movie?",
        # "what's your name",
        # "are you inteligent?",
        # "who is serj?",
        # "when are you goint to answer my question?",
        # "when did the customers sent their requests?",
        # "what companies are we working with?",
        # "who are our clients?",
        # "why are you not listening?",
        # "what is the name of our company",
        # "where is our company situated",
        # "how much money are we making",
        # "how many customers do we have",
        # "do u know the name of the company",
        # "ok thanks. can i teach you things and then you'll know them?",
        
        # # Follow-up questions
        # "why are they asking for canceling order",
        # "how agents respond to that",
        # "what are the main tactics of response?",
        
        # # Data-related questions
        # "are you connected to a dataset ?",
        # "do you have prices in the dataset?",
        # "do you have costs in the dataset?",
        # "what kind of data do we have in the dataset?",
        # "what data do we have?",
        # "what is the data",
        
        # # # Scope clarification questions
        # "ok whats in scope then?",
        # "any suggestion for a question i can ask you"
    ]
    
    print("Testing complete workflow...")
    print("=" * 60)
    
    manager = WorkflowManager()
    test_username = "workflow_test_user"
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        try:
            response, thread_id, tools_used = manager.run_query(question, thread_id="test___", user_name=test_username)
            print(f"Tools used: {tools_used}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    test_workflow()
