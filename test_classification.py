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
        ("what are all the intents?", "standard"),
        ("what are all the categories?", "standard"),
        ("what are all the values in the category column?", "standard"),
        ("do we have category order with intent other than cancel_order?", "standard"),
        ("which intents exist when category is order?", "standard"),
        ("which categories exist when intent is Obtain invoice?", "standard"),
        ("Show examples of Category account", "standard"),
        ("Show examples of intent", "standard"),
        ("what are the intents of delivery", "standard"),
        ("Show examples of Category contact", "standard"),
        ("Show examples of intent View invoice", "standard"),
        ("Show examples of Category refund", "standard"),
        ("Show examples of Category order", "standard"),
        ("What categories exist?", "standard"),
        ("What intents exist?", "standard"),
        ("which intents exist?", "standard"),
        ("Show intent distributions", "standard"),
        ("Show category distributions", "standard"),
        ("can you give me an example of canceled order?", "standard"),
        ("can you summarize me how do custommers approach us when canceling order", "standard"),
        ("display category distribution", "standard"),
        ("display intent distribution", "standard"),
        ("what is the intent distribution", "standard"),
        ("is there a category refund that its intent is not Review refund policy", "standard"),
        ("i want to see intents", "standard"),
        ("how many people asked to get a refund?", "standard"),
        ("what customers ask or request regarding Newsletter subscription", "standard"),
        ("give examples of customer questions or requests about contact", "standard"),
        ("why the customers cancel orders", "standard"),
        ("whats the complaint that is getting solved least times", "standard"),
        ("can you find requests that have replies which are inadequate", "standard"),
        ("how many people in total contacted us?", "standard"),
        ("how many customers in total sent us questions?", "standard"),
        ("which delivery options customers asked for", "standard"),
        ("what are the shipping methods?", "standard"),
        ("what are the categories and intents?", "standard"),
        ("what are the account types?", "standard"),
        ("to wich accounts users switched?", "standard"),
        ("How do I track the status of my order?", "standard"),

        # follow up quesions 
        ("why are they asking for canceling order", "standard"),
        ("how agents respond to that", "standard"),
        ("what are the main tactics of response?", "standard"),
        
        # Data-related questions
        ("are you connected to a dataset ?", "standard"),
        ("do you have prices in the dataset?", "standard"),
        ("do you have costs in the dataset?", "standard"),
        ("what kind of data do we have in the dataset?", "standard"),
        ("what data do we have?", "standard"),
        ("what is the data", "standard"),
        
        # Scope clarification questions
        ("ok whats in scope then?", "standard"),
        ("any suggestion for a question i can ask you", "standard"),
        
        # Memory questions
        ("What did you tell me before?", "memory"),
        ("Remember the categories we discussed?", "memory"),
        ("Show me more examples from the previous result", "memory"),
        ("What was the last intent you mentioned?", "memory"),
        ("Can you repeat what you said earlier?", "memory"),
        ("What did we talk about previously?", "memory"),
        ("You mentioned something about refunds before, what was it?", "memory"),
        ("From our earlier conversation, what did you say?", "memory"),
        ("What was your previous response?", "memory"),
        ("Tell me again what you told me before", "memory"),
        ("What did you say in our last interaction?", "memory"),
        ("Can you recall our previous discussion?", "memory"),
        ("What was the last thing you mentioned?", "memory"),
        ("From before, what did you tell me?", "memory"),
        ("What did you explain earlier?", "memory"),
        ("Can you remind me what you said before?", "memory"),
        ("What was your earlier response?", "memory"),
        ("From our conversation history, what did you say?", "memory"),
        ("What did you mention in the previous answer?", "memory"),
        ("Can you repeat your earlier explanation?", "memory"),
        ("What did you tell me in the last response?", "memory"),
        ("From what you said before, what was it?", "memory"),
        ("What did you mention earlier about this?", "memory"),

        
        # Out of scope questions
        ("What's the weather like today?", "out_of_scope"),
        ("Tell me about sports news", "out_of_scope"),
        ("How do I cook pasta?", "out_of_scope"),
        ("What's your favorite movie?", "out_of_scope"),
        ("what's your name", "out_of_scope"),
        ("are you inteligent?", "out_of_scope"),
        ("who is serj?", "out_of_scope"),
        ("when are you goint to answer my question?", "out_of_scope"),
        ("when did the customers sent their requests?", "out_of_scope"),
        ("what companies are we working with?", "out_of_scope"),
        ("who are our clients?", "out_of_scope"),
        ("why are you not listening?", "out_of_scope"),
        ("what is the name of our company", "out_of_scope"),
        ("where is our company situated", "out_of_scope"),
        ("how much money are we making", "out_of_scope"),
        ("how many customers do we have", "out_of_scope"),
        ("do u know the name of the company", "out_of_scope"),
        ("ok thanks. can i teach you things and then you'll know them?", "out_of_scope"),
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
