"""
Dataset tools for the LangGraph workflow
Adapted from the original ReAct agent tools
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import load_dataset
import json

class DatasetTools:
    """Tools for querying and analyzing the customer service dataset"""
    
    def __init__(self):
        """Initialize with the Bitext customer service dataset"""
        self.dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        self.df = pd.DataFrame(self.dataset['train'])
    
    def select_semantic_intent(self, intent_names: List[str]) -> List[Dict[str, Any]]:
        """Select conversations with specific intents"""
        filtered_df = self.df[self.df['intent'].isin(intent_names)]
        return filtered_df.to_dict('records')
    
    def select_semantic_category(self, category_names: List[str]) -> List[Dict[str, Any]]:
        """Select conversations with specific categories"""
        filtered_df = self.df[self.df['category'].isin(category_names)]
        return filtered_df.to_dict('records')
    
    def count_category(self, category: str) -> int:
        """Count conversations in a specific category"""
        return len(self.df[self.df['category'] == category])
    
    def count_intent(self, intent: str) -> int:
        """Count conversations with a specific intent"""
        return len(self.df[self.df['intent'] == intent])
    
    def show_examples(self, n: int = 5) -> List[Dict[str, Any]]:
        """Show n example conversations"""
        return self.df.head(n).to_dict('records')
    
    def get_intent_distribution(self, top_n: int = 10) -> Dict[str, int]:
        """Get the distribution of intents"""
        intent_counts = self.df['intent'].value_counts().head(top_n)
        return intent_counts.to_dict()
    
    def get_category_distribution(self, top_n: int = 10) -> Dict[str, int]:
        """Get the distribution of categories"""
        category_counts = self.df['category'].value_counts().head(top_n)
        return category_counts.to_dict()
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories"""
        return self.df['category'].unique().tolist()
    
    def get_all_intents(self) -> List[str]:
        """Get all unique intents"""
        return self.df['intent'].unique().tolist()
    
    def search_conversations(self, query: str, field: str = 'instruction') -> List[Dict[str, Any]]:
        """Search conversations by text content"""
        filtered_df = self.df[self.df[field].str.contains(query, case=False, na=False)]
        return filtered_df.to_dict('records')


class MemoryTools:
    """Tools for managing conversation memory"""
    
    def __init__(self):
        """Initialize memory storage"""
        self.interactions = []  # In-memory storage (could be replaced with persistent storage)
    
    def save_interaction(self, user_query: str, response: str, tools_used: List[str], 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save an interaction to memory"""
        interaction = {
            'timestamp': pd.Timestamp.now(),
            'user_query': user_query,
            'response': response,
            'tools_used': tools_used,
            'metadata': metadata or {}
        }
        self.interactions.append(interaction)
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search past interactions"""
        # Simple text-based search (could be enhanced with embeddings)
        relevant_interactions = []
        for interaction in self.interactions:
            if query.lower() in interaction['user_query'].lower() or \
               query.lower() in interaction['response'].lower():
                relevant_interactions.append(interaction)
        
        return relevant_interactions[-limit:]  # Return most recent matches
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        return self.interactions[-limit:]
    
    def summarize_interactions(self, interactions: List[Dict[str, Any]]) -> str:
        """Generate a summary of interactions"""
        if not interactions:
            return "No relevant past interactions found."
        
        # Simple summarization (could be enhanced with LLM)
        summary_parts = []
        for interaction in interactions:
            summary_parts.append(f"Q: {interaction['user_query'][:100]}...")
            summary_parts.append(f"A: {interaction['response'][:100]}...")
        
        return "\n".join(summary_parts)


class LLMTools:
    """Tools for LLM-based operations"""
    
    def __init__(self, llm_client=None):
        """Initialize with LLM client"""
        self.llm_client = llm_client
    
    def call_llm(self, prompt: str) -> str:
        """Generic LLM calling method"""
        if not self.llm_client:
            return "LLM client not available"
        
        response = self.llm_client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    
    def summarize(self, user_input: str, processing_results: List[Dict[str, Any]], 
                                memory_results: List[Dict[str, Any]], session_memory: List[Dict[str, Any]], 
                                tools_used: List[str]) -> str:
        """
        Generate a coherent, user-friendly response from workflow state information.
        
        Args:
            user_input: The user's original question
            processing_results: Results from processing nodes
            memory_results: Results from memory operations
            session_memory: Recent session interactions
            tools_used: List of tools used in processing
            
        Returns:
            A coherent response string
        """
        if not self.llm_client:
            # Fallback response
            return f"Analysis complete for: {user_input}. Tools used: {', '.join(tools_used)}"
        
        summary_prompt = f"""
        Generate a coherent, user-friendly response based on the following information:

        User Question: {user_input}
        
        Processing Results: {json.dumps(processing_results, indent=2)}
        
        Memory Results: {json.dumps(memory_results, indent=2)}
        
        Session Memory (recent interactions): {json.dumps(session_memory[-3:] if session_memory else [], indent=2)}
        
        Tools Used: {tools_used}

        Please create a clear, comprehensive response that:
        1. Directly answers the user's question
        2. Incorporates relevant information from processing results
        3. References previous context if available in session memory
        4. Is conversational and helpful
        5. Avoids technical jargon unless necessary
        
        Response:
        """
        
        return self.call_llm(summary_prompt)
    
    def classify_question(self, question: str) -> Dict[str, Any]:
        """Classify question type and structure"""
        if not self.llm_client:
            # Fallback to rule-based classification
            return self._rule_based_classification(question)
        
        # TODO: Implement LLM-based classification
        return self._rule_based_classification(question)
    
    def _rule_based_classification(self, question: str) -> Dict[str, Any]:
        """Rule-based question classification"""
        question_lower = question.lower()
        
        # Out of scope detection
        dataset_keywords = ['dataset', 'customer', 'service', 'intent', 'category', 'conversation']
        if not any(keyword in question_lower for keyword in dataset_keywords):
            return {
                'question_type': 'out_of_scope',
                'structure_type': None,
                'contains_memory_query': False
            }
        
        # Memory query detection
        memory_keywords = ['remember', 'previous', 'before', 'earlier', 'last time']
        contains_memory_query = any(keyword in question_lower for keyword in memory_keywords)
        
        if contains_memory_query and not any(keyword in question_lower for keyword in dataset_keywords):
            return {
                'question_type': 'memory',
                'structure_type': None,
                'contains_memory_query': True
            }
        
        # Structure type detection
        structured_keywords = ['how many', 'count', 'distribution', 'examples', 'show me', 'list']
        unstructured_keywords = ['summarize', 'analyze', 'insights', 'explain', 'describe']
        
        if any(keyword in question_lower for keyword in structured_keywords):
            structure_type = 'structured'
        elif any(keyword in question_lower for keyword in unstructured_keywords):
            structure_type = 'unstructured'
        else:
            structure_type = 'structured'  # Default to structured
        
        return {
            'question_type': 'standard',
            'structure_type': structure_type,
            'contains_memory_query': contains_memory_query
        }


# Utility functions
def sum_numbers(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


def finish() -> str:
    """Signal that processing is complete"""
    return "FINISHED"
