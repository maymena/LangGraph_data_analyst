"""
Streamlit app using LangGraph workflow for Customer Service Dataset Q&A
"""

import streamlit as st
import os
from langgraph_workflow import create_workflow, WorkflowState
from tools import DatasetTools, MemoryTools, LLMTools
from tools.llm_utils import client


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    
    if 'is_identified' not in st.session_state:
        st.session_state.is_identified = False
    
    if 'workflow' not in st.session_state:
        # Initialize LLMTools with Nebius client
        llm_tools = LLMTools(llm_client=client)
        
        # Create workflow with LLMTools
        st.session_state.workflow = create_workflow(llm_tools=llm_tools).compile()
    
    if 'dataset_tools' not in st.session_state:
        st.session_state.dataset_tools = DatasetTools()
    
    if 'memory_tools' not in st.session_state:
        st.session_state.memory_tools = MemoryTools()
    
    if 'llm_tools' not in st.session_state:
        # Initialize LLMTools with Nebius client
        st.session_state.llm_tools = LLMTools(llm_client=client)


def run_langgraph_workflow(user_input: str, user_name: str = "") -> str:
    """Run the LangGraph workflow with user input"""
    
    # Initial state
    initial_state = {
        "user_name": user_name,
        "is_identified": False,
        "user_input": user_input,
        "question_type": None,
        "structure_type": None,
        "contains_memory_query": False,
        "memory_results": [],
        "processing_results": [],
        "final_response": "",
        "tools_used": [],
        "error": "",
        "session_memory": []  # TODO: check if it is compatible with the new memory code
    }
    
    try:
        # Run the workflow
        result = st.session_state.workflow.invoke(initial_state)
        return result["final_response"]
    
    except Exception as e:
        st.error(f"Workflow error: {str(e)}")
        return "I encountered an error processing your request. Please try again."


def main():
    """Main Streamlit app"""
    
    st.set_page_config(
        page_title="Customer Service Dataset Q&A - LangGraph",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Dataset Q&A Agent")
        st.markdown("**LangGraph Workflow Version**")
        
        # User info section
        if st.session_state.is_identified:
            st.markdown("---")
            st.markdown("### Current User")
            st.markdown(f"**Name:** {st.session_state.user_name}")
            
            # TODO: chnage to be compatible with the new memory code
            # Show user stats
            # from tools.user_memory import get_user_stats
            # user_stats = get_user_stats(st.session_state.user_name)
            # if user_stats:
            #     st.markdown(f"**Conversations:** {user_stats.get('conversation_count', 0)}")
            #     st.markdown(f"**Last seen:** {user_stats.get('last_seen', 'Unknown')[:10]}")
        
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("""
        - **Question Classification**: Automatic routing based on question type
        - **Memory System**: Remembers past interactions
        - **Structured Queries**: Counts, distributions, examples
        - **Unstructured Analysis**: Summaries and insights
        - **Out-of-scope Handling**: Polite decline for unrelated questions
        """)
        
        st.markdown("---")
        st.markdown("### Workflow Nodes")
        st.markdown("""
        1. **Question Classification**
        2. **Memory/Out-of-scope/Standard Processing**
        3. **Structure Analysis** (for standard questions)
        4. **Structured/Unstructured Processing**
        5. **Memory Access** (if needed)
        6. **Summarization** (always)
        """)
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - "How many refund requests did we get?"
        - "Show me examples of billing category"
        - "Summarize customer complaints about delivery"
        - "What are the most frequent intents?"
        - "Remember our previous conversation about refunds?"
        """)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory_tools = MemoryTools()
            st.rerun()
        
        # Reset user session button
        if st.button("Reset User Session"):
            st.session_state.user_name = ""
            st.session_state.is_identified = False
            st.session_state.messages = []
            st.session_state.memory_tools = MemoryTools()
            st.rerun()
    
    # Main chat interface
    st.title("Customer Service Dataset Q&A Agent")
    st.markdown("*Powered by LangGraph Workflow*")
    
    # User identification section
    if not st.session_state.is_identified:
        st.markdown("### Please identify yourself to continue")
        
        with st.form("user_identification"):
            user_name = st.text_input("Enter your name:", placeholder="Your name here...")
            submit_button = st.form_submit_button("Start Session")
            
            if submit_button and user_name.strip():
                st.session_state.user_name = user_name.strip()
                st.session_state.is_identified = True
                
                # Run identification workflow
                with st.spinner("Setting up your session..."):
                    response = run_langgraph_workflow("", user_name)
                
                # Add welcome message to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            elif submit_button and not user_name.strip():
                st.error("Please enter your name to continue.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (only show if user is identified)
    if st.session_state.is_identified:
        if prompt := st.chat_input("Ask a question about the customer service dataset..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Processing your question through the workflow..."):
                    response = run_langgraph_workflow(prompt, st.session_state.user_name)
                
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display workflow status (optional debug info)
    with st.expander("Workflow Debug Info", expanded=False):
        st.markdown("### Recent Memory Interactions")
        recent_interactions = st.session_state.memory_tools.get_recent_interactions(5)
        if recent_interactions:
            for i, interaction in enumerate(recent_interactions):
                st.markdown(f"**{i+1}.** {interaction['user_query'][:100]}...")
        else:
            st.markdown("No interactions yet.")
        
        st.markdown("### Dataset Info")
        st.markdown(f"- Total conversations: {len(st.session_state.dataset_tools.df)}")
        st.markdown(f"- Categories: {len(st.session_state.dataset_tools.get_all_categories())}")
        st.markdown(f"- Intents: {len(st.session_state.dataset_tools.get_all_intents())}")


if __name__ == "__main__":
    main()
