#!/usr/bin/env python3
"""
New Streamlit App for Customer Service Dataset Q&A Agent
Features:
- User identification and persistent memory
- Session management
- Integration with WorkflowManager
"""

import streamlit as st
import uuid
from datetime import datetime
from workflow_manager import WorkflowManager
from tools.persistent_memory import persistent_memory, get_or_create_user, get_persistent_memory_context
import os

# Page configuration
st.set_page_config(
    page_title="Customer Service Q&A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "workflow_manager" not in st.session_state:
    st.session_state.workflow_manager = WorkflowManager()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_identified" not in st.session_state:
    st.session_state.user_identified = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_record" not in st.session_state:
    st.session_state.user_record = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_returning_user" not in st.session_state:
    st.session_state.is_returning_user = False

def check_user_exists(username: str) -> tuple[bool, dict]:
    """
    Check if user exists in persistent memory files
    
    Args:
        username: The username to check
        
    Returns:
        Tuple of (exists, user_record)
    """
    try:
        # Check if user file exists in persistent memory
        user_exists = persistent_memory.user_exists(username)
        
        if user_exists:
            # Load the user record
            user_record = persistent_memory.load_user_record(username)
            return True, user_record
        else:
            return False, None
            
    except Exception as e:
        st.error(f"Error checking user existence: {str(e)}")
        return False, None

def identify_user():
    """Handle user identification process"""
    st.title("🤖 Customer Service Dataset Q&A Agent")
    st.markdown("---")
    
    # User identification form
    with st.form("user_identification"):
        st.subheader("👋 Welcome! Please identify yourself")
        user_name = st.text_input(
            "What's your name?",
            placeholder="Enter your name to continue...",
            help="This helps me maintain your conversation history and provide personalized assistance."
        )
        
        submitted = st.form_submit_button("Continue", type="primary")
        
        if submitted and user_name.strip():
            try:
                # Check if user exists in persistent memory
                user_exists, existing_record = check_user_exists(user_name.strip())
                
                if user_exists and existing_record:
                    # Returning user
                    st.session_state.user_name = user_name.strip()
                    st.session_state.user_record = existing_record
                    st.session_state.user_identified = True
                    st.session_state.is_returning_user = True
                    
                    st.success(f"Welcome back {user_name}! 👋")
                    st.info(f"I found your profile with {existing_record['total_interactions']} previous interactions.")
                    
                else:
                    # New user - create record
                    user_record = get_or_create_user(user_name.strip())
                    
                    st.session_state.user_name = user_name.strip()
                    st.session_state.user_record = user_record
                    st.session_state.user_identified = True
                    st.session_state.is_returning_user = False
                    
                    st.success(f"Welcome {user_name}! 🎉")
                    st.info("You're a new user. I've created a profile for you to remember our conversations.")
                
                # Small delay to show the messages, then rerun
                st.rerun()
                
            except Exception as e:
                st.error(f"Error setting up your profile: {str(e)}. Please try again.")
        
        elif submitted and not user_name.strip():
            st.error("Please enter your name to continue.")

def show_user_info():
    """Show user information in sidebar"""
    with st.sidebar:
        st.markdown("### 👤 User Information")
        st.write(f"**Name:** {st.session_state.user_name}")
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        
        if st.session_state.user_record:
            st.write(f"**Status:** {'Returning User' if st.session_state.is_returning_user else 'New User'}")
            st.write(f"**Total Interactions:** {st.session_state.user_record['total_interactions']}")
            
            # Format dates nicely
            try:
                created_date = datetime.fromisoformat(st.session_state.user_record['created_at']).strftime("%Y-%m-%d %H:%M")
                last_accessed = datetime.fromisoformat(st.session_state.user_record['last_accessed']).strftime("%Y-%m-%d %H:%M")
                st.write(f"**Created:** {created_date}")
                st.write(f"**Last Seen:** {last_accessed}")
            except:
                st.write(f"**Created:** {st.session_state.user_record.get('created_at', 'Unknown')}")
                st.write(f"**Last Seen:** {st.session_state.user_record.get('last_accessed', 'Unknown')}")
        
        st.markdown("---")
        
        # Session management
        st.markdown("### 🔄 Session Management")
        if st.button("New Session", help="Start a new conversation session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("New session started!")
            st.rerun()
        
        if st.button("Clear Chat", help="Clear current conversation"):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
        
        if st.button("Change User", help="Switch to a different user"):
            # Reset user-related session state
            st.session_state.user_identified = False
            st.session_state.user_name = ""
            st.session_state.user_record = None
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.is_returning_user = False
            st.rerun()

def show_welcome_message():
    """Show personalized welcome message based on user status"""
    if st.session_state.is_returning_user:
        welcome_msg = f"Welcome back {st.session_state.user_name}! 👋\n\nI found your profile with {st.session_state.user_record['total_interactions']} previous interactions. I can reference our past conversations if needed.\n\nHow can I help you today?"
    else:
        welcome_msg = f"Welcome {st.session_state.user_name}! 🎉\n\nI'm your customer service data analyst. I can help you analyze the Bitext Customer Service dataset. You can ask me questions about:\n\n- Customer service categories and intents\n- Data distributions and statistics\n- Examples from specific categories\n- Summaries and insights\n\nWhat would you like to know?"
    
    # Add welcome message to chat if it's not already there
    if not st.session_state.messages or st.session_state.messages[0]["content"] != welcome_msg:
        st.session_state.messages.insert(0, {
            "role": "assistant",
            "content": welcome_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

def show_persistent_memory_info():
    """Show information about persistent memory in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💾 Memory System")
        
        if st.session_state.user_record:
            # Show recent interactions count
            recent_interactions = get_persistent_memory_context(st.session_state.user_name, limit=5)
            st.write(f"**Recent interactions:** {len(recent_interactions)}")
            
            if st.button("View Memory", help="Show recent conversation history"):
                with st.expander("Recent Interactions", expanded=True):
                    if recent_interactions:
                        for i, interaction in enumerate(recent_interactions[:3], 1):
                            st.write(f"**{i}.** {interaction.get('user_query', 'Unknown query')[:50]}...")
                            st.caption(f"Response: {interaction.get('response', 'No response')[:100]}...")
                            st.caption(f"Time: {interaction.get('timestamp', 'Unknown')}")
                            st.markdown("---")
                    else:
                        st.write("No previous interactions found.")

def main_chat_interface():
    """Main chat interface after user identification"""
    st.title("🤖 Customer Service Dataset Q&A Agent")
    
    # Show user info in sidebar
    show_user_info()
    show_persistent_memory_info()
    
    # Show welcome message
    show_welcome_message()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"⏰ {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the customer service dataset..."):
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"⏰ {timestamp}")
        
        # Generate response using workflow manager
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the workflow with user identification
                    # Use the actual user name that was provided during identification
                    current_user_name = st.session_state.get("user_name", "")
                    
                    # Debug: Show what user name we're using
                    if not current_user_name:
                        st.error("No user name found in session state!")
                        return
                    
                    response, thread_id = st.session_state.workflow_manager.run_query(
                        user_input=prompt,
                        thread_id=st.session_state.session_id,
                        user_name=current_user_name
                    )
                    
                    # Update session ID if it was generated
                    if thread_id != st.session_state.session_id:
                        st.session_state.session_id = thread_id
                    
                    # Display response
                    st.markdown(response)
                    response_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.caption(f"⏰ {response_timestamp}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": response_timestamp
                    })
                    
                    # Update user record in session state (it gets updated by the workflow)
                    try:
                        updated_record = persistent_memory.load_user_record(current_user_name)
                        if updated_record:
                            st.session_state.user_record = updated_record
                    except:
                        pass  # Don't fail if we can't update the record
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

def show_dataset_info():
    """Show dataset information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        **Bitext Customer Service Dataset**
        - 26,872 conversations
        - Categories: ACCOUNT, ORDER, REFUND, etc.
        - Intents: cancel_order, get_refund, etc.
        - Columns: instruction, category, intent, response
        """)
        
        st.markdown("### 🔧 Features")
        st.markdown("""
        - **Persistent Memory**: Remembers users across sessions
        - **Question Classification**: Handles different query types
        - **Smart Analysis**: Structured and unstructured processing
        - **File-based Storage**: User history saved to disk
        """)

def main():
    """Main application logic"""
    # Show dataset info in sidebar
    show_dataset_info()
    
    # Check if user is identified
    if not st.session_state.user_identified:
        identify_user()
    else:
        main_chat_interface()

if __name__ == "__main__":
    main()
