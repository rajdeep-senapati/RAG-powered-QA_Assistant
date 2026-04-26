import streamlit as st
# from agent import initialize_agent, process_query
from langchain_core.messages import HumanMessage, AIMessage
# from agent import initialize_agent, process_query
from agent import initialize_agent, process_query
st.set_page_config(
    page_title="RAG-Powered Multi-Agent Q&A",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    if "agent" not in st.session_state:
        with st.spinner("Initializing the agent..."):
            st.session_state.agent = initialize_agent()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.title("RAG-Powered Multi-Agent Q&A System")
    
    initialize_session_state()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        This system combines:
        - **RAG**: Retrieval Augmented Generation
        - **Multi-Agent**: Specialized tools for different query types
        - **LLM**: Groq's llama-4-scout model
        """)
        
        st.markdown("## Tools Available")
        st.markdown("""
        - **Document Retrieval**: Searches through ingested documents
        - **Dictionary**: Provides definitions of terms
        - **Calculator**: Performs mathematical calculations
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    # Query input
    query = st.chat_input("Ask a question...")
    
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=query))
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    # Process the query
                    result = process_query(
                        st.session_state.agent, 
                        query, 
                        st.session_state.chat_history[:-1]  # Exclude the just-added message
                    )
                
                # Display details in expandable sections
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Query Analysis"):
                        st.markdown(f"**Query Type**: {result['query_type']}")
                        if result['tools_used']:
                            st.markdown(f"**Tools Used**: {', '.join(result['tools_used'])}")
                
                with col2:
                    if "document_retrieval" in result['tools_used']:
                        with st.expander("Retrieved Contexts", expanded=True):
                            # Extract document retrieval content
                            for step in result["full_result"]["intermediate_steps"]:
                                if step[0].tool == "document_retrieval":
                                    st.markdown(step[1])
                
                # Display the answer
                message_placeholder.markdown(result["answer"])
                
                # Add assistant message to chat history
                st.session_state.chat_history.append(AIMessage(content=result["answer"]))

if __name__ == "__main__":
    main()
