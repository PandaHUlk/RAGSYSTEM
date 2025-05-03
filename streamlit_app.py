import os
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.pipeline.rag_pipeline import RAGPipeline
from src.exception import CustomException
from src.logger import logging

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Multi-Document RAG System",
    page_icon="📚",
    layout="wide",
)

# App title
st.title("Multi-Document RAG System")
st.markdown("Upload PDF documents, automatically tag them, and ask questions about their content.")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_provider = st.sidebar.selectbox(
    "Model Provider",
    options=["ollama", "gemini-ai"],
    help="Select the model provider. GOOGLE AI requires an API key."
)

# Model name selection based on provider
if model_provider == "ollama":
    model_name = st.sidebar.selectbox(
        "Model Name",
        options=["llama2", "mistral", "gemma"],
        help="Select the model to use"
    )
else:  # OpenAI
    model_name = st.sidebar.selectbox(
        "Model Name",
        options=["gemini-2.0-flash", "gemini-pro"],
        help="Select the GOOGLEAI model to use"
    )

# API Key for OpenAI
api_key = None
if model_provider == "gemini-ai":
    # Try to get from environment first
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    # Allow user to enter API key in UI if not in environment
    if not api_key:
        api_key = st.sidebar.text_input("GOOGLE API Key", type="password")
        if not api_key:
            st.sidebar.warning("Please enter an GOOGLE API key or set the GOOGLE_API_KEY environment variable")

# Vector store type
vector_store_type = st.sidebar.selectbox(
    "Vector Store Type",
    options=["faiss", "chroma"],
    help="Select the vector store implementation"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_topics" not in st.session_state:
    st.session_state.document_topics = {}
if "ready" not in st.session_state:
    st.session_state.ready = False

# Initialize RAG pipeline
def initialize_pipeline():
    try:
        st.session_state.rag_pipeline = RAGPipeline(
            model_provider=model_provider,
            model_name=model_name,
            vector_store_type=vector_store_type,
            api_key=api_key
        )
        st.session_state.ready = True
        st.success("RAG Pipeline initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        st.session_state.ready = False

# Initialize button
if st.sidebar.button("Initialize System"):
    with st.spinner("Initializing RAG pipeline..."):
        initialize_pipeline()

# Document uploader section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF documents"
)

# Process uploaded documents
if uploaded_files and st.button("Process Documents"):
    if not st.session_state.ready:
        st.error("Please initialize the system first.")
    else:
        with st.spinner("Processing documents..."):
            try:
                for uploaded_file in uploaded_files:
                    st.session_state.rag_pipeline.process_uploaded_file(uploaded_file)
                
                # Update document topics
                st.session_state.document_topics = st.session_state.rag_pipeline.get_document_topics()
                st.success(f"{len(uploaded_files)} documents processed successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Display document topics
if st.session_state.document_topics:
    st.header("Document Topics")
    for doc_name, topics in st.session_state.document_topics.items():
        with st.expander(f"📄 {doc_name}"):
            for topic in topics:
                st.write(f"- {topic}")

# Chat interface
st.header("Chat with Documents")

# Topic filter
topic_filter = None
source_filter = None

# Add filter options if documents are uploaded
if st.session_state.document_topics:
    col1, col2 = st.columns(2)
    
    # Gather all unique topics and sources
    all_topics = set()
    all_sources = list(st.session_state.document_topics.keys())
    
    for topics in st.session_state.document_topics.values():
        all_topics.update(topics)
    
    with col1:
        topic_filter = st.selectbox(
            "Filter by Topic",
            options=["All Topics"] + sorted(list(all_topics)),
            index=0
        )
        if topic_filter == "All Topics":
            topic_filter = None
    
    with col2:
        source_filter = st.selectbox(
            "Filter by Document",
            options=["All Documents"] + sorted(all_sources),
            index=0
        )
        if source_filter == "All Documents":
            source_filter = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.ready:
    prompt = st.chat_input("Ask a question about your documents")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_pipeline.answer_question(
                        question=prompt,
                        chat_mode=True,
                        filter_topic=topic_filter,
                        filter_source=source_filter
                    )
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("Please initialize the system to start chatting.")

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    if st.session_state.rag_pipeline:
        st.session_state.rag_pipeline.clear_chat_history()
    st.success("Chat history cleared!")


# App footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Initialize the system with your preferred model settings
2. Upload PDF documents
3. Process the documents to extract content and identify topics
4. Ask questions about the documents through the chat interface
5. Optionally filter by topic or document source
""")

if __name__ == "__main__":
    pass