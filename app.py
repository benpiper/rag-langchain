import streamlit as st
import rag
import os
import json
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for glassmorphism and modern look
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #134080 0%, #0a192f 100%);
        color: #e6f1ff;
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    .stSidebar {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #64ffda !important;
    }
    .stButton>button {
        background-color: #64ffda;
        color: #0a192f;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Application state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ RAG Settings")

    mode = st.selectbox("Operation Mode", ["agent", "chain"], index=0)
    provider = st.selectbox("Embedding Provider", ["openai", "ollama"], index=0)

    with st.expander("Advanced Settings"):
        k_value = st.slider("Retrieval K", 1, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    if st.button("🔄 Initialize/Refresh System"):
        with st.spinner("Initializing RAG system..."):
            try:
                # Update config dynamically (limited support in this demo)
                rag.config["retrieval"]["k"] = k_value
                rag.config["llm"]["temperature"] = temperature

                rag.setup_vector_store(provider=provider)
                rag.setup_reranker()

                # Try to load existing docs for hybrid search if possible
                # In a real app, we'd handle indexing separately or via UI
                st.session_state.system_ready = True
                st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Initialization failed: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info(
        "Upload documents to the 'docs' folder and refresh the system to index them."
    )

# Main title
st.title("🤖 RAG AI Assistant")
st.markdown("Knowledge-grounded conversations using Multi-source RAG.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask me anything about your documents..."):
    if not st.session_state.system_ready:
        st.warning("Please initialize the system using the sidebar first.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Thinking..."):
                    if mode == "agent":
                        response, _ = rag.run_agent(
                            query,
                            st.session_state.messages[:-1],
                            output_format="markdown",
                        )
                    else:
                        response, _ = rag.run_chain(
                            query,
                            st.session_state.messages[:-1],
                            output_format="markdown",
                        )

                    full_response = response
                    message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = f"I encountered an error: {e}"
                message_placeholder.markdown(full_response)

        # Add assistant message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# Footer
st.divider()
st.caption(f"Powered by LangChain, Milvus, and Streamlit | {datetime.now().year}")
