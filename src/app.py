"""
Streamlit Chat UI for the RAG Chatbot.
Run with: streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Allow importing from src/generation even though this file lives in src/
sys.path.append(str(Path(__file__).parent))

from generation.rag_pipeline import RAGPipeline


st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource
def load_pipeline():
    """
    Loads the embedding model + connects to ChromaDB once, and reuses it
    across interactions (st.cache_resource keeps it in memory instead of
    reloading on every question, which would be way too slow).
    """
    return RAGPipeline()


st.title("📚 RAG Chatbot")
st.caption("Ask questions about your PDF documents. Answers include source citations.")

with st.spinner("Loading models and connecting to vector database (first load only)..."):
    pipeline = load_pipeline()

# Keep chat history across interactions within a session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View retrieved chunks & sources"):
                for chunk in message["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['filename']}**, page {chunk['page_number']} "
                        f"(distance: {chunk['distance']:.4f})"
                    )
                    st.text(chunk["text"][:300] + "...")
                    st.divider()
                st.caption(f"Retrieval time: {message['retrieval_time']:.2f}s | Total time: {message['total_time']:.2f}s")

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer..."):
            start = time.time()
            chunks = pipeline.retrieve(query)
            retrieval_time = time.time() - start
            answer = pipeline.generate_answer(query, chunks)
            total_time = time.time() - start

        st.markdown(answer)

        # De-duplicated source list
        seen = set()
        sources = []
        for chunk in chunks:
            key = (chunk["filename"], chunk["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append(key)

        st.markdown("**Sources:**")
        for filename, page in sources:
            st.markdown(f"- {filename}, page {page}")

        st.caption(f"⏱️ Retrieval: {retrieval_time:.2f}s | Total: {total_time:.2f}s")

        with st.expander("View retrieved chunks (retrieval visualization)"):
            for chunk in chunks:
                st.markdown(
                    f"**{chunk['filename']}**, page {chunk['page_number']} "
                    f"(distance: {chunk['distance']:.4f})"
                )
                st.text(chunk["text"][:300] + "...")
                st.divider()

    # Save assistant message with metadata for history re-render
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "retrieval_time": retrieval_time,
        "total_time": total_time,
    })