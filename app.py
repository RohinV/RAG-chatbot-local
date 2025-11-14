import streamlit as st
import os
from pathlib import Path
import shutil
import subprocess

from rag import get_rag_chain
from config import DOCS_DIR


st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (Local + FAISS + Ollama)")

st.write("""
This is chatbot application uses your local documents and a local OllamaLLM to answer questions.
Upload your documents on the left sidebar, and then start chatting on the right.
""")


if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.memory = ""
    st.rerun()

st.sidebar.header("🤖 Choose LLM Model")
model_choice = st.sidebar.selectbox(
    "Select an LLM:",
    [
        "llama3:latest",
        "mistral:7b", 
        "qwen3:8b",
        "deepseek-r1:8b"
    ],
    index=0
)


st.sidebar.header("Upload Your Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files to upload (PDF, TXT, MD)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.sidebar.button("Upload and Chat"):
        # Clear Existing docs
        if Path(DOCS_DIR).exists():
            shutil.rmtree(DOCS_DIR)
        os.makedirs(DOCS_DIR, exist_ok=True)

        for file in uploaded_files:
            file_path = Path(DOCS_DIR) / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
        
        # Run the ingest script to rebuild the FAISS index
        st.sidebar.write("⚙️ Rebuilding FAISS index...")
        result = subprocess.run(
            ["python", "ingest.py"],
            capture_output=True,
            text=True,
        )

        # Show output
        st.sidebar.code(result.stdout + "\n" + result.stderr)
        st.sidebar.success("Index built successfully! Refresh to reload chat and ask the bot questions.")


st.sidebar.write("---")
st.sidebar.write("💡 If you add new documents, rebuild the index.\n" \
"Then refresh the page to start asking questions based on the new documents.")
if st.sidebar.button("❌ Close App"):
    st.sidebar.write("Closing the app...Goodbye!")
    os._exit(0)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ""

rag_chain = get_rag_chain(model_choice)

# Main chat interface
st.header("💬Chat with your Documents")
user_input = st.chat_input("Ask a question:")



def doc_info(doc):
    meta = doc.metadata
    return {
        "author": meta.get("author", "Unknown"),
        "title": meta.get("title", "Unknown"),
        "page": meta.get("page_label", meta.get("page", "Unknown"))
    }

if user_input:
    # Create assistant bubble early
    assistant_msg = st.chat_message("assistant")
    assistant_placeholder = assistant_msg.empty()

    streamed_answer = ""
    retrieved_docs = None

    for chunk in rag_chain.stream({
        "question": user_input,
        "history": st.session_state.memory
    }):
        if "answer" in chunk:
            # Append each streaming token
            streamed_answer += chunk["answer"]
            assistant_placeholder.write(streamed_answer)

        if "context" in chunk:
            # context is only sent once at end
            retrieved_docs = chunk["context"]


    st.session_state.memory += f"User: {user_input}\nAssistant: {streamed_answer}\n"

    # Store the chat history
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": streamed_answer,
        "context": retrieved_docs
    })


# Display chat history
for turn in st.session_state.chat_history[:-1]:
    st.chat_message("user").write(turn["question"])
    st.chat_message("assistant").write(turn["answer"])

    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]

    st.chat_message("user").write(last["question"])
    st.chat_message("assistant").write(last["answer"])

    if last.get("context"):
        with st.expander("📄 Sources"):
            for i, doc in enumerate(last["context"]):
                info = doc_info(doc)
                st.markdown(f"""
                **Source {i+1}**
                - **Title:** {info['title']}
                - **Author:** {info['author']}
                - **Page:** {info['page']}
                """)
                st.write("---")

    if turn.get("context"):
        with st.expander("📄 Sources"):
            for i, doc in enumerate(turn["context"]):
                info = doc_info(doc)
                st.markdown(f"""
                **Chunk {i+1}**  
                **Title:** {info['title']}  
                **Author:** {info['author']}  
                **Page:** {info['page']}  
                """)
                st.write("---")