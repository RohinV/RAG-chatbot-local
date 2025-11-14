🤖 RAG Chatbot — Local Document Q&A with FAISS + Ollama

A fully local Retrieval-Augmented Generation (RAG) chatbot that allows you to upload documents (PDF, TXT, MD), embed them using Ollama embeddings, index them with FAISS, and query them through a local open-source LLM such as:

DeepSeek R1

LLaMA 3

Qwen 3

Mistral 7B

All models run locally through Ollama, with no external API calls.

🚀 Features
🔍 Document-Aware Q&A

Ask questions directly about your PDFs, notes, and text documents.

📚 Local Vector Search

Uses FAISS for high-performance similarity search.

🧠 Local LLM Inference

Runs fully offline with Ollama + open-source models.

🧩 Model Switching

Choose your preferred LLM:

deepseek-r1:8b

llama3:latest

qwen3:8b

mistral:7b

🧷 Context-Aware Prompting

The RAG prompt ensures:

No hallucination

Answers clearly distinguish between document-based facts and LLM prior knowledge

Answers are clear but not overly verbose

Model acknowledges when info is not from documents

🗂️ Document Memory

The app keeps track of your previous questions and responses.

🖥️ Streamlit UI

User-friendly interface for uploading files and chatting.

⚡ Fully Local & Open Source

No network use.
No API fees.
All modules open-source.

📦 Project Structure
rag_chatbot/
│── app.py                 # Streamlit UI
│── rag.py                 # RAG pipeline (LLM + retriever + prompt)
│── ingest.py              # Builds FAISS index from documents
│── config.py              # Paths and model config
│── requirements.txt       # pip dependencies
│── environment.yml        # Conda environment (optional)
│── README.md              # You are here
│── docs/                  # Uploaded user documents
│── faiss_store/           # Saved FAISS index
│── .gitignore

🛠️ Installation
1️⃣ Install Conda Environment (Recommended)

Create your environment:

conda env create -f environment.yml
conda activate rag_chatbot


Or generate it manually:

conda create -n rag_chatbot python=3.10
conda activate rag_chatbot
pip install -r requirements.txt

2️⃣ Install Ollama

Download and install Ollama:

👉 https://ollama.com/download

Then pull the models you want:

ollama pull deepseek-r1:8b
ollama pull llama3
ollama pull qwen3:8b
ollama pull mistral:7b

📥 Ingest Documents

Place your PDFs/TXT/MD into the docs/ folder
OR upload them through the Streamlit sidebar.

Then run:

python ingest.py


This will:

Load your documents

Chunk text

Generate embeddings

Build a local FAISS index

💬 Run the Chatbot UI

Start Streamlit:

streamlit run app.py


Then open:

👉 http://localhost:8501

🧠 How It Works (RAG Pipeline)
1. Document Loading

PDFs → text via PyPDFLoader
TXT/MD → loaded directly

2. Text Chunking

Using RecursiveCharacterTextSplitter (LangChain v1).

3. Embeddings

Generated via OllamaEmbeddings(model="nomic-embed-text") (or your chosen embed model).

4. Vector Index

Stored locally using FAISS.

5. Query Flow

User Question → Embedding → FAISS Retrieval → Prompt → LLM → Final Answer

🧪 Example Query
Q:

“What is Adam optimization?”

A:
Adam is a stochastic gradient-based optimization method that blends ideas from
AdaGrad and RMSProp. This information was found in your uploaded documents.
Additionally, based on general LLM knowledge (not in your documents), Adam is 
widely used due to its adaptive learning rate and momentum.

🛠️ Troubleshooting
❌ FAISS index not found

Run:

python ingest.py

❌ Model not found

Pull it with Ollama:

ollama pull deepseek-r1:8b

❌ Streamlit fails to load

Close all terminals and restart:

streamlit run app.py

❌ “I don't know based on your documents” too often

Increase retrieval depth in rag.py:

search_kwargs={"k": 8, "fetch_k": 16}

🧾 License

This project is fully open-source and free to use or modify.