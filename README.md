# 🤖 RAG Chatbot — Local Document Q&A with FAISS + Ollama

A fully local **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents (PDF, TXT, MD), embed them using **Ollama embeddings**, index them with **FAISS**, and ask questions using local open-source LLMs like:

- DeepSeek R1
- LLaMA 3
- Qwen 3
- Mistral 7B

Everything runs locally — no external APIs.

---

## 🚀 Features

### 🔍 Document-Aware Q&A
Ask questions based on your own uploaded documents.

### 📚 Local Vector Search
High-performance similarity search powered by **FAISS**.

### 🧠 Local LLM Inference
Uses Ollama to run all AI models locally.

### 🧩 Model Switching
Choose from multiple models:
- `deepseek-r1:8b`
- `llama3:latest`
- `qwen3:8b`
- `mistral:7b`

### 🧷 Intelligent RAG Prompting
- Avoids hallucination  
- Adds LLM general knowledge when needed  
- Clearly states when info is *not* from documents  
- Clear but not overly verbose answers  

### 🗂️ Document Upload & Memory
Upload PDFs/TXT/MD and chat with them persistently.

### 🖥️ Streamlit UI
Clean and user-friendly interface.

---

## 📦 Project Structure

```
rag_chatbot/
│── app.py                # Streamlit UI
│── rag.py                # RAG pipeline: LLM + retriever + prompt
│── ingest.py             # Document ingestion & FAISS index creation
│── config.py             # Paths and model configuration
│── requirements.txt      # Python dependencies
│── environment.yml       # Conda environment file
│── faiss_store/          # Local FAISS index
│── docs/                 # Uploaded documents
│── README.md
│── .gitignore
```

---

## 🛠️ Installation

### 1️⃣ Create Conda Environment

```bash
conda env create -f environment.yml
conda activate rag_chatbot
```

Or install via pip:

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Install Ollama

Download Ollama:

https://ollama.com/download

Then pull the models you want:

```bash
ollama pull deepseek-r1:8b
ollama pull llama3
ollama pull qwen3:8b
ollama pull mistral:7b
```

---

## 📥 Ingest Documents

Place your documents in the `docs/` folder or upload them in the Streamlit sidebar.

Then run:

```bash
python ingest.py
```

This will:
- Load documents  
- Split into text chunks  
- Create embeddings  
- Build a FAISS index  

---

## 💬 Run the Chatbot

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 🧠 How It Works (RAG Pipeline)

1. Load documents (PDF/TXT/MD)  
2. Split text using recursive chunking  
3. Generate embeddings using Ollama  
4. Store embeddings in FAISS  
5. Retrieve relevant chunks  
6. Pass chunks + question into your selected LLM  
7. LLM answers while distinguishing:
   - Document-based info  
   - General knowledge  

---

## 🧪 Example Q&A

**Q:**  
*What is Adam optimization?*

**A:**  
Adam is a stochastic gradient optimization method that combines ideas from AdaGrad and RMSProp.  
*(This information was found in your documents.)*

Additionally, based on general LLM knowledge, Adam is widely used in neural network training due to adaptive learning rates.  
*(This part is not from your documents.)*

---

## 🛠️ Troubleshooting

### ❌ FAISS index not found
Run:
```bash
python ingest.py
```

### ❌ Model not found
Pull the model:
```bash
ollama pull deepseek-r1:8b
```

### ❌ Streamlit not updating
Restart:
```bash
streamlit run app.py
```

---

## 🧾 License
This project is open-source and free to use or modify.

---

## ⭐ Optional Enhancements
- Chat history saving  
- Dockerfile  
- Hybrid search (FAISS + BM25)  
- GPU acceleration  
- UI redesign  
- Conversation memory

