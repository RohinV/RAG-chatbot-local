import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import DOCS_DIR, FAISS_INDEX_PATH, OLLAMA_EMBED_MODEL


def load_documents_from_directory(directory_path: str):

    docs = []
    doc_path = Path(directory_path)

    if not doc_path.exists():
        print(f"[WARN] Directory {doc_path} does not exist.")
        return []

    for file_path in doc_path.iterdir():
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            try:
                pdf_docs = loader.load()
                for pdf in pdf_docs:
                    pdf.metadata["source"] = file_path.name
                docs.extend(pdf_docs)
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {file_path}: {e}")
            

        elif file_path.suffix.lower() == ".txt":
            try:
                loader = TextLoader(str(file_path), encoding="utf8")
                text_docs = loader.load()
                for txt in text_docs:
                    txt.metadata["source"] = file_path.name
                docs.extend(text_docs)
            except Exception as e:
                print(f"[ERROR] Failed to load TXT {file_path}: {e}")
        elif file_path.suffix.lower() == ".md":
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                md_docs = loader.load()
                for md in md_docs:
                    md.metadata["source"] = file_path.name
                docs.extend(md_docs)
            except Exception as e:
                print(f"[ERROR] Failed to load MD {file_path}: {e}")
        
        else:
            print(f"Skipping unsupported file: {file_path.name}")
        
    return docs

def chunk_documents(documents, chunk_size=500, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    chunks = [x for x in chunks if x.page_content.strip()]  # Remove empty chunks

    return chunks

def build_faiss_index(chunks, index_path: str):

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Ensure directory exists
    save_dir = os.path.dirname(index_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save FAISS index
    vectorstore.save_local(index_path)

    print(f"[OK] Saved FAISS index at: {index_path}")



if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents_from_directory(DOCS_DIR)
    print(f"Loaded {len(docs)} raw documents.")

    print("Splitting documents into chunks...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} document chunks.")

    print("Creating FAISS index...")
    build_faiss_index(chunks, FAISS_INDEX_PATH)
    print("FAISS index creation complete.")
    print("Done!")