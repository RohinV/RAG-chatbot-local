from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from config import OLLAMA_EMBED_MODEL, FAISS_INDEX_PATH


def load_vectorestore():

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    vectorestore = FAISS.load_local(FAISS_INDEX_PATH, 
                                    embeddings, 
                                    allow_dangerous_deserialization=True)
    
    return vectorestore


def get_rag_chain(model_name = "llama3:latest"):


    try:
        llm = OllamaLLM(model=model_name, stream=True)
    except:
        llm = OllamaLLM(model=model_name, stream=False)

    vectorstore = load_vectorestore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5
        })

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions using BOTH:
1. Retrieved document context (primary source)
2. Your own general knowledge (secondary source)

Your job is to:
- First check whether the answer can be found in the provided context.
- If the context contains the answer: 
    * Answer based on the context.
    * You MAY add extra correct information from your own knowledge, 
      but clearly mark it as: "(Additional info, NOT from the documents)" before adding it.
- If the context does NOT contain the answer:
    * Answer using your own knowledge.
    * Clearly state, "This information was NOT found in the provided documents.", before your answer.

### STRICT RULES
- Do NOT hallucinate. If you do not know, say so.
- Your answer must be helpful. It should not be concise but also not overly verbose.
- NEVER invent document content.
- NEVER claim something is in the documents unless it truly appears there.
- Do NOT repeat the question.
- Do NOT repeat large passages from context.

### CONTEXT (retrieved from documents)
{context}

### QUESTION
{question}

### ANSWER
""")

   
    extract_question = RunnableLambda(lambda x: x["question"])

    extract_history = RunnableLambda(lambda x: x.get("history", ""))   


    
    rag_inputs = RunnableParallel(
        question = extract_question,
        history = extract_history,
        docs = extract_question | retriever
    )
 
   
    def prepare_prompt(x):

        cleaned_chunks = []
        q_lower = x["question"].strip().lower()

        for doc in x["docs"]:
            text = doc.page_content.strip()

            if len(text) < 20:
                continue
            if text.lower() == q_lower:
                continue
            cleaned_chunks.append(text)
        
        context_text = "\n\n".join(cleaned_chunks) if cleaned_chunks else "No relevant context found."

        return {
            "question": x["question"],
            "history": x["history"],
            "context": context_text
        }

    prep_prompt = RunnableLambda(prepare_prompt)

   
    answer_chain = prep_prompt | prompt | llm

    
    rag_chain = RunnableParallel(
        answer=answer_chain,
        context=RunnableLambda(lambda x: x["docs"])
    )

    # combine inputs and final output
    final_chain = rag_inputs | rag_chain
    return final_chain




def ask(question: str, history: str = "", model_name="llama3:latest"):

    chain = get_rag_chain(model_name)
    result = chain.invoke({"question": question, "history": history})
    return result


if __name__ == "__main__":
    print("\nRAG Chatbot (CLI Mode)\n")

    while True:
        question = input("Ask a question (or type 'exit' or 'quit' to stop): ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        output = ask(question, history)
        answer = output["answer"]

        print("\n Answer:", answer)
        print("\n Retireved Chunks:")
        for doc in output["context"]:
            print("-", doc.metadata.get("source", "Unknown"))
            print(doc.page_content[0:200] + "...\n")
        
        history += f"User: {question}\nAssistant: {answer}\n"