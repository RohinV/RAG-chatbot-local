# def get_rag_chain():

#     llm = OllamaLLM(model=OLLAMA_LLM_MODEL)

#     vectorstore = load_vectorestore()
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     extract_question = RunnableLambda(lambda x: x["question"])

#     prompt_template = ChatPromptTemplate.from_template("""
# You are an AI assistant that helps users by providing information based on the context provided from a set of documents.
# Strongly use the following context as much as possible to answer the questions.
# If the answer is not in the context, say "I don't know based on the documents.

# <context>
# {context}
# </context>
                                                       
# Question: {question}
                                                       
# Answer:
# """)
    
#     prepare_inputs = RunnableParallel(
#         context = extract_question | retriever,
#         question = extract_question
#     )

#     def join_docs(x):
#         return{
#             "question": x["question"],
#             "context": "\n\n".join(doc.page_content for doc in x["context"])           
#         }
    
#     format_prompt_inputs = RunnableLambda(join_docs)

#     answer_chain = format_prompt_inputs | prompt_template | llm

#     rag_chain = RunnableParallel(
#         answer=answer_chain,
#         context= prepare_inputs | (lambda x: x["context"])
#     )

#     return rag_chain