import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  
# UPDATE: Import from langchain_classic instead of langchain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_analyst_response(query):
    # 1. Load the existing database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # We set up the retriever to fetch the top 5 most relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # 2. Setup the LLM (Gemini 1.5 Pro)
    # Temperature 0 for strict financial accuracy
    # UPDATE: Using the 2026 stable flagship model
    # UPDATE: Using 1.5 Flash to bypass the strict Pro quota limits
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 3. Create the Modern Prompt Template
    system_prompt = (
        "You are a Senior Financial Research Assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Keep the answer concise and professional. Use bullet points for data.\n\n"
        "CONTEXT:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. The Modern Retrieval Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the chain
    return rag_chain.invoke({"input": query})

if __name__ == "__main__":
    print("📈 AI Financial Copilot Active. Ask about the report (or type 'exit')")
    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'exit':
            break
        
        response = get_analyst_response(user_query)
        print("\n--- INSIGHT ---")
        # Note: the modern chain returns the response under the key "answer"
        print(response["answer"])
        
        # PRO PORTFOLIO TIP: Show citations
        print("\n--- SOURCES ---")
        # Note: the modern chain returns the docs under the key "context"
        for doc in response["context"]:
            print(f"-> {doc.page_content[:100]}...\n")