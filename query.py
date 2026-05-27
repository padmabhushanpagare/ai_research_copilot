import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load keys (Ensure your .env has GEMINI_API_KEY)
load_dotenv(override=True)

def get_analyst_response(query):
    # 2. Load the Local Database using the correct HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # We set up the retriever to fetch the top 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Setup the LLM
    # Temperature 0 for strict financial accuracy
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 4. Create the Modern Prompt Template
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

    # 5. The Modern Retrieval Chain
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
        
        print("⏳ Searching database and analyzing context...")
        response = get_analyst_response(user_query)
        
        print("\n--- INSIGHT ---")
        print(response["answer"])
        
        print("\n--- SOURCES ---")
        for i, doc in enumerate(response["context"]):
            print(f"Source {i+1}: {doc.page_content[:150]}...\n")