import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load keys (Ensure your .env has GOOGLE_API_KEY)
load_dotenv(override=True)

# 2. Load the Local Database (The Memory)
print("🧠 Connecting to Local Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Configure the retriever to fetch the top 5 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# 3. Initialize Google Gemini (The Brain)
print("🤖 Booting up Google Gemini 2.5 Flash...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Create the Financial Analyst Prompt
system_prompt = (
    "You are an elite financial quantitative analyst. Use the following retrieved "
    "context from SEC filings to answer the user's question. If you cannot find "
    "the answer in the context, explicitly state that you do not know. "
    "Keep your answers highly accurate, concise, and professional.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the RAG Pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Execute a Query
query = "What was Tesla's total automotive revenue, and what were the primary factors driving any changes?"

print(f"\n❓ Question: {query}")
print("⏳ Searching database and analyzing context...\n")

# Run the chain
response = rag_chain.invoke({"input": query})

print("✅ Answer:")
print(response["answer"])

# Optional: Print the exact source chunks Gemini used to formulate the answer
print("\n---")
print("📚 Sources Used:")
for i, doc in enumerate(response["context"]):
    print(f"Source {i+1} Snippet: {doc.page_content[:150]}...")