import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def get_hybrid_retriever(k=5):
    """
    Builds an Enterprise-Grade Hybrid Retriever.
    Combines ChromaDB (Semantic/Dense) with BM25 (Keyword/Sparse).
    """
    print("⚙️ Booting Hybrid Search Engine (BM25 + Chroma)...")
    
    # 1. Setup the Dense (Semantic) Retriever
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # 2. Extract existing chunks from Chroma to feed the BM25 algorithm
    # BM25 operates in-memory, so it needs the raw text.
    db_data = vectordb.get()
    
    if not db_data['documents']:
        raise ValueError("ChromaDB is empty! Please ingest a PDF first.")
        
    documents = [
        Document(page_content=txt, metadata=meta or {})
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    
    # 3. Setup the Sparse (Keyword) Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # 4. The Ensemble (The Hybrid Brain)
    # Weights: 50% Importance to Exact Keywords, 50% to Semantic Meaning
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever