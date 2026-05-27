import os
import time
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

load_dotenv()

# --- 1. THE STATE (The Agent's Memory) ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    retry_count: int  # Added to prevent infinite loops

# --- 2. THE GRADING SCHEMA ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# --- 3. NODE 1: The Retriever ---
def retrieve(state: GraphState):
    """
    Retrieve documents from ChromaDB based on the question.
    """
    print("\n🤖 [NODE: Retriever] -> Hunting for documents...")
    question = state["question"]
    
    # 🌟 UPDATED TO MATCH YOUR LOCAL DATABASE
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    documents = retriever.invoke(question)
    print(f"   -> Found {len(documents)} chunks.")
    
    return {"documents": documents, "question": question, "retry_count": 0}

# --- 4. NODE 2: The Grader (BATCH OPTIMIZED) ---
def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant.
    Optimized to grade all documents in a single API call to bypass strict free-tier limits.
    """
    print("\n🤖 [NODE: Grader] -> Checking context relevance (Batch Mode)...")
    question = state["question"]
    documents = state["documents"]
    
    # Setup our LLM Grader
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    # The grading prompt updated for a batch context
    system_prompt = """You are a strict grader assessing relevance of a retrieved context to a user question. 
    If the context contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the context as a whole is relevant."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved context: \n\n {context} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # Combine all 5 documents into one string to save API calls!
    context_string = "\n\n".join(doc.page_content for doc in documents)
    
    # One single API call instead of 5
    score = retrieval_grader.invoke({"question": question, "context": context_string})
    
    if score.binary_score.lower() == "yes":
        print("   -> Context Passed ✅")
        # Give the API a tiny 2-second breather before moving to the Generator
        time.sleep(2) 
        return {"documents": documents, "question": question}
    else:
        print("   -> Context Failed ❌ (Discarding to prevent hallucination)")
        return {"documents": [], "question": question}

# --- 5. THE HALLUCINATION SCHEMA ---
class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# --- 6. NODE 3: The Generator ---
def generate(state: GraphState):
    """
    Generate answer using the filtered documents.
    """
    print("\n🤖 [NODE: Generator] -> Drafting the response...")
    question = state["question"]
    documents = state["documents"]
    
    # Increment the retry count inside the node (Edges cannot update state directly)
    current_retries = state.get("retry_count", 0) + 1
    
    # Format the surviving documents into a single string
    context = "\n\n".join(doc.page_content for doc in documents)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # If this is a retry, we can optionally make the prompt stricter
    prompt_instructions = "You are an AI assistant for research tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
    if current_retries > 1:
        prompt_instructions += " WARNING: Your previous attempt failed validation. You MUST answer ONLY using the exact facts in the context below. Do not infer."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_instructions),
        ("human", "Question: {question} \nContext: {context}")
    ])
    
    rag_chain = prompt | llm
    generation = rag_chain.invoke({"context": context, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation.content, "retry_count": current_retries}

# --- 7. THE CONDITIONAL EDGE: Hallucination Checker ---
def check_hallucinations(state: GraphState):
    """
    Determines whether the generation is grounded in the document.
    """
    print("\n🕵️ [EDGE: Manager] -> Checking for hallucinations...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retries = state.get("retry_count", 0)
    
    # --- THE KILL SWITCH ---
    if retries >= 3:
        print("   -> 🚨 MAX RETRIES REACHED. Forcing a stop to prevent infinite loop.")
        return "max_retries"
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    
    # Format docs for the grader
    context = "\n\n".join(doc.page_content for doc in documents)
    score = hallucination_grader.invoke({"documents": context, "generation": generation})
    
    if score.binary_score.lower() == "yes":
        print("   -> PASS: Answer is strictly grounded in the PDF. Sending to user.")
        return "useful"
    else:
        print(f"   -> FAIL: Answer contains hallucinations. Forcing rewrite. (Retry {retries}/3)")
        return "not supported"

# --- 8. BUILD THE GRAPH ---
print("\n⚙️ Compiling the LangGraph Agent...")
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Build the edges (The Logic)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")

# The Conditional Routing
workflow.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not supported": "generate", # Loop back and try again!
        "useful": END, # Finish the loop successfully
        "max_retries": END # Hard stop to prevent infinite loop
    }
)

# Compile it!
app = workflow.compile()
print("✅ Agent is online and ready for deployment.\n")


# --- 9. TEST THE AGENT ---
if __name__ == "__main__":
    # Feel free to change this question to test different scenarios
    test_question = "What are the primary risks to revenue growth?"
    
    print(f"USER QUERY: {test_question}")
    
    # Run the graph
    inputs = {"question": test_question}
    for output in app.stream(inputs):
        # Stream just lets us see what node is running in the terminal
        for key, value in output.items():
            pass 
            
    # Print the final validated answer
    print("\n==================================")
    print("FINAL VALIDATED ANSWER:")
    # Handle the case where the key might be missing if it hard-stopped early
    if "generation" in value:
        print(value["generation"])
    else:
        print("Agent stopped before generating a final answer.")
    print("==================================")