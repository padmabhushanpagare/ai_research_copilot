import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from agents.quant_agent import quantitative_agent

load_dotenv()

# --- 1. THE SUPERVISOR STATE ---
class SupervisorState(TypedDict):
    question: str
    route: str
    final_answer: str

# --- 2. THE ROUTING SCHEMA ---
class RouteDecision(BaseModel):
    """Strict schema forcing the LLM to pick one of our specific agents."""
    next_agent: Literal["Quantitative_Agent", "Sentiment_Agent", "Research_Agent"] = Field(
        description="The specialized agent that should handle the user's query."
    )

# --- 3. THE SUPERVISOR NODE ---
def supervisor_node(state: SupervisorState):
    print("\n👔 [SUPERVISOR] -> Analyzing user intent...")
    question = state["question"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # We force the LLM to respond ONLY with our RouteDecision schema
    router_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Chief Financial AI Supervisor. Direct the user's query to the correct specialized agent.
        - Use 'Quantitative_Agent' for questions asking for specific numbers, historical data, or financial trends.
        - Use 'Sentiment_Agent' for questions about management tone, forward-looking outlook, or qualitative risks.
        - Use 'Research_Agent' for general factual questions about the company's operations or products."""),
        ("human", "{question}")
    ])
    
    chain = prompt | router_llm
    decision = chain.invoke({"question": question})
    
    print(f"   -> Routing to: {decision.next_agent}")
    return {"route": decision.next_agent, "question": question}

# --- 4. THE WORKER NODES (Sub-Agents) ---
# In a full app, these functions would trigger your other scripts!

def quantitative_agent(state: SupervisorState):
    print("📊 [QUANT AGENT] -> Booting up math and table extraction logic...")
    # NOTE: You would paste your JSON trend extraction logic here!
    return {"final_answer": "Here is the extracted numerical trend data."}

def sentiment_agent(state: SupervisorState):
    print("🧠 [SENTIMENT AGENT] -> Booting up tone analysis logic...")
    # NOTE: You would paste your Sentiment analysis logic here!
    return {"final_answer": "Management tone is highly cautious regarding inflation."}

def research_agent(state: SupervisorState):
    print("📚 [RESEARCH AGENT] -> Triggering the Hallucination-Free RAG Graph...")
    # NOTE: You would trigger your self-correcting agent.py graph here!
    return {"final_answer": "Here is the verified, grounded response."}

# --- 5. COMPILE THE SWARM GRAPH ---
print("\n⚙️ Compiling Supervisor Swarm...")
workflow = StateGraph(SupervisorState)

# Add all the agents
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Quantitative_Agent", quantitative_agent)
workflow.add_node("Sentiment_Agent", sentiment_agent)
workflow.add_node("Research_Agent", research_agent)

# The entry point is ALWAYS the supervisor
workflow.set_entry_point("supervisor")

# The dynamic routing logic!
def route_to_worker(state: SupervisorState):
    return state["route"] # Returns the string the LLM chose

workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "Quantitative_Agent": "Quantitative_Agent",
        "Sentiment_Agent": "Sentiment_Agent",
        "Research_Agent": "Research_Agent"
    }
)

# All workers just end the graph when they finish
workflow.add_edge("Quantitative_Agent", END)
workflow.add_edge("Sentiment_Agent", END)
workflow.add_edge("Research_Agent", END)

swarm_app = workflow.compile()
print("✅ Swarm is online.\n")

# --- 6. TEST THE SUPERVISOR ---
if __name__ == "__main__":
    
    # Test 1: A general question
    print("--- TEST 1 ---")
    swarm_app.invoke({"question": "What are the company's main products?"})
    
    # Test 2: A tone question
    print("\n--- TEST 2 ---")
    swarm_app.invoke({"question": "Is management worried about the supply chain?"})
    
    # Test 3: A math question
    print("\n--- TEST 3 ---")
    swarm_app.invoke({"question": "Extract the historical revenue trend."})