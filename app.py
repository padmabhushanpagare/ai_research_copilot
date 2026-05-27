import streamlit as st
import os
import json
import re
import time
import pandas as pd
from typing import Literal, TypedDict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# IMPORT YOUR SELF-CORRECTING RAG AGENT!
from agent import app as research_agent_app
from agents.quant_agent import quantitative_agent

# 1. Setup the Page
st.set_page_config(page_title="AI research copilot", page_icon="📈", layout="wide")
st.title("📈 Enterprise AI Financial research copilot")
st.markdown("Ask anything. The Supervisor Agent will route your query to the correct AI specialist.")

load_dotenv()

# --- CACHE THE RETRIEVER ---
@st.cache_resource
def load_retriever():
    print("🧠 Connecting to Local Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Bumped 'k' to 15 to ensure the Sentiment agent gets enough narrative context!
    return vectordb.as_retriever(search_kwargs={"k": 15})

retriever = load_retriever()

# ==========================================
# 🧠 THE SWARM COGNITIVE ARCHITECTURE
# ==========================================

class SwarmState(TypedDict):
    question: str
    route: str
    reply_type: str # "text", "chart", "sentiment", or "kpi_table"
    content: Any
    sources: Any

class RouteDecision(BaseModel):
    next_agent: Literal["Quantitative_Agent", "Sentiment_Agent", "Research_Agent"] = Field(
        description="The specialized agent to handle the query."
    )


def supervisor_node(state: dict):
    print("👔 [SUPERVISOR] -> Evaluating Intent...")
    
    # 1. Define the LLM, Prompt, and Pipeline specifically for the Supervisor
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Supervisor Agent of a Financial AI Swarm. 
        Route the user's query to the most appropriate agent:
        - 'Quantitative_Agent': For extracting numerical data, financial metrics, tables, calculating ratios, or YoY growth.
        - 'Sentiment_Agent': For evaluating the tone, confidence, or qualitative drivers of management.
        - 'Research_Agent': For general factual questions, summarization, or deep-dives into the text."""),
        ("human", "{question}")
    ])
    
    # We name it 'supervisor_chain' to avoid Python thinking it's the 'itertools.chain' module!
    supervisor_chain = prompt | structured_llm
    
    # 2. 🛡️ THE RESILIENCE WRAPPER
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Try to route the query
            decision = supervisor_chain.invoke({"question": state["question"]})
            print(f"   -> Routing to: {decision.next_agent}")
            
            # Return the updated state so the graph knows where to go
            return {"route": decision.next_agent}
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"   ⚠️ Supervisor Rate Limit Hit! Sleeping for 20 seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(20)
                if attempt == max_retries - 1:
                    raise Exception("API is completely maxed out. Please try again in a few minutes.")
            else:
                raise e # If it's a different kind of error, crash normally


def sentiment_agent(state: SwarmState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the tone of the provided context. Output a structured report with Overall Tone, Key Drivers, and Management Confidence."),
        ("human", "Question: {question}\nContext: {context}")
    ])
    
    result = (prompt | llm).invoke({"question": state["question"], "context": context}).content
    return {"reply_type": "sentiment", "content": result, "sources": docs}


def research_agent(state: SwarmState):
    # TRIGGER THE EXTERNAL GRAPH FROM agent.py!
    final_state = research_agent_app.invoke({"question": state["question"]})
    
    if "generation" in final_state:
        return {"reply_type": "text", "content": final_state["generation"], "sources": final_state.get("documents", [])}
    return {"reply_type": "text", "content": "The agent stopped early to prevent a hallucination loop.", "sources": []}


# --- COMPILE THE SWARM ---
@st.cache_resource
def build_swarm():
    workflow = StateGraph(SwarmState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Quantitative_Agent", quantitative_agent)
    workflow.add_node("Sentiment_Agent", sentiment_agent)
    workflow.add_node("Research_Agent", research_agent)
    
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", lambda state: state["route"], {
        "Quantitative_Agent": "Quantitative_Agent",
        "Sentiment_Agent": "Sentiment_Agent",
        "Research_Agent": "Research_Agent"
    })
    
    workflow.add_edge("Quantitative_Agent", END)
    workflow.add_edge("Sentiment_Agent", END)
    workflow.add_edge("Research_Agent", END)
    
    return workflow.compile()

swarm_app = build_swarm()

# ==========================================
# 🖥️ THE STREAMLIT UI
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages dynamically based on type
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "sentiment":
            st.info("🧠 **Sentiment Analysis Report**")
            st.markdown(msg["content"])
        elif msg["type"] == "chart":
            st.success("📊 **Data Extracted Successfully**")
            st.bar_chart(msg["content"])
        elif msg["type"] == "kpi_table": 
            st.success("🧮 **Financial Ratios Computed**")
            st.dataframe(msg["content"], use_container_width=True)
            # Safely render the line chart if data exists
            if isinstance(msg["content"], pd.DataFrame) and 'Revenue Growth YoY (%)' in msg["content"].columns:
                try:
                    plot_df = msg["content"].copy()
                    plot_df['Revenue Growth YoY (%)'] = pd.to_numeric(plot_df['Revenue Growth YoY (%)'], errors='coerce')
                    st.line_chart(plot_df.set_index('year')['Revenue Growth YoY (%)'])
                except:
                    pass
            
        if msg.get("sources"):
            with st.expander("📚 View Source Citations"):
                for i, source in enumerate(msg["sources"]):
                    st.caption(f"**Source {i+1}:** {source}...")

# Chat Input
if user_query := st.chat_input("e.g., Extract revenue trend, or What is the management tone?"):
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("🤖 Swarm is processing...", expanded=True) as status:
            st.write("👔 Supervisor evaluating intent...")
            
            # RUN THE SWARM!
            try:
                result = swarm_app.invoke({"question": user_query})
                
                # Update status based on route
                route_taken = result.get('route', 'Unknown')
                st.write(f"🔄 Routed to: **{route_taken.replace('_', ' ')}**")
                status.update(label="Analysis Complete", state="complete", expanded=False)

                reply_type = result.get("reply_type", "text")
                content = result.get("content", "Error processing request.")
                sources = result.get("sources", [])
                
                # Ensure sources are strings for rendering safely
                saved_sources = []
                for doc in sources:
                    if hasattr(doc, 'page_content'):
                        saved_sources.append(doc.page_content[:200])
                    elif isinstance(doc, str):
                        saved_sources.append(doc[:200])
                    else:
                        saved_sources.append(str(doc)[:200])

                # Render output dynamically
                if reply_type == "text":
                    st.markdown(content)
                elif reply_type == "sentiment":
                    st.info("🧠 **Sentiment Analysis Report**")
                    st.markdown(content)
                elif reply_type == "chart":
                    st.success("📊 **Data Extracted Successfully**")
                    st.bar_chart(content)
                elif reply_type == "kpi_table": 
                    st.success("🧮 **Financial Ratios Computed**")
                    st.dataframe(content, use_container_width=True)
                    if isinstance(content, pd.DataFrame) and 'Revenue Growth YoY (%)' in content.columns:
                        try:
                            plot_df = content.copy()
                            plot_df['Revenue Growth YoY (%)'] = pd.to_numeric(plot_df['Revenue Growth YoY (%)'], errors='coerce')
                            st.line_chart(plot_df.set_index('year')['Revenue Growth YoY (%)'])
                        except:
                            pass

                if saved_sources:
                    with st.expander("📚 View Source Citations"):
                        for i, doc in enumerate(saved_sources):
                            st.caption(f"**Source {i+1}:** {doc}...")

                # Save to memory so it survives page reloads
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": reply_type, 
                    "content": content,
                    "sources": saved_sources
                })
            except Exception as e:
                status.update(label="Error Occurred", state="error", expanded=True)
                st.error(f"An error occurred: {e}")