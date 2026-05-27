# 📈 Enterprise AI Financial Research Copilot

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local_Vector_Store-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-blueviolet)

An institutional-grade, multi-agent Retrieval-Augmented Generation (RAG) system designed to autonomously ingest, analyze, and extract quantitative and qualitative insights from SEC filings (10-K/10-Q). 

This project demonstrates a production-ready cognitive architecture featuring **deterministic routing, local vector storage, structured data extraction, and self-correcting hallucination loops.**

## 🧠 Cognitive Architecture

The system utilizes a **LangGraph Orchestrator** to route complex financial queries to specialized AI agents based on intent.

* **👔 Supervisor Agent (The Router):** Evaluates user intent and deterministically routes the query to the appropriate specialist agent. Includes built-in API rate-limit resilience and exponential backoff.
* **🧮 Quantitative Agent:** Bypasses LLM math hallucination by forcing the LLM to extract raw data into a strict `Pydantic` schema, which is then passed to a pure `Pandas` engine to calculate margins, YoY growth, and render dynamic UI charts.
* **🕵️‍♂️ Research Agent (Nested Graph):** A fully autonomous LangGraph sub-routine that executes a Retrieve-Grade-Generate loop. It features a strict hallucination "kill-switch" that forces the model to rewrite answers if they are not mathematically grounded in the retrieved SEC context.
* **🗣️ Sentiment Agent:** Analyzes management tone, forward-looking statements, and confidence levels regarding macroeconomic headwinds and strategic acquisitions.

## 🔒 Security & Data Governance

To align with strict financial data governance standards, this architecture utilizes a **Hybrid-Compute RAG Pipeline**:
* **Memory / Embeddings:** 100% Local and Offline. Utilizes HuggingFace `all-MiniLM-L6-v2` (384 dimensions) and ChromaDB to ensure proprietary financial documents never leave the local environment during the embedding phase.
* **Reasoning:** Google Gemini 2.5 Flash for high-speed, cost-effective structured reasoning and schema extraction.

## 📊 Telemetry & Observability

Full execution tracing is integrated via **LangSmith** to provide a complete audit trail of the AI's reasoning process.

> ![alt text](626shots_so.png)
*Caption: LangSmith trace demonstrating successful intent routing, token consumption tracking, and latency monitoring.*

## 🚀 Key Features

* **Dynamic UI Rendering:** The Streamlit frontend automatically adapts to the agent's output, rendering Markdown text, Sentiment Reports, or interactive Pandas DataFrames and Line Charts based on the routed intent.
* **Batch Document Grading:** Optimized document relevance grading to minimize API calls and prevent context bloat.
* **Graceful Degradation:** If the Quant Agent cannot find sufficient data to populate the Pandas matrix, it safely falls back to a natural language response rather than hallucinating numbers.

> ![alt text](925shots_so.png)
*Caption: The Streamlit UI rendering a dynamic Pandas dataframe and YoY growth chart from the Quant Agent.*

## 🛠️ Installation & Usage

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/padmabhushanpagare/ai_research_copilot.git
cd ai_research_copilot
pip install -r requirements.txt

2. Set up your environment variables:
Create a .env file in the root directory and add your API keys:

# AI Provider
GEMINI_API_KEY=your_google_api_key_here

# LangSmith Observability (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="quant_ai_research_copilot"

3. Run the application:

streamlit run app.py

(Note: Ensure your SEC filings / PDFs are placed in the designated data/ folder before initializing the vector store).

🗂️ Project Structure
app.py: Main Streamlit UI entry point.

graph_builder.py: LangGraph state management and node orchestration.

agents/: Contains the logic for the Supervisor, Quant, Research, and Sentiment agents.

tools/: Python/Pandas mathematical execution environment and ChromaDB retriever.

🔮 Future Roadmap
Hybrid Search Integration: Transitioning from pure semantic vector search to a Hybrid Search model (BM25 + Vector) to guarantee exact-match retrieval for highly specific accounting terminology.

Context Reranking: Implementing a Cross-Encoder (e.g., Cohere) to maximize context density within the LLM window while minimizing token costs.