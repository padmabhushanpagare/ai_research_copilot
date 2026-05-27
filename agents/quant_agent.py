import os
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Force load environment variables
load_dotenv(override=True)

# --- 1. THE DATA SCHEMAS (The Backbone of the Engine) ---
class IncomeStatement(BaseModel):
    """Structured extraction of core income statement metrics."""
    year: str = Field(description="The fiscal year of the data (e.g., '2025')")
    revenue: float = Field(description="Total revenue or sales in millions")
    operating_income: float = Field(description="Operating income or profit in millions")
    net_income: float = Field(description="Net income in millions")
    rd_expense: Optional[float] = Field(description="Research and Development expense in millions, if available", default=None)

class FinancialReport(BaseModel):
    """The master schema holding multiple years of data."""
    reports: list[IncomeStatement] = Field(description="Financial data across multiple years.")


# --- 2. THE ANALYTICS ENGINE (The Math) ---
def compute_kpis(financial_data: FinancialReport) -> pd.DataFrame:
    """Takes structured JSON from the LLM and computes financial ratios."""
    
    # 🚨 SAFETY VALVE: Check if the LLM found any data at all
    if not financial_data.reports:
        return pd.DataFrame() 
        
    # Convert the Pydantic objects to a list of dictionaries, then to a Pandas DataFrame
    data = [report.model_dump() for report in financial_data.reports]
    df = pd.DataFrame(data)
    
    # Sort by year so YoY calculations work accurately
    df = df.sort_values(by="year").reset_index(drop=True)
    
    # Compute Margins
    df['Net Margin (%)'] = (df['net_income'] / df['revenue']) * 100
    df['Operating Margin (%)'] = (df['operating_income'] / df['revenue']) * 100
    
    # Safely compute R&D Intensity (handling potential None values)
    if 'rd_expense' in df.columns:
        # Fill missing R&D data with 0 so the math doesn't crash
        df['rd_expense'] = pd.to_numeric(df['rd_expense']).fillna(0)
        df['R&D Intensity (%)'] = (df['rd_expense'] / df['revenue']) * 100
         
    # Compute Year-over-Year (YoY) Growth
    df['Revenue Growth YoY (%)'] = df['revenue'].pct_change() * 100
    
    # Round everything for clean UI presentation
    df = df.round(2)
    df = df.fillna("-") 
    
    return df


# --- 3. DATABASE CONNECTION (The Memory) ---
def get_retriever():
    """Connects to the local ChromaDB we built during the ingestion phase."""
    print("   -> Connecting to Local Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Using k=15 to ensure we capture both the numerical tables and the MD&A narrative
    return vectorstore.as_retriever(search_kwargs={"k": 15})


# --- 4. THE NODE (The LLM Extractor) ---
def quantitative_agent(state: dict):
    print("\n📊 [QUANT ENGINE] -> Booting up Institutional Analysis...")
    question = state.get("question", "")
    
    # Boot up the smartest available reasoning model for structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 🌟 Query Expansion (Translate casual English to Accounting Search Terms)
    print("   -> Optimizing search query...")
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. 
        Translate the user's casual query into a highly optimized search string to find financial tables in a 10-K report. 
        Include exact accounting terms like 'Consolidated Statement of Operations', 'Revenue', 'Operating Income', 'Net Income'. 
        Output ONLY the raw search string, nothing else."""),
        ("human", "{question}")
    ])
    search_query = (rewrite_prompt | llm).invoke({"question": question}).content
    print(f"   -> Expanded Query: {search_query}")
    
    # 1. Fetch Context from ChromaDB
    retriever = get_retriever()
    docs = retriever.invoke(search_query)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # 2. Extract Structured Data
    print("   -> Extracting raw line items via Pydantic...")
    structured_llm = llm.with_structured_output(FinancialReport)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Institutional Financial Analyst. 
        Extract the core Income Statement metrics from the provided text.
        Ensure all numbers are standardized to millions (e.g., if the text says 1.5 billion, output 1500).
        If a specific metric is not found for a year, do not guess. Leave it blank or omit it."""),
        ("human", "Extract the financial metrics from this text:\n\n{context}")
    ])
    
    extracted_data = (prompt | structured_llm).invoke({"context": context})
    
    # 3. Compute Ratios using pure Python
    print("   -> Computing margins and YoY growth...")
    kpi_dataframe = compute_kpis(extracted_data)
    
    # 🚨 GRACEFUL FALLBACK
    if kpi_dataframe.empty:
        print("   -> ⚠️ No structured data found. Falling back to text response.")
        return {
            "reply_type": "text", 
            "content": "I couldn't find enough structured financial data in the retrieved documents to compute those KPIs. Try asking about a specific year or metric.", 
            "sources": docs
        }
    
    print("   -> ✅ KPI Matrix successfully generated.")
    
    # Format for downstream UI/Graph
    return {
        "reply_type": "kpi_table", 
        "content": kpi_dataframe, 
        "sources": docs
    }

# --- 5. STANDALONE TEST EXECUTION ---
if __name__ == "__main__":
    # Simulate a LangGraph state input
    test_state = {"question": "How did Tesla's revenue and operating income perform over the last 3 years?"}
    
    result = quantitative_agent(test_state)
    
    print("\n\n" + "="*50)
    print("FINAL OUTPUT")
    print("="*50)
    if result["reply_type"] == "kpi_table":
        print(result["content"].to_markdown(index=False))
    else:
        print(result["content"])