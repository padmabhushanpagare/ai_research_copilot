import os
import json
import time
import base64
import fitz  # PyMuPDF
import io
from PIL import Image
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (Looking for GOOGLE_API_KEY)
load_dotenv(override=True)

# --- CONFIGURATION ---
PDF_FILES = [
    r"D:\ai_research_copilot\0001628280-26-003952_tsla-20251231.pdf", 
    r"D:\workflow_automation_system\archive\tsla-20260331-gen.pdf" 
]

DB_DIR = "./chroma_db"
MD_BACKUP = "extracted_backup.md"
STATE_FILE = "ingestion_state.json"

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            if "last_page" in data and not any(isinstance(v, dict) for v in data.values()):
                return {"0001628280-26-003952_tsla-20251231.pdf": {"last_page": data["last_page"], "is_complete": False}}
            return data
    return {}

def get_file_progress(file_path: str) -> int:
    state = load_state()
    filename = os.path.basename(file_path)
    return state.get(filename, {}).get("last_page", 0)

def save_file_progress(file_path: str, page_num: int, is_complete: bool = False):
    state = load_state()
    filename = os.path.basename(file_path)
    
    if filename not in state:
        state[filename] = {}
        
    state[filename]["last_page"] = page_num
    state[filename]["is_complete"] = is_complete
    
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def extract_markdown_with_graceful_exit(pdf_path: str):
    filename = os.path.basename(pdf_path)
    print(f"\n📄 Opening {filename} for Vision OCR with Google Gemini...")
    
    # 🌟 RESTORED GOOGLE GEMINI LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_page = get_file_progress(pdf_path)
    
    if start_page >= total_pages:
        print(f"✅ {filename} is already 100% processed! Skipping.")
        save_file_progress(pdf_path, total_pages, is_complete=True)
        return True 
        
    print(f"🔄 Resuming {filename} from Page {start_page + 1}...")
    
    with open(MD_BACKUP, "a", encoding="utf-8") as f:
        if start_page == 0:
            f.write(f"\n\n# --- SOURCE DOCUMENT: {filename} ---\n\n")
            f.flush()
            
        for page_num in range(start_page, total_pages):
            print(f"   -> Processing Page {page_num + 1}/{total_pages}...")
            
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            base64_image = encode_image_to_base64(img)
            
            prompt = """
            You are a highly precise document extraction AI. 
            Extract all text from this image exactly as it appears. 
            CRITICAL: If you see a table or financial statement, format it as a perfect Markdown grid. 
            Do not add any conversational text or descriptions. Only output the extracted Markdown.
            """
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            
            success = False
            retries = 0
            
            while not success and retries < 3:
                try:
                    response = llm.invoke([message])
                    f.write(f"\n\n## Page {page_num + 1}\n\n" + response.content)
                    f.flush() 
                    save_file_progress(pdf_path, page_num + 1)
                    success = True
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print(f"\n🛑 API Quota Reached on {filename} Page {page_num + 1}! Saving progress and shutting down.")
                        return False 
                    
                    retries += 1
                    print(f"      ⚠️ API Glitch: {error_msg}. Retrying ({retries}/3). Sleeping 10s...")
                    time.sleep(10)
            
            if not success:
                print(f"\n❌ Unresolvable error on {filename} Page {page_num + 1}. Halting to save progress.")
                return False 
                
            # 🛑 CRITICAL FOR GOOGLE FREE TIER: 15 Requests per minute = 4 seconds per request
            time.sleep(4.5)
                
    print(f"\n🎉 Extraction 100% Complete for {filename}!")
    save_file_progress(pdf_path, total_pages, is_complete=True)
    return True

def build_vector_database():
    if not os.path.exists(MD_BACKUP):
        print("❌ No markdown file found.")
        return
        
    print("\n🪓 Compiling Vector Database from saved Markdown...")
    with open(MD_BACKUP, "r", encoding="utf-8") as f:
        markdown_text = f.read()
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n## ", "\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.create_documents([markdown_text])
    print(f"   -> Created {len(chunks)} chunks. Saving to ChromaDB in batches...")
    
    # 🌟 RESTORED GOOGLE EMBEDDINGS
    print("   -> Initializing Local HuggingFace Embeddings (Unlimited!)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    batch_size = 15 # Smaller batch size for Google limits
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) // batch_size) + 1
        
        print(f"      -> Embedding batch {current_batch_num}/{total_batches}...")
        
        success = False
        while not success:
            try:
                vectorstore.add_documents(batch)
                success = True
            except Exception as e:
                print(f"         ⚠️ Embedding Error: {e}. Sleeping 30s before retrying...")
                time.sleep(30)
        
        if current_batch_num < total_batches:
            time.sleep(10) # Respecting Google's embedding rate limit
            
    print("✅ Vector Database successfully built/updated!")

if __name__ == "__main__":
    api_blocked = False
    
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            print(f"❌ Error: Could not find '{pdf}'. Make sure the path is correct.")
            continue
            
        success = extract_markdown_with_graceful_exit(pdf)
        
        if not success:
            api_blocked = True
            break 
            
    build_vector_database()
    
    if api_blocked:
        print("\n⚠️ NOTE: The script was halted due to API Quota. Run again tomorrow to resume.")