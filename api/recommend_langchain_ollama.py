import os
from dotenv import load_dotenv 

# --- CRITICAL FIX: Environment Loading MUST happen first ---
# Load .env variables immediately after os is available
load_dotenv()

# Set HuggingFace Cache variables to ensure paths are stable before libraries load
hf_cache_path = os.path.join(os.path.dirname(__file__), "../.hf_cache")
os.makedirs(hf_cache_path, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_path
os.environ["TRANSFORMERS_CACHE"] = hf_cache_path
os.environ["HF_DATASETS_CACHE"] = hf_cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_path
# --- END CRITICAL SETUP ---

from fastapi import FastAPI, APIRouter, HTTPException, status
from pydantic import BaseModel
import joblib, logging, uuid, time, json
from typing import List, Dict, Any
import numpy as np

# RAG/ML Libraries
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from ollama import chat 

# ------------------ PATHS & CACHE SETUP ------------------
BASE_DIR = os.path.dirname(__file__)
EMB_DIR = os.path.join(BASE_DIR, "../embeddings")
BM25_INDEX_PATH = os.path.join(EMB_DIR, "bm25_index.pkl")
PERSONA_META = os.path.join(EMB_DIR, "persona_docs.pkl")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../demo")
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------ DEDICATED LOGGER SETUP ------------------
rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rag_logger.addHandler(file_handler)

# ------------------ FASTAPI INITIALIZATION ------------------
app = FastAPI(title="Lead HeatScore API",
              description="Classify leads and generate first personalized outreach message",
              version="1.0")
router = APIRouter()

# ------------------ MODEL & DATA LOADING ------------------
# Pinecone Setup (Reads from .env)
API_KEY = os.getenv("PINECONE_API_KEY")
ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "lead-heatscore-qeagle"
pc = Pinecone(api_key=API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# Load local models and data
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, cache_folder=hf_cache_path)
bm25_index = joblib.load(BM25_INDEX_PATH)
persona_df = joblib.load(PERSONA_META)

# Cross-Encoder Setup
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    have_reranker = True
except Exception as e:
    rag_logger.warning("CrossEncoder not available: %s", e)
    reranker = None
    have_reranker = False

# Ollama & Policy Config
ollama_model_name = "gemma:2b"
POLICY = {
    "tone": "friendly",
    "CTA": "click link to explore course",
    "channel_priority": ["WhatsApp", "Email"]
}

# ------------------ PYDANTIC DATA MODEL ------------------
class Lead(BaseModel):
    lead_id: int
    source: str
    recency_days: int
    region: str
    role: str
    campaign: str
    page_views: int
    last_touch: str
    prior_course_interest: int

# ------------------ HELPER FUNCTIONS ------------------
# FIX: Renamed top_n to top_k to align with API parameters
def rerank_snippets(query: str, docs: List[Dict], top_k: int = 1):
    if not have_reranker:
        # Returns docs and a list of None for scores
        return docs[:top_k], [None]*min(top_k, len(docs))
    pairs = [(query, d['text']) for d in docs]
    scores = reranker.predict(pairs)
    idx = np.argsort(scores)[::-1]
    top_idx = idx[:top_k]
    # Ensure this function always returns two values (docs and scores)
    return [docs[i] for i in top_idx], [float(scores[i]) for i in top_idx]

def generate_message_with_ollama(lead, top_snippets: List[str]):
    prompt = (
        f"You are a friendly sales agent. Your task is to craft a single, professional outreach message "
        f"for a {lead.role} that references their interest in the {lead.campaign} campaign. "
        f"Do not include any placeholders or text formatting like bolding. "
        f"Synthesize the key points from the provided persona snippets to craft the message. "
        f"Policy: tone={POLICY['tone']}, CTA={POLICY['CTA']}. Keep the message to a maximum of 40 words.\n"
        f"Persona snippets:\n" + "\n".join(f"- {t}" for t in top_snippets)
    )
    
    ollama_start_time = time.perf_counter()
    response = chat(
        model=ollama_model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    ollama_end_time = time.perf_counter()
    ollama_latency_ms = (ollama_end_time - ollama_start_time) * 1000
    rag_logger.info(f"Ollama inference latency: {ollama_latency_ms:.2f} ms")
    
    return response['message']['content']

# ------------------ CORE RAG LOGIC (SHARED) ------------------
def execute_rag_pipeline(request_id: str, lead: Lead, search_type: str, top_k: int = 1):
    start_time = time.perf_counter()
    lead_dict = lead.dict()
    
    query_text = f"{lead.role} {lead.campaign} interest:{lead.prior_course_interest} pages:{lead.page_views}"
    query_emb = embeddings.embed_query(query_text)

    # Hybrid Search (Combined base retrieval logic)
    vector_search_results = pinecone_index.query(vector=query_emb, top_k=5, include_metadata=True)
    vector_docs = [match['metadata'] for match in vector_search_results['matches']]
    
    tokenized_query = query_text.split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1]
    bm25_docs = [persona_df.iloc[idx].to_dict() for idx in bm25_top_indices if bm25_scores[idx] > 0]
    
    combined_docs = {doc['pid']: doc for doc in vector_docs}
    for doc in bm25_docs:
        if 'pid' in doc and doc['pid'] not in combined_docs:
            combined_docs[doc['pid']] = doc
    
    combined_docs_list = list(combined_docs.values())
    
    # Ablation/Final Retrieval Logic based on search_type
    retrieved_docs = []
    rerank_scores = None # Initialize to None

    if search_type == "vector-only":
        retrieved_docs = vector_docs[:top_k]
    elif search_type == "hybrid":
        retrieved_docs = combined_docs_list[:top_k]
    elif search_type == "hybrid-rerank":
        reranked_docs, rerank_scores = rerank_snippets(query_text, combined_docs_list, top_k=top_k)
        retrieved_docs = reranked_docs
        rag_logger.info(f"[{request_id}] Reranker selected {len(retrieved_docs)} documents.")
    
    # Post-Retrieval Filtering
    filtered_docs = [doc for doc in retrieved_docs if lead.role.lower() in doc.get('text', '').lower()]
    if not filtered_docs:
        filtered_docs = retrieved_docs[:1]

    top_texts = [d['text'] for d in filtered_docs]
    references = [d['pid'] for d in filtered_docs]

    message = generate_message_with_ollama(lead, top_texts)
    
    # Final Output Construction
    channel = POLICY['channel_priority'][0]
    rationale = f"Ablation Test: {search_type}. Lead role: {lead.role}" if "ablation" in search_type else f"Lead role: {lead.role}, page_views: {lead.page_views}, prior_course_interest: {lead.prior_course_interest}"
    
    # Only add rerank scores to the rationale if reranking was actually performed
    if rerank_scores is not None and search_type == "hybrid-rerank": 
        # Map back to the specific filtered docs used
        final_scores = [score for doc, score in zip(retrieved_docs, rerank_scores) if doc['pid'] in references]
        if final_scores:
            rationale += f", rerank_scores: {[round(s,3) for s in final_scores]}"

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    rag_logger.info(f"[{request_id}] RAG pipeline completed. Latency: {latency_ms:.2f} ms")

    return {
        "channel": channel,
        "message": message,
        "rationale": rationale,
        "references": references,
        "search_type": search_type if "ablation" in search_type else "hybrid-rerank"
    }

# ------------------ API ROUTE: ABLATION TEST ------------------
@router.post("/ablation-test/{search_type}")
def ablation_test(search_type: str, lead: Lead, top_k: int = 1):
    rag_logger.info("\n" + "="*50)
    request_id = str(uuid.uuid4())
    
    if search_type not in ["vector-only", "hybrid", "hybrid-rerank"]:
         raise HTTPException(status_code=400, detail="Invalid search_type. Must be 'vector-only', 'hybrid', or 'hybrid-rerank'.")

    try:
        response = execute_rag_pipeline(request_id, lead, search_type, top_k)
        return response
    except Exception as e:
        error_message = f"An unexpected error occurred during ablation test: {e}"
        rag_logger.error(f"[{request_id}] Ablation test failed with error: {error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

# ------------------ API ROUTE: MAIN RECOMMENDATION ------------------
@router.post("/")
def recommend_action(lead: Lead, top_k: int = 1):
    rag_logger.info("\n" + "="*50)
    request_id = str(uuid.uuid4())
    
    try:
        # Calls the shared pipeline function with the final, production search type
        response = execute_rag_pipeline(request_id, lead, "hybrid-rerank", top_k)
        # Remove the 'search_type' key from the final production response
        del response['search_type']
        return response
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        rag_logger.error(f"[{request_id}] RAG pipeline failed with error: {error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )


# ------------------ Register Router ------------------
app.include_router(router, prefix="/recommend")

# ------------------ Root Endpoint ------------------
@app.get("/")
def root():
    return {"message": "Lead HeatScore API with Ollama Recommender is running!"}
