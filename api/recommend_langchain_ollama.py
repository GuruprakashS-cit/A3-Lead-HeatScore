import os
# Move all environment variable setups to the very top
hf_cache_path = os.path.join(os.path.dirname(__file__), "../.hf_cache")
os.makedirs(hf_cache_path, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_path
os.environ["TRANSFORMERS_CACHE"] = hf_cache_path
os.environ["HF_DATASETS_CACHE"] = hf_cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_path

from fastapi import FastAPI, APIRouter, HTTPException, status
from pydantic import BaseModel
import joblib, logging, uuid, time, json
from typing import List, Dict, Any
import numpy as np

# Updated imports for Pinecone and BM25
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from ollama import chat 

# ------------------ Setup ------------------
# Define the path for the new log directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "../demo")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a dedicated logger for the RAG pipeline
rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)
# Create a file handler that writes to the new log file
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
file_handler.setLevel(logging.INFO)
# Create a formatter and set it for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# Add the file handler to the logger
rag_logger.addHandler(file_handler)

app = FastAPI(title="Lead HeatScore API",
              description="Classify leads and generate first personalized outreach message",
              version="1.0")
router = APIRouter()

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(__file__)
EMB_DIR = os.path.join(BASE_DIR, "../embeddings")
# Updated paths for the new indexes
BM25_INDEX_PATH = os.path.join(EMB_DIR, "bm25_index.pkl")
PERSONA_META = os.path.join(EMB_DIR, "persona_docs.pkl")

# ------------------ Pinecone Setup ------------------
# Replace with your own API key and environment
API_KEY = "pcsk_5VSpwm_16JSye9ViLTq6D4KQL1mmAdf9An5SEsJPJCUoyP6tYrVCi8hSVg3nKJ6ZzsMact"
ENVIRONMENT = "us-east-1"
INDEX_NAME = "lead-heatscore-qeagle"

# Connect to Pinecone and the specific index
pc = Pinecone(api_key=API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# ------------------ Embeddings & Models ------------------
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, cache_folder=hf_cache_path)

# Load BM25 index and persona metadata
bm25_index = joblib.load(BM25_INDEX_PATH)
persona_df = joblib.load(PERSONA_META)

# ------------------ Cross-Encoder ------------------
try:
    # This will now download the model to the new cache path
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    have_reranker = True
except Exception as e:
    rag_logger.warning("CrossEncoder not available: %s", e)
    reranker = None
    have_reranker = False

# ------------------ Ollama ------------------
ollama_model_name = "gemma:2b"

POLICY = {
    "tone": "friendly",
    "CTA": "click link to explore course",
    "channel_priority": ["WhatsApp", "Email"]
}

# ------------------ Pydantic Models ------------------
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

# ------------------ Helper Functions ------------------
def rerank_snippets(query: str, docs: List[Dict], top_n: int = 1):
    if not have_reranker:
        return docs[:top_n], [None]*min(top_n, len(docs))
    pairs = [(query, d['text']) for d in docs]
    scores = reranker.predict(pairs)
    idx = np.argsort(scores)[::-1]
    top_idx = idx[:top_n]
    return [docs[i] for i in top_idx], [float(scores[i]) for i in top_idx]

def generate_message_with_ollama(lead, top_snippets: List[str]):
    # Updated prompt with a persona and stronger instructions
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

# ------------------ API Route for Ablation Test ------------------
@router.post("/ablation-test/{search_type}")
def ablation_test(search_type: str, lead: Lead, top_k: int = 1):
    rag_logger.info("\n" + "="*50)
    
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    lead_dict = lead.dict()

    rag_logger.info(f"[{request_id}] Starting Ablation Test: {search_type} for lead: {lead_dict['lead_id']}")

    try:
        query_text = f"{lead.role} {lead.campaign} interest:{lead.prior_course_interest} pages:{lead.page_views}"
        query_emb = embeddings.embed_query(query_text)

        # Step 1: Perform the search based on search_type
        retrieved_docs = []
        if search_type == "vector-only":
            vector_search_results = pinecone_index.query(
                vector=query_emb,
                top_k=5,
                include_metadata=True
            )
            retrieved_docs = [match['metadata'] for match in vector_search_results['matches']]
            rag_logger.info(f"[{request_id}] Vector-only search retrieved {len(retrieved_docs)} documents.")

        elif search_type == "hybrid":
            vector_search_results = pinecone_index.query(
                vector=query_emb,
                top_k=5,
                include_metadata=True
            )
            vector_docs = [match['metadata'] for match in vector_search_results['matches']]
            
            tokenized_query = query_text.split(" ")
            bm25_scores = bm25_index.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1]
            bm25_docs = [persona_df.iloc[idx].to_dict() for idx in bm25_top_indices if bm25_scores[idx] > 0]
            
            combined_docs = {doc['pid']: doc for doc in vector_docs}
            for doc in bm25_docs:
                if 'pid' in doc and doc['pid'] not in combined_docs:
                    combined_docs[doc['pid']] = doc
            retrieved_docs = list(combined_docs.values())
            rag_logger.info(f"[{request_id}] Hybrid search retrieved {len(retrieved_docs)} documents.")

        elif search_type == "hybrid-rerank":
            # This is your existing logic, which we'll reuse
            vector_search_results = pinecone_index.query(
                vector=query_emb,
                top_k=5,
                include_metadata=True
            )
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
            
            reranked_docs, rerank_scores = rerank_snippets(query_text, combined_docs_list, top_n=top_k)
            retrieved_docs = reranked_docs
            rag_logger.info(f"[{request_id}] Hybrid + Rerank selected {len(retrieved_docs)} documents.")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid search_type. Must be 'vector-only', 'hybrid', or 'hybrid-rerank'.")

        # Use the documents to generate the message
        filtered_docs = [doc for doc in retrieved_docs if lead.role.lower() in doc.get('text', '').lower()]
        
        if not filtered_docs:
            filtered_docs = retrieved_docs[:1]

        top_texts = [d['text'] for d in filtered_docs]
        references = [d['pid'] for d in filtered_docs]

        message = generate_message_with_ollama(lead, top_texts)
        rag_logger.info(f"[{request_id}] Message generated successfully.")

        channel = POLICY["channel_priority"][0]
        rationale = f"Ablation Test: {search_type}. Lead role: {lead.role}"

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        rag_logger.info(f"[{request_id}] Ablation Test completed. Latency: {latency_ms:.2f} ms")

        return {
            "channel": channel,
            "message": message,
            "rationale": rationale,
            "references": references,
            "search_type": search_type
        }
    
    except Exception as e:
        error_message = f"An unexpected error occurred during ablation test: {e}"
        rag_logger.error(f"[{request_id}] Ablation test failed with error: {error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

# ------------------ API Route for Main Recommendation ------------------
@router.post("/")
def recommend_action(lead: Lead, top_k: int = 1):
    rag_logger.info("\n" + "="*50)
    
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    lead_dict = lead.dict()

    rag_logger.info(f"[{request_id}] Starting RAG for lead: {lead_dict['lead_id']}")

    try:
        query_text = f"{lead.role} {lead.campaign} interest:{lead.prior_course_interest} pages:{lead.page_views}"
        query_emb = embeddings.embed_query(query_text)

        # Hybrid Search: Vector + Keyword
        vector_search_results = pinecone_index.query(
            vector=query_emb,
            top_k=5,
            include_metadata=True
        )
        vector_docs = [match['metadata'] for match in vector_search_results['matches']]
        rag_logger.info(f"[{request_id}] Vector search retrieved {len(vector_docs)} documents.")

        tokenized_query = query_text.split(" ")
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1]
        bm25_docs = [persona_df.iloc[idx].to_dict() for idx in bm25_top_indices if bm25_scores[idx] > 0]
        rag_logger.info(f"[{request_id}] Keyword search retrieved {len(bm25_docs)} documents.")
        
        combined_docs = {doc['pid']: doc for doc in vector_docs}
        for doc in bm25_docs:
            if 'pid' in doc and doc['pid'] not in combined_docs:
                combined_docs[doc['pid']] = doc
        
        combined_docs_list = list(combined_docs.values())
        
        reranked_docs, rerank_scores = rerank_snippets(query_text, combined_docs_list, top_n=top_k)
        rag_logger.info(f"[{request_id}] Reranker selected {len(reranked_docs)} top documents with scores: {rerank_scores}")

        filtered_docs = [doc for doc in reranked_docs if lead.role.lower() in doc.get('text', '').lower()]
        
        if not filtered_docs:
            filtered_docs = reranked_docs[:1]

        top_texts = [d['text'] for d in filtered_docs]
        references = [d['pid'] for d in filtered_docs]

        message = generate_message_with_ollama(lead, top_texts)
        rag_logger.info(f"[{request_id}] Message generated successfully.")

        channel = POLICY["channel_priority"][0]
        rationale = f"Lead role: {lead.role}, page_views: {lead.page_views}, prior_course_interest: {lead.prior_course_interest}"
        if have_reranker and rerank_scores:
            rationale += f", rerank_scores: {[round(s,3) for s in rerank_scores]}"

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        rag_logger.info(f"[{request_id}] RAG pipeline completed. Latency: {latency_ms:.2f} ms")

        return {
            "channel": channel,
            "message": message,
            "rationale": rationale,
            "references": references
        }
    
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
