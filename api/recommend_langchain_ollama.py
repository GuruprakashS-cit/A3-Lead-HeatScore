import os
# Move all environment variable setups to the very top
hf_cache_path = os.path.join(os.path.dirname(__file__), "../.hf_cache")
os.makedirs(hf_cache_path, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_path
os.environ["TRANSFORMERS_CACHE"] = hf_cache_path
os.environ["HF_DATASETS_CACHE"] = hf_cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_path

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import joblib, logging, uuid
from typing import List, Dict
import numpy as np

# Updated imports for Pinecone and BM25
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from ollama import chat 

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO)
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
    logging.warning("CrossEncoder not available: %s", e)
    reranker = None
    have_reranker = False

# ------------------ Ollama ------------------
ollama_model_name = "llama3.2:3b"

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
def rerank_snippets(query: str, docs: List[Dict], top_n: int = 3):
    if not have_reranker:
        return docs[:top_n], [None]*min(top_n, len(docs))
    pairs = [(query, d['text']) for d in docs]
    scores = reranker.predict(pairs)
    idx = np.argsort(scores)[::-1]
    top_idx = idx[:top_n]
    return [docs[i] for i in top_idx], [float(scores[i]) for i in top_idx]

def generate_message_with_ollama(lead, top_snippets: List[str]):
    prompt = (
        f"You are an assistant that crafts short friendly outreach messages. "
        f"Policy: tone={POLICY['tone']}, CTA={POLICY['CTA']}. Keep <= 40 words.\n"
        f"Lead info: {lead.dict()}\n"
        f"Persona snippets:\n" + "\n".join(f"- {t}" for t in top_snippets)
    )
    response = chat(
        model=ollama_model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    # Correct way to get text from the latest ollama chat client
    return response['message']['content']

# ------------------ API Route ------------------
@router.post("/")
def recommend_action(lead: Lead, top_k: int = 3):
    request_id = str(uuid.uuid4())
    lead_dict = lead.dict()

    # Create query
    query_text = f"{lead.role} {lead.campaign} interest:{lead.prior_course_interest} pages:{lead.page_views}"
    query_emb = embeddings.embed_query(query_text)

    # Hybrid Search: Vector + Keyword
    # 1. Vector Search (Pinecone)
    vector_search_results = pinecone_index.query(
        vector=query_emb,
        top_k=5, # Get a few more results for reranking
        include_metadata=True
    )
    vector_docs = [match['metadata'] for match in vector_search_results['matches']]

    # 2. Keyword Search (BM25)
    tokenized_query = query_text.split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1]
    bm25_docs = [persona_df.iloc[idx].to_dict() for idx in bm25_top_indices if bm25_scores[idx] > 0]
    
    # 3. Combine and remove duplicates
    combined_docs = {doc['pid']: doc for doc in vector_docs}
    for doc in bm25_docs:
        if 'pid' in doc and doc['pid'] not in combined_docs:
            combined_docs[doc['pid']] = doc
    
    combined_docs_list = list(combined_docs.values())
    
    # 4. Rerank the combined results
    reranked_docs, rerank_scores = rerank_snippets(query_text, combined_docs_list, top_n=top_k)
    
    # FIX: Access the 'text' key, which is present in both Pinecone and BM25 results
    top_texts = [d['text'] for d in reranked_docs]
    references = [d['pid'] for d in reranked_docs]

    # Generate message
    message = generate_message_with_ollama(lead, top_texts)

    # Select channel
    channel = POLICY["channel_priority"][0]
    rationale = f"Lead role: {lead.role}, page_views: {lead.page_views}, prior_course_interest: {lead.prior_course_interest}"
    if have_reranker and rerank_scores:
        rationale += f", rerank_scores: {[round(s,3) for s in rerank_scores]}"

    logging.info({"request_id": request_id, "lead": lead_dict, "references": references})

    return {
        "channel": channel,
        "message": message,
        "rationale": rationale,
        "references": references
    }

# ------------------ Register Router ------------------
app.include_router(router, prefix="/recommend")

# ------------------ Root Endpoint ------------------
@app.get("/")
def root():
    return {"message": "Lead HeatScore API with Ollama Recommender is running!"}