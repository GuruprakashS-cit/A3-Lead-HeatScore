# api/recommend_langchain_ollama.py
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import os, joblib, logging, uuid
from typing import List
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from ollama import chat  # Correct usage

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Lead HeatScore API",
              description="Classify leads and generate first personalized outreach message",
              version="1.0")
router = APIRouter()

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(__file__)
EMB_DIR = os.path.join(BASE_DIR, "../embeddings")
PERSONA_META = os.path.join(EMB_DIR, "persona_docs.pkl")

# ------------------ HuggingFace cache setup ------------------
hf_cache_path = os.path.join(BASE_DIR, "../.hf_cache")
os.makedirs(hf_cache_path, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_path
os.environ["TRANSFORMERS_CACHE"] = hf_cache_path
os.environ["HF_DATASETS_CACHE"] = hf_cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_path

# ------------------ Embeddings ------------------
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

# Load persisted FAISS vectorstore
vectorstore = FAISS.load_local(EMB_DIR, embeddings, allow_dangerous_deserialization=True)

# Load persona metadata
persona_df = joblib.load(PERSONA_META)

# ------------------ Cross-Encoder ------------------
try:
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
def rerank_snippets(query: str, docs: List[Document], top_n: int = 3):
    if not have_reranker:
        return docs[:top_n], [None]*min(top_n, len(docs))
    pairs = [(query, d.page_content) for d in docs]
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

    # Call Ollama
    response: ChatResponse = chat(
        model=ollama_model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    # Correct way to get text
    return response.message.content
# ------------------ API Route ------------------
@router.post("/")
def recommend_action(lead: Lead, top_k: int = 3):
    request_id = str(uuid.uuid4())
    lead_dict = lead.dict()

    # Create query
    query_text = f"{lead.role} {lead.campaign} interest:{lead.prior_course_interest} pages:{lead.page_views}"
    query_emb = embeddings.embed_query(query_text)

    # Search
    docs = vectorstore.similarity_search_by_vector(query_emb, k=10)
    reranked_docs, rerank_scores = rerank_snippets(query_text, docs, top_n=top_k)

    top_texts = [d.page_content for d in reranked_docs]
    references = [d.metadata.get("pid") for d in reranked_docs]

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
