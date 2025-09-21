from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

router = APIRouter()

# --- Load persona snippets ---
PERSONA_FILE = os.path.join(os.path.dirname(__file__), "../data/persona_snippets.csv")
persona_df = pd.read_csv(PERSONA_FILE)

# --- Embed persona snippets ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=r"G:\Qeagle\.hf_cache")
persona_embeddings = embed_model.encode(persona_df['snippet'].tolist(), convert_to_numpy=True)

# --- Build FAISS index ---
dim = persona_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(persona_embeddings)

# --- Simple policy ---
POLICY = {
    "tone": "friendly",
    "CTA": "click link to explore course",
    "channel_priority": ["WhatsApp", "Email"]
}

# --- Lead schema ---
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

@router.post("/")
def recommend_action(lead: Lead, top_k: int = 3):
    lead_dict = lead.dict()

    # --- Create query vector from lead info ---
    query_text = f"{lead.role} {lead.campaign}"
    query_vec = embed_model.encode([query_text], convert_to_numpy=True)

    # --- Retrieve top-k snippets ---
    D, I = faiss_index.search(query_vec, top_k)
    top_snippets = persona_df.iloc[I[0]]  # top k rows
    references = top_snippets.index.tolist()

    # --- Craft message using top snippet + policy ---
    combined_snippets = " ".join(top_snippets['snippet'].tolist())
    message = f"Hi! {combined_snippets} Please {POLICY['CTA']}."

    # --- Choose channel ---
    channel = POLICY['channel_priority'][0]

    # --- Rationale ---
    rationale = f"Lead role: {lead.role}, page_views: {lead.page_views}, prior_course_interest: {lead.prior_course_interest}"

    return {
        "channel": channel,
        "message": message,
        "rationale": rationale,
        "references": references
    }
