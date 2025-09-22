# scripts/build_persona_faiss.py
import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Adjust cache folder:
# Force HuggingFace/SBERT cache to custom folder
CUSTOM_CACHE = r"G:\Qeagle\.hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CUSTOM_CACHE


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/persona_snippets.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../embeddings")
os.makedirs(OUT_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "persona_faiss.faiss")
METADATA_PATH = os.path.join(OUT_DIR, "persona_docs.pkl")

# Load persona csv
df = pd.read_csv(DATA_PATH)
# Ensure an id column
df = df.reset_index().rename(columns={"index":"pid"})

# Use sentence-transformers for embeddings via LangChain's wrapper (faster offline)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # local-ish model from HF
embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Create Document objects for LangChain FAISS
documents = []
for _, row in df.iterrows():
    meta = {"pid": int(row["pid"]), "role": row["role"]}
    documents.append(Document(page_content=row["snippet"], metadata=meta))

# Build FAISS vectorstore and persist
vectorstore = FAISS.from_documents(documents, embed_model)
vectorstore.save_local(OUT_DIR)

# Save the dataframe metadata so endpoint can map results to csv rows
joblib.dump(df, METADATA_PATH)
print("Saved FAISS index and persona metadata to:", OUT_DIR)
