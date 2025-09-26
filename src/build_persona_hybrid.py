import os
import pandas as pd
import joblib
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
# The RecursiveCharacterTextSplitter import is no longer needed
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ Pinecone Setup ------------------
API_KEY = "pcsk_5VSpwm_16JSye9ViLTq6D4KQL1mmAdf9An5SEsJPJCUoyP6tYrVCi8hSVg3nKJ6ZzsMact"
ENVIRONMENT = "us-east-1"
INDEX_NAME = "lead-heatscore-qeagle"

# ------------------ Path Setup ------------------
# This path is now correctly updated to point to your CSV file
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/persona_snippets.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../embeddings")
os.makedirs(OUT_DIR, exist_ok=True)
BM25_INDEX_PATH = os.path.join(OUT_DIR, "bm25_index.pkl")
METADATA_PATH = os.path.join(OUT_DIR, "persona_docs.pkl")

# Force HuggingFace cache to a custom folder
CUSTOM_CACHE = r"G:\Qeagle\.hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)

# ------------------ Model & Data Loading ------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, cache_folder=CUSTOM_CACHE)

# ------------------ Load from CSV directly, no chunking ------------------
print("Reading persona snippets from CSV...")
df = pd.read_csv(DATA_PATH)
df = df.reset_index().rename(columns={"index": "pid"})
documents = df["snippet"].tolist()
print(f"Loaded {len(documents)} persona snippets.")

# ------------------ Build and Upload to Pinecone ------------------
print("Connecting to Pinecone...")
pc = Pinecone(api_key=API_KEY)

# Check for index existence using a more robust method
try:
    pc.describe_index(INDEX_NAME)
    print(f"Index '{INDEX_NAME}' already exists.")
except Exception as e:
    # If the describe_index call fails, the index does not exist.
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=embed_model._client.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region=ENVIRONMENT)
    )

index = pc.Index(INDEX_NAME)

print("Generating embeddings and uploading to Pinecone...")
vectors_to_upsert = []
for i, row in df.iterrows():
    vector = embed_model.embed_documents([row["snippet"]])[0]
    metadata = {"pid": i, "role": row["role"], "text": row["snippet"]}
    vectors_to_upsert.append((str(i), vector, metadata))

index.upsert(vectors=vectors_to_upsert, batch_size=100)
print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

# ------------------ Build and Save BM25 Index ------------------
print("Building and saving BM25 index...")
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# FIX: Changed to correctly reference the documents list of strings
df_from_chunks = pd.DataFrame([{"pid": i, "text": doc} for i, doc in enumerate(documents)])

joblib.dump(bm25, BM25_INDEX_PATH)
joblib.dump(df, METADATA_PATH)
print("Saved BM25 index and persona metadata locally.")
