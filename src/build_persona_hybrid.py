import os
import pandas as pd
import joblib
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ Pinecone Setup ------------------
API_KEY = "pcsk_5VSpwm_16JSye9ViLTq6D4KQL1mmAdf9An5SEsJPJCUoyP6tYrVCi8hSVg3nKJ6ZzsMact"
ENVIRONMENT = "us-east-1"
INDEX_NAME = "lead-heatscore-qeagle"

# ------------------ Path Setup ------------------
# This path is now updated to point to your dictionary.md file.
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/dictionary.md")
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

# ------------------ Perform Chunking ------------------
print("Reading knowledge base and performing semantic chunking...")
# Read the entire knowledge base file
with open(DATA_PATH, "r", encoding="utf-8") as f:
    knowledge_base_text = f.read()

# Create a text splitter that splits based on Markdown headings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Adjust size as needed
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
documents = text_splitter.create_documents([knowledge_base_text])
print(f"Created {len(documents)} chunks from the knowledge base.")

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
for i, doc in enumerate(documents):
    vector = embed_model.embed_documents([doc.page_content])[0]
    metadata = {"pid": i, "role": "persona", "text": doc.page_content}
    vectors_to_upsert.append((str(i), vector, metadata))

index.upsert(vectors=vectors_to_upsert, batch_size=100)
print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

# ------------------ Build and Save BM25 Index ------------------
print("Building and saving BM25 index...")
tokenized_corpus = [doc.page_content.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

df_from_chunks = pd.DataFrame([{"pid": i, "text": doc.page_content} for i, doc in enumerate(documents)])

joblib.dump(bm25, BM25_INDEX_PATH)
joblib.dump(df_from_chunks, METADATA_PATH)
print("Saved BM25 index and persona metadata locally.")