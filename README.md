# Qeagle Lead Recommendation (Local Ollama + FAISS Pipeline)

This project provides a **local, unlimited, high-quality lead recommendation system** using:

- Ollama LLM (`llama3`) for text generation  
- FAISS for vector similarity search  
- HuggingFace embeddings (`all-MiniLM-L6-v2`)  
- Optional CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`)  

---

## **Setup Instructions**

## 1. Clone repository
```bash
git clone https://github.com/GuruprakashS-cit/A3-Lead-HeatScore.git
cd A3-Lead-HeatScore

##2. Set up virtual environment and activate:
python -m venv venv
#Activate venv
Windows: venv\Scripts\activate


##3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

##4. Install Ollama and download the model

Install Ollama locally from https://ollama.com/download .

Open Ollama app or CLI and run: 
ollama pull llama3

##5. Run the FastAPI server

uvicorn main:app --reload
The API server will start at http://127.0.0.1:8000.