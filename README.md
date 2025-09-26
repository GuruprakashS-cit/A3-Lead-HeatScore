# Project A3: Lead HeatScore - A RAG-powered Lead Nurturing System

This project is a **lead recommendation system** that combines a machine learning classifier with a Retrieval-Augmented Generation (RAG) agent to **prioritize sales outreach** and **generate personalized messages**. It is designed to run locally for **speed** and **privacy**.

---

## 🚀 Core Components

- **Classifier**: A `scikit-learn` Logistic Regression model to classify leads as **Hot**, **Warm**, or **Cold**.  
- **RAG System**: A hybrid search pipeline using a **Pinecone vector database** and a **BM25 keyword index**.  
- **LLM**: A local Large Language Model (`gemma:2b`) running via **Ollama** for personalized message generation.  
- **Front-end**: A **React** application for interacting with the API, scoring leads, and visualizing metrics.  

---

## 🛠️ Setup Instructions

Follow these steps to get the project up and running locally:

### 1. Clone the repository
```bash
git clone https://github.com/GuruprakashS-cit/A3-Lead-HeatScore.git
cd A3-Lead-HeatScore
```

### 2. Set up and activate the Python virtual environment
```bash
python -m venv venv
# Activate the virtual environment
venv\Scripts\activate   # Windows
```

### 3. Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Ollama and download the language model
- Install Ollama locally from: [https://ollama.com/download](https://ollama.com/download)  
- Once installed, open a terminal and pull the required lightweight model:
```bash
ollama pull gemma:2b
```

### 5. Configure and build the RAG knowledge base
- You will need a **Pinecone API key** and **environment**. Log in to your Pinecone account to get these values.  
- Open `src/build_persona_hybrid.py`.  
- Replace the placeholders:
  ```python
  "YOUR_PINECONE_API_KEY"
  "YOUR_PINECONE_ENVIRONMENT"
  ```
- Run the script:
```bash
python src/build_persona_hybrid.py
```

This will build and upload your knowledge base to Pinecone and create the local BM25 index.

### 6. Train the classifier model
```bash
python src/train_classifier.py
```
This will train the Logistic Regression model and save the necessary files for the API.

### 7. Run the FastAPI server
```bash
uvicorn main:app --reload
```
The API server will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 8. Run the React front-end
Open a new terminal window in the project root (`A3-Lead-HeatScore`) and run:
```bash
cd frontend/heatscore-ui
npm install
npm run dev
```
The front-end application will be available at: [http://localhost:5173](http://localhost:5173)

---

## 🏗️ Project Architecture & Reasoning

The project follows a modular **three-part architecture**:

1. **Data Processing (Offline)**  
   Scripts for generating the synthetic dataset and building the hybrid search indexes.

2. **Backend API**  
   A FastAPI server that exposes endpoints for scoring leads and generating personalized messages.

3. **Frontend UI**  
   A React application for user interaction and visualization.

🔎 **Core of the system**:  
A user's query is simultaneously sent to:
- **Vector Search** (semantic understanding)  
- **Keyword Search** (exact matches)  

The results from both are combined and re-ranked by a **CrossEncoder**, ensuring the LLM receives only the most relevant and accurate context.

---

## ⚖️ Trade-offs & Decisions

- **Local LLM**: Chosen for **cost control** and **data privacy**, ensuring no data is sent to third-party services.  
- **Model Selection**: A smaller model (`gemma:2b`) was chosen to meet the **P95 latency bar of 2.5s** on consumer-grade hardware.  

---

## 📊 Evaluation & Metrics

The project includes a `/metrics` API endpoint that provides detailed evaluation data on a held-out test set:

- **F1 Score (macro)** → overall performance  
- **Confusion Matrix** → classification accuracy visualization  
- **Brier Score & ROC Curves** → calibration and class separation ability  

Additionally, logs are written to `demo/app.log`, providing granular data for every request. This enables **reproducible cost and latency reports**.

---
