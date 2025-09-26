# Project A3: Lead HeatScore - A RAG-powered Lead Nurturing System

This project is a **lead recommendation system** that uses a combination of a machine learning classifier and a Retrieval-Augmented Generation (RAG) agent to prioritize sales outreach and generate personalized messages. It's designed to run on a local machine for speed and privacy.

**Core Components:**

* **Classifier**: A `scikit-learn` Logistic Regression model to classify leads as **Hot, Warm, or Cold**.
* **RAG System**: A hybrid search pipeline using a **Pinecone vector database** and a **BM25 keyword index**.
* **LLM**: A local Large Language Model (`gemma:2b`) running via Ollama for message generation.
* **Front-end**: A React application to interact with the API, score leads, and visualize metrics.

### Setup Instructions

Follow these steps to get the project up and running locally.

#### 1. Clone the repository

```bash
git clone [https://github.com/GuruprakashS-cit/A3-Lead-HeatScore.git](https://github.com/GuruprakashS-cit/A3-Lead-HeatScore.git)
cd A3-Lead-HeatScore

 #### 2. Set up and activate the Python virtual environment
   python -m venv venv
   # Activate the virtual environment
   venv\Scripts\activate

 ####3. Install Python dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

 ####4. Set up Ollama and download the language model
    Install Ollama locally from [https://ollama.com/download.]

    Once installed, open a terminal and pull the required lightweight model:

    ollama pull gemma:2b

 ####5. Configure and build the RAG knowledge base
    You will need a Pinecone API key and environment. Log in to your Pinecone account to get these values.

    Open src/build_persona_hybrid.py.

    Replace "YOUR_PINECONE_API_KEY" and "YOUR_PINECONE_ENVIRONMENT" with your actual credentials.

    Run the script to build and upload your knowledge base to Pinecone and create the local BM25 index.

 ####6. Train the classifier model
    This script will train the model and save the necessary files for the API.
    
    python src/train_classifier.py

 ####7. Run the FastAPI server
    This starts the backend API, which serves both the classifier and the RAG agent.

    uvicorn main:app --reload

    The API server will be available at [http://127.0.0.1:8000.]

 ####8. Run the React front-end
    Open a new terminal window in the project's root folder (A3-Lead-HeatScore) and run the following commands:
    
    cd frontend
    cd heatscore-ui
    npm install
    npm run dev

    The front-end application will start at [http://localhost:5173.]


 ####9. Project Architecture & Reasoning
    The project follows a modular, three-part architecture:

    Data Processing (Offline): Scripts for generating the synthetic dataset and building the hybrid search indexes.

    Backend API: A FastAPI server that exposes endpoints for scoring leads and generating personalized messages.

    Frontend UI: A React application for user interaction and visualization.

    The core of the system is the hybrid retrieval pipeline. A user's query is simultaneously sent to a vector search (for semantic understanding) and a keyword search (for exact matches). The results from both are then combined and re-ranked by a CrossEncoder to ensure the LLM receives only the most relevant and accurate context.

    Trade-offs & Decisions:

    Local LLM: The decision to use a local LLM (Ollama) was made to prioritize cost control and data privacy. It ensures no data is sent to a third-party service.

    Model Selection: While larger models are more powerful, a smaller model like gemma:2b was chosen to meet the strict P95 latency quality bar of 2.5 seconds on consumer-grade hardware.

    Evaluation & Metrics
    The project includes an /metrics API endpoint that provides detailed evaluation data on a held-out test set, including:

    F1 Score (macro): A comprehensive metric to measure overall model performance.

    Confusion Matrix: A table that visualizes the model's classification accuracy.

    Brier Score & ROC Curves: Metrics to evaluate the model's calibration and ability to distinguish between classes.

    The logs written to demo/app.log provide granular data for every request, allowing for a reproducible cost and latency report.






