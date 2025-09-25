# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import the middleware

# Import routers
from api.score import router as score_router
from api.recommend_langchain_ollama import router as recommend_router 

app = FastAPI(
    title="Lead HeatScore API",
    description="Classify leads and generate first personalized outreach message",
    version="1.0"
)

# Define a list of allowed origins. This is your frontend's URL.
# This ensures only your frontend can access the API.
origins = [
    "http://localhost:5173", # The URL where your Vite development server runs
]

# Add the CORS middleware to your main application instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows requests from the specified frontend origin
    allow_credentials=True, # Allows cookies and authorization headers
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allows all headers
)

# Include routers
app.include_router(score_router, prefix="/score", tags=["Score"])
app.include_router(recommend_router, prefix="/recommend", tags=["Recommend"])

@app.get("/")
def root():
    return {"message": "Lead HeatScore API is running!"}