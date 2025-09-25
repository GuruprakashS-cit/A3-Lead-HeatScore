from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from api.score import router as score_router
from api.recommend_langchain_ollama import router as recommend_router
from api.metrics import router as metrics_router # New import for the metrics router

app = FastAPI(
    title="Lead HeatScore API",
    description="Classify leads and generate first personalized outreach message",
    version="1.0"
)

# Define a list of allowed origins. This is your frontend's URL.
origins = [
    "http://localhost:5173",
]

# Add the CORS middleware to your main application instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(score_router, prefix="/score", tags=["Score"])
app.include_router(recommend_router, prefix="/recommend", tags=["Recommend"])
app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"]) # Router for the new metrics endpoint

@app.get("/")
def root():
    return {"message": "Lead HeatScore API is running!"}
