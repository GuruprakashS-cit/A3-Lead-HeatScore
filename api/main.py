from fastapi import FastAPI
from api.score import router as score_router
from api.recommend import router as recommend_router

app = FastAPI(
    title="Lead HeatScore API",
    description="Classify leads and generate first personalized outreach message",
    version="1.0"
)

# Include routers
app.include_router(score_router, prefix="/score", tags=["Score"])
app.include_router(recommend_router, prefix="/recommend", tags=["Recommend"])

@app.get("/")
def root():
    return {"message": "Lead HeatScore API is running!"}
