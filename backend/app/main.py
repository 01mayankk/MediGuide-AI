from fastapi import FastAPI
from backend.app.routes import predict, health

app = FastAPI(
    title="MediGuide-AI",
    description="AI-powered healthcare risk assessment",
    version="1.0.0"
)

app.include_router(predict.router)
app.include_router(predict.router)
app.include_router(health.router)