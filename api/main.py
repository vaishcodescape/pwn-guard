"""
High-performance REST API: spam detection, scam classification, threat intel,
rule-based escalation simulation, risk score and recommended action.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_VERSION
from core.model_loader import load_models
from api.routes import health, analyze, simulation, reports, reference, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("=" * 50)
    print("Starting Spam Detection API...")
    print("=" * 50)
    load_models()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Spam Detection & Scam Simulation API",
    description="REST API: spam detection, scam classification, threat intel, rule-based escalation, risk score. CPU-only, Docker & GCP ready.",
    version=API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(simulation.router)
app.include_router(reports.router)
app.include_router(reference.router)
app.include_router(chat.router)
