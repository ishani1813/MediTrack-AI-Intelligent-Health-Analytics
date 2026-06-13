from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import app_logger
from app.db.database import init_db
from app.api.routes import auth, patients, predict, triage, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Starting Health AI Platform...")
    try:
        await init_db()
        app_logger.info("Database initialized")
    except Exception as e:
        app_logger.warning(f"DB init skipped (may already exist): {e}")
    yield
    app_logger.info("Shutting down...")


app = FastAPI(
    title="Health AI Platform API",
    description="AI-Powered Health Checkup & Predictive Analytics Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(predict.router)
app.include_router(triage.router)
app.include_router(analytics.router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Health AI Platform",
        "docs": "/docs",
        "version": "1.0.0",
    }
