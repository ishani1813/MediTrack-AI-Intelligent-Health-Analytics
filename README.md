# AI-Powered Health Checkup & Predictive Analytics Platform

A production-grade full-stack platform combining **LLM-powered symptom triage** (LangChain RAG), **explainable ML predictions** (XGBoost + SHAP), and a **real-time analytics dashboard** (React + FastAPI + Redis).

> Built by Ishani Sarkar — NIT Durgapur, B.Tech CSE 2026

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js + Vite + Tailwind CSS |
| Backend API | FastAPI (async Python) |
| ML Pipeline | XGBoost · Scikit-learn · SHAP |
| LLM / RAG | LangChain · ChromaDB · OpenAI/Ollama |
| Cache | Redis |
| Database | MySQL |
| Auth | JWT + RBAC |
| Infra | Docker Compose |

---

## Features

- **AI Symptom Triage** — LangChain RAG retrieves from a medical knowledge base and generates risk summaries per patient
- **Explainable ML** — XGBoost + Random Forest ensemble with SHAP visualizations per prediction
- **Real-time Dashboard** — Power BI-style KPI cards, trend charts, cohort heatmaps
- **Role-Based Access** — Admin, Doctor, Patient roles with JWT-secured endpoints
- **Redis Caching** — Prediction results cached for <200 ms p95 under 300+ concurrent users
- **One-Command Deploy** — Docker Compose spins up all 5 services

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Node.js 18+
- Python 3.11+

### 1. Clone & configure
```bash
git clone https://github.com/ishani1813/health-ai-platform.git
cd health-ai-platform
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY (or set USE_LOCAL_LLM=true for Ollama)
```

### 2. Launch with Docker Compose
```bash
docker-compose up --build
```

Services start at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redis: localhost:6379
- MySQL: localhost:3306

### 3. Seed sample data
```bash
docker exec -it health_backend python scripts/seed_data.py
```

### 4. Run locally (without Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
health_ai_platform/
├── backend/
│   ├── app/
│   │   ├── api/routes/        # FastAPI routers
│   │   ├── core/              # Config, security, logging
│   │   ├── db/                # DB engine, session, init
│   │   ├── models/            # SQLAlchemy ORM models
│   │   ├── schemas/           # Pydantic request/response schemas
│   │   └── services/
│   │       ├── ml/            # XGBoost + SHAP prediction service
│   │       ├── rag/           # LangChain RAG symptom triage
│   │       └── cache/         # Redis caching layer
│   ├── scripts/               # DB seed, model training scripts
│   └── tests/                 # Pytest test suite
├── frontend/
│   └── src/
│       ├── components/        # Dashboard, Patient, AI components
│       ├── pages/             # Route pages
│       ├── services/          # Axios API clients
│       └── hooks/             # Custom React hooks
├── ml_pipeline/
│   ├── data/                  # Sample datasets
│   ├── models/                # Saved model artifacts
│   └── notebooks/             # EDA + training notebooks
├── docker/                    # Dockerfiles
├── docker-compose.yml
└── .env.example
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/login` | JWT login |
| POST | `/auth/register` | Register new user |
| GET | `/patients/` | List all patients (Doctor/Admin) |
| POST | `/patients/` | Create patient record |
| GET | `/patients/{id}` | Get patient details |
| POST | `/predict/risk` | ML risk prediction + SHAP |
| POST | `/triage/symptom` | LangChain RAG symptom analysis |
| GET | `/analytics/dashboard` | Dashboard KPIs |
| GET | `/analytics/cohort` | Cohort analysis data |

---

## ML Model Performance

| Model | Accuracy | AUC-ROC | F1 |
|---|---|---|---|
| Random Forest (base) | 78.4% | 0.81 | 0.76 |
| XGBoost (base) | 82.1% | 0.86 | 0.80 |
| **Stacked Ensemble** | **85.3%** | **0.89** | **0.84** |

SHAP explanations available per prediction via `/predict/risk` endpoint.

---

## Environment Variables

See `.env.example` for all variables. Key ones:

```env
OPENAI_API_KEY=sk-...          # Or leave blank + set USE_LOCAL_LLM=true
USE_LOCAL_LLM=false            # true = Ollama (free, local)
MYSQL_URL=mysql+aiomysql://...
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key
```

---

## License
MIT
