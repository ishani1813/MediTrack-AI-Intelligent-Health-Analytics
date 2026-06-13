"""
API test suite.
Run: cd backend && pytest tests/ -v
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_register_and_login(client):
    # Register
    resp = await client.post("/auth/register", json={
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "Test@1234",
        "role": "doctor",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    assert token

    # Login
    resp2 = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "Test@1234",
    })
    assert resp2.status_code == 200
    assert resp2.json()["access_token"]


@pytest.mark.anyio
async def test_login_wrong_password(client):
    resp = await client.post("/auth/login", json={
        "email": "notexist@example.com",
        "password": "WrongPass",
    })
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_predict_risk_rule_based(client):
    """Test ML prediction with rule-based fallback (no trained models needed)."""
    from app.services.ml.predictor import ml_service
    result = await ml_service.predict({
        "age": 55,
        "blood_pressure_systolic": 165,
        "blood_pressure_diastolic": 100,
        "heart_rate": 88,
        "blood_glucose": 145,
        "bmi": 31.5,
        "cholesterol_total": 245,
        "cholesterol_hdl": 38,
        "cholesterol_ldl": 165,
        "hemoglobin": 12.5,
        "oxygen_saturation": 95,
    })
    assert "risk_score" in result
    assert result["risk_level"] in ("low", "medium", "high", "critical")
    assert len(result["top_risk_factors"]) > 0
    assert result["risk_score"] > 0.0


@pytest.mark.anyio
async def test_predict_low_risk(client):
    """Healthy vitals should produce low risk."""
    from app.services.ml.predictor import ml_service
    result = await ml_service.predict({
        "age": 25,
        "blood_pressure_systolic": 115,
        "blood_pressure_diastolic": 75,
        "heart_rate": 70,
        "blood_glucose": 88,
        "bmi": 21.0,
        "cholesterol_total": 175,
        "cholesterol_hdl": 65,
        "cholesterol_ldl": 95,
        "hemoglobin": 14.5,
        "oxygen_saturation": 99,
    })
    assert result["risk_level"] in ("low", "medium")


@pytest.mark.anyio
async def test_triage_rule_based():
    """Test RAG triage fallback without LLM."""
    from app.services.rag.triage import rag_service
    result = await rag_service.triage(
        symptoms="severe chest pain shortness of breath sweating",
        patient_age=55,
    )
    assert result["urgency_level"] in ("emergency", "urgent")
    assert len(result["recommended_actions"]) > 0


@pytest.mark.anyio
async def test_unauthorized_patients(client):
    resp = await client.get("/patients/")
    assert resp.status_code == 403
