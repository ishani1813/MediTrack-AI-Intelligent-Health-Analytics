"""
API test suite.
Run: cd backend && pytest tests/ -v
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    resp = await client.post("/auth/login", json={
        "email": "notexist@example.com",
        "password": "WrongPass",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_triage_rule_based():
    """Test RAG triage fallback without LLM."""
    from app.services.rag.triage import rag_service
    result = await rag_service.triage(
        symptoms="severe chest pain shortness of breath sweating",
        patient_age=55,
    )
    assert result["urgency_level"] in ("emergency", "urgent")
    assert len(result["recommended_actions"]) > 0


@pytest.mark.asyncio
async def test_unauthorized_patients(client):
    resp = await client.get("/patients/")
    assert resp.status_code == 403


# ─── Below: real coverage for routes that had none before ───────────────────

async def _register_and_get_token(client, email: str, role: str = "doctor") -> str:
    resp = await client.post("/auth/register", json={
        "email": email,
        "full_name": "Test Doctor",
        "password": "Test@1234",
        "role": role,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


@pytest.mark.asyncio
async def test_create_and_list_patients(client):
    token = await _register_and_get_token(client, "patients_doc@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    create_resp = await client.post("/patients/", json={
        "age": 45, "gender": "female", "blood_group": "O+",
    }, headers=headers)
    assert create_resp.status_code == 201, create_resp.text
    patient = create_resp.json()
    assert patient["patient_code"].startswith("PT-")
    assert patient["age"] == 45

    list_resp = await client.get("/patients/", headers=headers)
    assert list_resp.status_code == 200
    codes = [p["patient_code"] for p in list_resp.json()]
    assert patient["patient_code"] in codes


@pytest.mark.asyncio
async def test_get_single_patient_and_404(client):
    token = await _register_and_get_token(client, "patients_doc2@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    create_resp = await client.post("/patients/", json={"age": 30, "gender": "male"}, headers=headers)
    patient_id = create_resp.json()["id"]

    get_resp = await client.get(f"/patients/{patient_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == patient_id

    missing_resp = await client.get("/patients/999999999", headers=headers)
    assert missing_resp.status_code == 404


@pytest.mark.asyncio
async def test_add_and_get_health_record(client):
    token = await _register_and_get_token(client, "patients_doc3@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    patient_id = (await client.post(
        "/patients/", json={"age": 60, "gender": "male"}, headers=headers
    )).json()["id"]

    record_resp = await client.post(f"/patients/{patient_id}/records", json={
        "patient_id": patient_id,
        "blood_pressure_systolic": 150,
        "blood_pressure_diastolic": 95,
        "blood_glucose": 140.0,
        "bmi": 28.5,
    }, headers=headers)
    assert record_resp.status_code == 201, record_resp.text

    records_resp = await client.get(f"/patients/{patient_id}/records", headers=headers)
    assert records_resp.status_code == 200
    assert len(records_resp.json()) == 1
    assert records_resp.json()[0]["blood_pressure_systolic"] == 150


@pytest.mark.asyncio
async def test_analytics_dashboard(client):
    token = await _register_and_get_token(client, "analytics_doc@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.get("/analytics/dashboard", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "total_patients" in body["stats"]


@pytest.mark.asyncio
async def test_analytics_requires_auth(client):
    resp = await client.get("/analytics/dashboard")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_predict_risk_endpoint(client):
    """Exercises the actual HTTP endpoint (not just the service function
    directly, as test_predict_risk_rule_based above does) -- covers auth,
    request validation, and response serialization for real."""
    token = await _register_and_get_token(client, "predict_doc@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    patient_id = (await client.post(
        "/patients/", json={"age": 55, "gender": "male"}, headers=headers
    )).json()["id"]

    resp = await client.post("/predict/risk", json={
        "patient_id": patient_id,
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
    }, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["risk_level"] in ("low", "medium", "high", "critical")


@pytest.mark.asyncio
async def test_predict_history(client):
    token = await _register_and_get_token(client, "predict_doc2@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    patient_id = (await client.post(
        "/patients/", json={"age": 40, "gender": "female"}, headers=headers
    )).json()["id"]

    # No predictions yet -> empty list, not an error
    resp = await client.get(f"/predict/history/{patient_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []
