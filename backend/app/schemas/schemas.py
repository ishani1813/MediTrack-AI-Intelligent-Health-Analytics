from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, EmailStr, Field

# ─── Enums ───────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    admin = "admin"
    doctor = "doctor"
    patient = "patient"

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class UrgencyLevel(str, Enum):
    routine = "routine"
    soon = "soon"
    urgent = "urgent"
    emergency = "emergency"


# ─── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.patient

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str
    full_name: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Patient ─────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")
    blood_group: str | None = None
    contact_number: str | None = None
    address: str | None = None
    medical_history: dict[str, Any] | None = {}

class PatientResponse(BaseModel):
    id: int
    patient_code: str
    age: int
    gender: str
    blood_group: str | None
    contact_number: str | None
    address: str | None
    medical_history: dict | None
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Health Records ───────────────────────────────────────────────────────────

class HealthRecordCreate(BaseModel):
    patient_id: int
    blood_pressure_systolic: int | None = Field(None, ge=60, le=250)
    blood_pressure_diastolic: int | None = Field(None, ge=40, le=150)
    heart_rate: int | None = Field(None, ge=30, le=220)
    blood_glucose: float | None = Field(None, ge=0, le=600)
    bmi: float | None = Field(None, ge=10, le=70)
    cholesterol_total: float | None = None
    cholesterol_hdl: float | None = None
    cholesterol_ldl: float | None = None
    hemoglobin: float | None = None
    temperature: float | None = Field(None, ge=32, le=43)
    oxygen_saturation: float | None = Field(None, ge=50, le=100)
    notes: str | None = None

class HealthRecordResponse(HealthRecordCreate):
    id: int
    recorded_at: datetime

    class Config:
        from_attributes = True


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    patient_id: int
    health_record_id: int | None = None
    # Direct vitals input (if no health_record_id)
    age: int | None = None
    blood_pressure_systolic: int | None = None
    blood_pressure_diastolic: int | None = None
    heart_rate: int | None = None
    blood_glucose: float | None = None
    bmi: float | None = None
    cholesterol_total: float | None = None
    cholesterol_hdl: float | None = None
    cholesterol_ldl: float | None = None
    hemoglobin: float | None = None
    oxygen_saturation: float | None = None

class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str  # "increases_risk" | "decreases_risk"

class PredictResponse(BaseModel):
    prediction_id: int
    patient_id: int
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    top_risk_factors: list[ShapFeature]
    shap_summary: dict[str, Any]
    model_version: str
    recommendation: str
    cached: bool = False


# ─── RAG Triage ──────────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    patient_id: int | None = None
    symptoms: str = Field(..., min_length=5, max_length=2000)
    patient_age: int | None = None
    patient_gender: str | None = None
    medical_history: list[str] | None = []

class RetrievedDoc(BaseModel):
    condition: str
    relevance_score: float
    excerpt: str

class TriageResponse(BaseModel):
    session_id: int
    urgency_level: UrgencyLevel
    ai_assessment: str
    possible_conditions: list[str]
    recommended_actions: list[str]
    retrieved_references: list[RetrievedDoc]
    disclaimer: str = "This is an AI-assisted triage, not a medical diagnosis. Please consult a qualified physician."


# ─── Analytics ───────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_patients: int
    records_this_month: int
    high_risk_count: int
    avg_risk_score: float
    triage_sessions_today: int

class RiskDistribution(BaseModel):
    low: int
    medium: int
    high: int
    critical: int

class TrendPoint(BaseModel):
    date: str
    value: float
    label: str | None = None

class DashboardResponse(BaseModel):
    stats: DashboardStats
    risk_distribution: RiskDistribution
    risk_trend: list[TrendPoint]
    top_risk_factors_global: list[dict[str, Any]]
    recent_predictions: list[PredictResponse]
