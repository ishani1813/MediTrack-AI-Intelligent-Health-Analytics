from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


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
    blood_group: Optional[str] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None
    medical_history: Optional[Dict[str, Any]] = {}

class PatientResponse(BaseModel):
    id: int
    patient_code: str
    age: int
    gender: str
    blood_group: Optional[str]
    contact_number: Optional[str]
    address: Optional[str]
    medical_history: Optional[Dict]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Health Records ───────────────────────────────────────────────────────────

class HealthRecordCreate(BaseModel):
    patient_id: int
    blood_pressure_systolic: Optional[int] = Field(None, ge=60, le=250)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150)
    heart_rate: Optional[int] = Field(None, ge=30, le=220)
    blood_glucose: Optional[float] = Field(None, ge=0, le=600)
    bmi: Optional[float] = Field(None, ge=10, le=70)
    cholesterol_total: Optional[float] = None
    cholesterol_hdl: Optional[float] = None
    cholesterol_ldl: Optional[float] = None
    hemoglobin: Optional[float] = None
    temperature: Optional[float] = Field(None, ge=32, le=43)
    oxygen_saturation: Optional[float] = Field(None, ge=50, le=100)
    notes: Optional[str] = None

class HealthRecordResponse(HealthRecordCreate):
    id: int
    recorded_at: datetime

    class Config:
        from_attributes = True


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    patient_id: int
    health_record_id: Optional[int] = None
    # Direct vitals input (if no health_record_id)
    age: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    blood_glucose: Optional[float] = None
    bmi: Optional[float] = None
    cholesterol_total: Optional[float] = None
    cholesterol_hdl: Optional[float] = None
    cholesterol_ldl: Optional[float] = None
    hemoglobin: Optional[float] = None
    oxygen_saturation: Optional[float] = None

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
    top_risk_factors: List[ShapFeature]
    shap_summary: Dict[str, Any]
    model_version: str
    recommendation: str
    cached: bool = False


# ─── RAG Triage ──────────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    patient_id: Optional[int] = None
    symptoms: str = Field(..., min_length=5, max_length=2000)
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    medical_history: Optional[List[str]] = []

class RetrievedDoc(BaseModel):
    condition: str
    relevance_score: float
    excerpt: str

class TriageResponse(BaseModel):
    session_id: int
    urgency_level: UrgencyLevel
    ai_assessment: str
    possible_conditions: List[str]
    recommended_actions: List[str]
    retrieved_references: List[RetrievedDoc]
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
    label: Optional[str] = None

class DashboardResponse(BaseModel):
    stats: DashboardStats
    risk_distribution: RiskDistribution
    risk_trend: List[TrendPoint]
    top_risk_factors_global: List[Dict[str, Any]]
    recent_predictions: List[PredictResponse]
