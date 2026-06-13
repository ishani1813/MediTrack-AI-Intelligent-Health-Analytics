from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, JSON,
    Enum, ForeignKey, TIMESTAMP, func
)
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


class UserRole(str, enum.Enum):
    admin = "admin"
    doctor = "doctor"
    patient = "patient"


class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class UrgencyLevel(str, enum.Enum):
    routine = "routine"
    soon = "soon"
    urgent = "urgent"
    emergency = "emergency"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.patient)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="user", uselist=False)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    patient_code = Column(String(20), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(Enum("male", "female", "other"), nullable=False)
    blood_group = Column(String(5))
    contact_number = Column(String(15))
    address = Column(Text)
    medical_history = Column(JSON, default={})
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="patient")
    health_records = relationship("HealthRecord", back_populates="patient", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")
    triage_sessions = relationship("TriageSession", back_populates="patient")


class HealthRecord(Base):
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    recorded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    heart_rate = Column(Integer)
    blood_glucose = Column(Float)
    bmi = Column(Float)
    cholesterol_total = Column(Float)
    cholesterol_hdl = Column(Float)
    cholesterol_ldl = Column(Float)
    hemoglobin = Column(Float)
    temperature = Column(Float)
    oxygen_saturation = Column(Float)
    notes = Column(Text)
    recorded_at = Column(TIMESTAMP, server_default=func.now())

    patient = relationship("Patient", back_populates="health_records")
    predictions = relationship("Prediction", back_populates="health_record")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    health_record_id = Column(Integer, ForeignKey("health_records.id", ondelete="SET NULL"))
    risk_score = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    model_version = Column(String(20), nullable=False)
    shap_values = Column(JSON)
    top_risk_factors = Column(JSON)
    prediction_metadata = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())

    patient = relationship("Patient", back_populates="predictions")
    health_record = relationship("HealthRecord", back_populates="predictions")


class TriageSession(Base):
    __tablename__ = "triage_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="SET NULL"))
    symptoms = Column(Text, nullable=False)
    rag_response = Column(Text)
    urgency_level = Column(Enum(UrgencyLevel), nullable=False)
    retrieved_docs = Column(JSON)
    session_metadata = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())

    patient = relationship("Patient", back_populates="triage_sessions")


class MedicalKnowledge(Base):
    __tablename__ = "medical_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), nullable=False, index=True)
    condition_name = Column(String(255), nullable=False)
    symptoms = Column(Text, nullable=False)
    risk_factors = Column(Text)
    description = Column(Text)
    urgency_guidelines = Column(Text)
    embedding_id = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())
