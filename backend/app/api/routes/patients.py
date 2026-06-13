import random, string
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db.database import get_db
from app.models.models import Patient, HealthRecord
from app.schemas.schemas import PatientCreate, PatientResponse, HealthRecordCreate, HealthRecordResponse
from app.core.security import get_current_user, require_doctor_or_admin

router = APIRouter(prefix="/patients", tags=["Patients"])


def _generate_code() -> str:
    return "PT-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=7))


@router.post("/", response_model=PatientResponse, status_code=201)
async def create_patient(
    payload: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    patient = Patient(
        **payload.model_dump(),
        patient_code=_generate_code(),
        user_id=None,
    )
    db.add(patient)
    await db.flush()
    await db.refresh(patient)
    return patient


@router.get("/", response_model=list[PatientResponse])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    result = await db.execute(select(Patient).offset(skip).limit(limit))
    return result.scalars().all()


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    payload: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(patient, field, value)
    await db.flush()
    await db.refresh(patient)
    return patient


# ── Health Records ───────────────────────────────────────────────────────────

@router.post("/{patient_id}/records", response_model=HealthRecordResponse, status_code=201)
async def add_health_record(
    patient_id: int,
    payload: HealthRecordCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    record = HealthRecord(
        **payload.model_dump(),
        patient_id=patient_id,
        recorded_by=int(current_user["sub"]),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


@router.get("/{patient_id}/records", response_model=list[HealthRecordResponse])
async def list_health_records(
    patient_id: int,
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(HealthRecord)
        .where(HealthRecord.patient_id == patient_id)
        .order_by(HealthRecord.recorded_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
