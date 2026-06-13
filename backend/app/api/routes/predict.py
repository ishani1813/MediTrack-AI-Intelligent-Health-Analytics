from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import get_db
from app.models.models import Patient, HealthRecord, Prediction
from app.schemas.schemas import PredictRequest, PredictResponse, ShapFeature
from app.core.security import get_current_user
from app.services.ml.predictor import ml_service
from app.services.cache.redis_cache import cache_get, cache_set, make_cache_key

router = APIRouter(prefix="/predict", tags=["ML Prediction"])


@router.post("/risk", response_model=PredictResponse)
async def predict_risk(
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Validate patient exists
    p_result = await db.execute(select(Patient).where(Patient.id == payload.patient_id))
    patient = p_result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Build input features dict
    if payload.health_record_id:
        r_result = await db.execute(
            select(HealthRecord).where(HealthRecord.id == payload.health_record_id)
        )
        record = r_result.scalar_one_or_none()
        if not record:
            raise HTTPException(status_code=404, detail="Health record not found")
        raw_input = {
            "age": patient.age,
            "blood_pressure_systolic": record.blood_pressure_systolic,
            "blood_pressure_diastolic": record.blood_pressure_diastolic,
            "heart_rate": record.heart_rate,
            "blood_glucose": record.blood_glucose,
            "bmi": record.bmi,
            "cholesterol_total": record.cholesterol_total,
            "cholesterol_hdl": record.cholesterol_hdl,
            "cholesterol_ldl": record.cholesterol_ldl,
            "hemoglobin": record.hemoglobin,
            "oxygen_saturation": record.oxygen_saturation,
        }
    else:
        raw_input = payload.model_dump(exclude={"patient_id", "health_record_id"})
        raw_input["age"] = raw_input.get("age") or patient.age

    # Check Redis cache
    cache_key = make_cache_key(f"predict:p{payload.patient_id}", raw_input)
    cached = await cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return PredictResponse(**cached)

    # Run ML pipeline
    result = await ml_service.predict(raw_input)

    # Persist prediction to DB
    prediction_record = Prediction(
        patient_id=payload.patient_id,
        health_record_id=payload.health_record_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        model_version=result["model_version"],
        shap_values={f["feature"]: f["shap_value"] for f in result["top_risk_factors"]},
        top_risk_factors=result["top_risk_factors"],
        prediction_metadata={"confidence": result["confidence"]},
    )
    db.add(prediction_record)
    await db.flush()
    await db.refresh(prediction_record)

    response = PredictResponse(
        prediction_id=prediction_record.id,
        patient_id=payload.patient_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        top_risk_factors=[ShapFeature(**f) for f in result["top_risk_factors"]],
        shap_summary=result["shap_summary"],
        model_version=result["model_version"],
        recommendation=result["recommendation"],
        cached=False,
    )

    # Cache result
    await cache_set(cache_key, response.model_dump())

    return response


@router.get("/history/{patient_id}")
async def prediction_history(
    patient_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    predictions = result.scalars().all()
    return [
        {
            "id": p.id,
            "risk_score": p.risk_score,
            "risk_level": p.risk_level,
            "model_version": p.model_version,
            "top_risk_factors": p.top_risk_factors,
            "created_at": p.created_at,
        }
        for p in predictions
    ]
