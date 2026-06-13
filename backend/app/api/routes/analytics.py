from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from datetime import datetime, timedelta
from app.db.database import get_db
from app.models.models import Patient, Prediction, TriageSession, HealthRecord
from app.schemas.schemas import DashboardResponse, DashboardStats, RiskDistribution, TrendPoint
from app.core.security import require_doctor_or_admin
from app.services.cache.redis_cache import cache_get, cache_set

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    cache_key = "analytics:dashboard"
    cached = await cache_get(cache_key)
    if cached:
        return DashboardResponse(**cached)

    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0)
    today_start = now.replace(hour=0, minute=0, second=0)

    # KPI queries
    total_patients = (await db.execute(select(func.count(Patient.id)))).scalar() or 0
    records_this_month = (
        await db.execute(
            select(func.count(HealthRecord.id)).where(HealthRecord.recorded_at >= month_start)
        )
    ).scalar() or 0
    high_risk = (
        await db.execute(
            select(func.count(Prediction.id)).where(
                Prediction.risk_level.in_(["high", "critical"])
            )
        )
    ).scalar() or 0
    avg_risk = (
        await db.execute(select(func.avg(Prediction.risk_score)))
    ).scalar() or 0.0
    triage_today = (
        await db.execute(
            select(func.count(TriageSession.id)).where(TriageSession.created_at >= today_start)
        )
    ).scalar() or 0

    # Risk distribution
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    rows = (
        await db.execute(
            select(Prediction.risk_level, func.count(Prediction.id))
            .group_by(Prediction.risk_level)
        )
    ).all()
    for level, cnt in rows:
        risk_counts[str(level)] = cnt

    # 30-day risk trend
    trend_rows = (
        await db.execute(
            select(
                func.date(Prediction.created_at).label("day"),
                func.avg(Prediction.risk_score).label("avg_score"),
            )
            .where(Prediction.created_at >= now - timedelta(days=30))
            .group_by("day")
            .order_by("day")
        )
    ).all()
    risk_trend = [
        TrendPoint(date=str(row.day), value=round(float(row.avg_score), 3))
        for row in trend_rows
    ]

    # Top global SHAP factors (aggregate from recent predictions)
    recent_preds = (
        await db.execute(
            select(Prediction.top_risk_factors)
            .where(Prediction.top_risk_factors.isnot(None))
            .order_by(Prediction.created_at.desc())
            .limit(100)
        )
    ).scalars().all()

    factor_agg: dict = {}
    for factors in recent_preds:
        if not factors:
            continue
        for f in factors:
            name = f.get("feature", "")
            val = abs(f.get("shap_value", 0))
            if name:
                factor_agg[name] = factor_agg.get(name, 0) + val

    top_factors = sorted(
        [{"feature": k, "avg_shap": round(v / max(len(recent_preds), 1), 4)} for k, v in factor_agg.items()],
        key=lambda x: x["avg_shap"],
        reverse=True,
    )[:8]

    stats = DashboardStats(
        total_patients=total_patients,
        records_this_month=records_this_month,
        high_risk_count=high_risk,
        avg_risk_score=round(float(avg_risk), 3),
        triage_sessions_today=triage_today,
    )
    distribution = RiskDistribution(**risk_counts)

    response = DashboardResponse(
        stats=stats,
        risk_distribution=distribution,
        risk_trend=risk_trend,
        top_risk_factors_global=top_factors,
        recent_predictions=[],
    )

    await cache_set(cache_key, response.model_dump(), ttl=120)
    return response


@router.get("/cohort")
async def cohort_analysis(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(require_doctor_or_admin),
):
    """Age-group × risk level cohort breakdown."""
    rows = (
        await db.execute(
            select(
                Patient.age,
                Prediction.risk_level,
                func.count(Prediction.id).label("count"),
            )
            .join(Prediction, Prediction.patient_id == Patient.id)
            .group_by(Patient.age, Prediction.risk_level)
        )
    ).all()

    cohorts: dict = {}
    for age, level, count in rows:
        bracket = _age_bracket(age)
        if bracket not in cohorts:
            cohorts[bracket] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        cohorts[bracket][str(level)] = count

    return {"cohorts": cohorts}


def _age_bracket(age: int) -> str:
    if age < 30:
        return "18-29"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    elif age < 75:
        return "60-74"
    return "75+"
