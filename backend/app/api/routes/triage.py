from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.models.models import TriageSession
from app.schemas.schemas import TriageRequest, TriageResponse, RetrievedDoc
from app.core.security import get_current_user
from app.services.rag.triage import rag_service

router = APIRouter(prefix="/triage", tags=["RAG Triage"])


@router.post("/symptom", response_model=TriageResponse)
async def symptom_triage(
    payload: TriageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await rag_service.triage(
        symptoms=payload.symptoms,
        patient_age=payload.patient_age,
        patient_gender=payload.patient_gender,
        medical_history=payload.medical_history,
    )

    # Persist session
    session = TriageSession(
        patient_id=payload.patient_id,
        symptoms=payload.symptoms,
        rag_response=result.get("ai_assessment"),
        urgency_level=result["urgency_level"],
        retrieved_docs=result.get("retrieved_references", []),
        session_metadata={
            "patient_age": payload.patient_age,
            "patient_gender": payload.patient_gender,
        },
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)

    return TriageResponse(
        session_id=session.id,
        urgency_level=result["urgency_level"],
        ai_assessment=result["ai_assessment"],
        possible_conditions=result["possible_conditions"],
        recommended_actions=result["recommended_actions"],
        retrieved_references=[RetrievedDoc(**r) for r in result.get("retrieved_references", [])],
    )
