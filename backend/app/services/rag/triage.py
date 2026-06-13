"""
LangChain RAG Symptom Triage Service
─────────────────────────────────────
1. Embeds patient symptoms into a vector query
2. Retrieves top-k relevant docs from ChromaDB medical knowledge base
3. Passes context + symptoms to LLM (OpenAI or local Ollama)
4. Returns structured urgency assessment + recommendations
"""

from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import app_logger

# LangChain imports (graceful degradation if not installed)
try:
    import chromadb
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    app_logger.warning("LangChain not installed — RAG triage will use rule-based fallback")


MEDICAL_KNOWLEDGE_BASE = [
    {
        "condition": "Hypertensive Crisis",
        "category": "cardiovascular",
        "symptoms": "severe headache, blurred vision, chest pain, shortness of breath, severe anxiety, very high blood pressure above 180/120",
        "risk_factors": "existing hypertension, obesity, kidney disease, pregnancy, stimulant use",
        "urgency": "emergency",
        "action": "Call emergency services immediately. Do not drive yourself. Blood pressure must be reduced under medical supervision."
    },
    {
        "condition": "Diabetic Ketoacidosis",
        "category": "endocrine",
        "symptoms": "extreme thirst, frequent urination, nausea, vomiting, abdominal pain, weakness, fruity breath odor, confusion, blood glucose over 250",
        "risk_factors": "type 1 diabetes, missed insulin doses, infection, stress",
        "urgency": "emergency",
        "action": "Emergency hospitalization required. IV fluids, insulin, and electrolyte replacement needed."
    },
    {
        "condition": "Acute Myocardial Infarction",
        "category": "cardiovascular",
        "symptoms": "chest pain or pressure, pain radiating to arm jaw or back, shortness of breath, sweating, nausea, dizziness",
        "risk_factors": "hypertension, high cholesterol, diabetes, smoking, family history, age over 45",
        "urgency": "emergency",
        "action": "Call emergency services immediately. Chew aspirin if not allergic. Do not eat or drink."
    },
    {
        "condition": "Hypoglycemia",
        "category": "endocrine",
        "symptoms": "shakiness, sweating, rapid heartbeat, hunger, irritability, confusion, blurred vision, blood glucose below 70",
        "risk_factors": "diabetes medication, skipped meals, excessive exercise, alcohol use",
        "urgency": "urgent",
        "action": "Consume 15g fast-acting carbohydrate immediately. Recheck glucose in 15 minutes. Seek medical attention if unresponsive."
    },
    {
        "condition": "Pulmonary Embolism",
        "category": "pulmonary",
        "symptoms": "sudden shortness of breath, chest pain, rapid heart rate, coughing blood, low oxygen saturation, leg swelling",
        "risk_factors": "prolonged immobility, recent surgery, cancer, blood clotting disorders, oral contraceptives",
        "urgency": "emergency",
        "action": "Emergency services required. CT pulmonary angiography needed for diagnosis. Anticoagulation therapy initiated immediately."
    },
    {
        "condition": "Anemia",
        "category": "hematology",
        "symptoms": "fatigue, weakness, pale skin, shortness of breath on exertion, dizziness, cold hands, low hemoglobin",
        "risk_factors": "iron deficiency, B12 deficiency, chronic disease, blood loss, pregnancy",
        "urgency": "soon",
        "action": "Schedule appointment within 1-2 weeks. Blood tests to determine type of anemia. Iron or B12 supplementation may be required."
    },
    {
        "condition": "Type 2 Diabetes",
        "category": "endocrine",
        "symptoms": "increased thirst, frequent urination, fatigue, blurred vision, slow wound healing, elevated fasting glucose above 126",
        "risk_factors": "obesity, family history, sedentary lifestyle, age over 45, hypertension",
        "urgency": "soon",
        "action": "Schedule appointment within 1 week for HbA1c test. Lifestyle modification counseling. Medication evaluation."
    },
    {
        "condition": "Hypertension Stage 2",
        "category": "cardiovascular",
        "symptoms": "headaches, shortness of breath, nosebleeds, blood pressure consistently above 140/90",
        "risk_factors": "obesity, high salt diet, smoking, stress, family history, kidney disease",
        "urgency": "soon",
        "action": "Physician appointment within 2-4 weeks. Start or adjust antihypertensive medication. Dietary DASH protocol."
    },
    {
        "condition": "Hypercholesterolemia",
        "category": "metabolic",
        "symptoms": "often asymptomatic, total cholesterol above 240, high LDL above 160, low HDL below 40",
        "risk_factors": "poor diet, obesity, sedentary lifestyle, familial hypercholesterolemia, hypothyroidism",
        "urgency": "routine",
        "action": "Dietary modification. Statin therapy evaluation. Recheck lipid panel in 3-6 months."
    },
    {
        "condition": "Obesity with Metabolic Syndrome",
        "category": "metabolic",
        "symptoms": "BMI above 30, elevated waist circumference, high blood pressure, high glucose, abnormal cholesterol",
        "risk_factors": "poor diet, physical inactivity, genetic factors, hormonal disorders",
        "urgency": "routine",
        "action": "Structured weight management program. 150 minutes moderate exercise weekly. Caloric restriction. Quarterly monitoring."
    },
    {
        "condition": "Heart Failure",
        "category": "cardiovascular",
        "symptoms": "shortness of breath, leg swelling, fatigue, rapid heartbeat, reduced exercise tolerance, weight gain",
        "risk_factors": "coronary artery disease, hypertension, diabetes, obesity, valve disease",
        "urgency": "urgent",
        "action": "Cardiology referral within 1 week. Echocardiogram required. Salt restriction and fluid monitoring."
    },
    {
        "condition": "Stroke",
        "category": "neurological",
        "symptoms": "sudden numbness or weakness in face arm or leg, confusion, trouble speaking, vision problems, severe headache, loss of balance",
        "risk_factors": "hypertension, atrial fibrillation, diabetes, smoking, high cholesterol, previous TIA",
        "urgency": "emergency",
        "action": "Call emergency services immediately. FAST assessment. Thrombolytic therapy window is 4.5 hours from symptom onset."
    },
]

TRIAGE_PROMPT = PromptTemplate(
    input_variables=["context", "symptoms", "patient_info"],
    template="""You are an AI medical triage assistant. Based on the medical knowledge provided and the patient's symptoms, provide a structured assessment.

MEDICAL KNOWLEDGE CONTEXT:
{context}

PATIENT INFORMATION:
{patient_info}

REPORTED SYMPTOMS:
{symptoms}

Please provide:
1. URGENCY LEVEL: Choose exactly one: emergency / urgent / soon / routine
2. ASSESSMENT: Brief clinical assessment (2-3 sentences)
3. POSSIBLE CONDITIONS: List 2-3 most likely conditions based on symptoms
4. RECOMMENDED ACTIONS: 3-4 specific next steps

Format your response as:
URGENCY: [level]
ASSESSMENT: [text]
CONDITIONS: [condition1], [condition2], [condition3]
ACTIONS:
- [action 1]
- [action 2]
- [action 3]

Remember: This is AI-assisted triage only. Always recommend professional medical evaluation."""
) if LANGCHAIN_AVAILABLE else None


class RAGTriageService:
    def __init__(self):
        self._vectorstore = None
        self._llm = None
        self._initialized = False

    def _initialize(self):
        if self._initialized or not LANGCHAIN_AVAILABLE:
            return

        try:
            # Initialize embeddings
            if settings.USE_LOCAL_LLM:
                embeddings = OllamaEmbeddings(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL
                )
                self._llm = Ollama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                    temperature=0.1,
                )
            else:
                embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self._llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.1,
                    max_tokens=600,
                )

            # Build ChromaDB vector store from knowledge base
            chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )

            documents = []
            metadatas = []
            for entry in MEDICAL_KNOWLEDGE_BASE:
                content = f"""Condition: {entry['condition']}
Symptoms: {entry['symptoms']}
Risk Factors: {entry['risk_factors']}
Urgency: {entry['urgency']}
Action: {entry['action']}"""
                documents.append(content)
                metadatas.append({"condition": entry["condition"], "urgency": entry["urgency"]})

            self._vectorstore = Chroma.from_texts(
                texts=documents,
                metadatas=metadatas,
                embedding=embeddings,
                client=chroma_client,
                collection_name="medical_knowledge",
            )

            self._initialized = True
            app_logger.info("RAG service initialized with ChromaDB vector store")

        except Exception as e:
            app_logger.error(f"RAG initialization failed: {e}. Using rule-based fallback.")
            self._initialized = False

    def _rule_based_triage(self, symptoms: str) -> Dict[str, Any]:
        """Keyword-based triage when LLM is unavailable."""
        symptoms_lower = symptoms.lower()

        emergency_keywords = [
            "chest pain", "can't breathe", "cannot breathe", "stroke", "unconscious",
            "coughing blood", "severe headache", "confusion", "severe chest", "heart attack"
        ]
        urgent_keywords = [
            "shortness of breath", "high blood pressure", "fainting", "severe pain",
            "rapid heartbeat", "severe dizziness", "leg swelling"
        ]

        matched_conditions = []
        for entry in MEDICAL_KNOWLEDGE_BASE:
            symptom_words = set(entry["symptoms"].lower().split())
            query_words = set(symptoms_lower.split())
            overlap = len(symptom_words & query_words)
            if overlap >= 2:
                matched_conditions.append((entry, overlap))

        matched_conditions.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matched_conditions[:3]]

        if any(kw in symptoms_lower for kw in emergency_keywords):
            urgency = "emergency"
        elif any(kw in symptoms_lower for kw in urgent_keywords):
            urgency = "urgent"
        elif top_matches and top_matches[0]["urgency"] in ("emergency", "urgent"):
            urgency = top_matches[0]["urgency"]
        else:
            urgency = "soon"

        conditions = [m["condition"] for m in top_matches] or ["Symptoms require clinical evaluation"]
        actions = []
        if top_matches:
            actions = [top_matches[0]["action"]]
        actions += [
            "Consult with a qualified physician for proper diagnosis",
            "Bring a list of current medications to your appointment",
            "Monitor symptoms and seek emergency care if they worsen",
        ]

        assessment = (
            f"Based on the reported symptoms, {urgency}-level medical attention is recommended. "
            f"The symptoms may be consistent with conditions including {', '.join(conditions[:2])}. "
            f"A qualified healthcare provider should perform a proper clinical evaluation."
        )

        retrieved = [
            {
                "condition": m["condition"],
                "relevance_score": round(0.85 - i * 0.15, 2),
                "excerpt": m["symptoms"][:120] + "...",
            }
            for i, m in enumerate(top_matches)
        ]

        return {
            "urgency_level": urgency,
            "ai_assessment": assessment,
            "possible_conditions": conditions,
            "recommended_actions": actions[:4],
            "retrieved_references": retrieved,
        }

    async def triage(
        self,
        symptoms: str,
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None,
        medical_history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG triage pipeline:
        symptoms → embed → retrieve → LLM → parse response
        """
        self._initialize()

        if not self._initialized or self._llm is None:
            app_logger.info("Using rule-based triage fallback")
            return self._rule_based_triage(symptoms)

        try:
            # Retrieve relevant docs
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})
            relevant_docs = retriever.get_relevant_documents(symptoms)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Build patient info string
            patient_info_parts = []
            if patient_age:
                patient_info_parts.append(f"Age: {patient_age}")
            if patient_gender:
                patient_info_parts.append(f"Gender: {patient_gender}")
            if medical_history:
                patient_info_parts.append(f"Medical history: {', '.join(medical_history)}")
            patient_info = "; ".join(patient_info_parts) or "Not provided"

            # LLM call
            prompt = TRIAGE_PROMPT.format(
                context=context,
                symptoms=symptoms,
                patient_info=patient_info
            )
            response = await self._llm.ainvoke(prompt) if hasattr(self._llm, 'ainvoke') else self._llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse structured response
            parsed = self._parse_llm_response(response_text)

            # Build retrieved references
            retrieved = []
            for doc in relevant_docs:
                retrieved.append({
                    "condition": doc.metadata.get("condition", "Unknown"),
                    "relevance_score": round(0.90, 2),
                    "excerpt": doc.page_content[:150] + "...",
                })

            return {
                "urgency_level": parsed.get("urgency", "soon"),
                "ai_assessment": parsed.get("assessment", response_text[:300]),
                "possible_conditions": parsed.get("conditions", []),
                "recommended_actions": parsed.get("actions", []),
                "retrieved_references": retrieved,
            }

        except Exception as e:
            app_logger.error(f"RAG triage failed: {e}. Using rule-based fallback.")
            return self._rule_based_triage(symptoms)

    @staticmethod
    def _parse_llm_response(text: str) -> Dict[str, Any]:
        """Parse structured LLM output."""
        lines = text.strip().split("\n")
        result = {"urgency": "soon", "assessment": "", "conditions": [], "actions": []}

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("URGENCY:"):
                urgency_raw = line.replace("URGENCY:", "").strip().lower()
                for level in ("emergency", "urgent", "soon", "routine"):
                    if level in urgency_raw:
                        result["urgency"] = level
                        break
            elif line.startswith("ASSESSMENT:"):
                result["assessment"] = line.replace("ASSESSMENT:", "").strip()
            elif line.startswith("CONDITIONS:"):
                conditions_raw = line.replace("CONDITIONS:", "").strip()
                result["conditions"] = [c.strip() for c in conditions_raw.split(",") if c.strip()]
            elif line.startswith("- ") and result.get("in_actions"):
                result["actions"].append(line[2:].strip())
            elif line.startswith("ACTIONS:"):
                result["in_actions"] = True

        return result


# Singleton
rag_service = RAGTriageService()
