"""
Seed the database with demo data.
Run: docker exec -it health_backend python scripts/seed_data.py
"""
import asyncio
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings
from app.core.security import hash_password
from app.models.models import User, Patient, HealthRecord, MedicalKnowledge

engine = create_async_engine(settings.MYSQL_URL, echo=False)
Session = async_sessionmaker(engine, expire_on_commit=False)

MEDICAL_KB = [
    ("cardiovascular", "Hypertension", "high blood pressure headache dizziness", "obesity sedentary lifestyle high salt diet", "Persistently elevated blood pressure above 130/80 mmHg", "urgent"),
    ("endocrine", "Type 2 Diabetes", "thirst frequent urination fatigue blurred vision", "obesity family history sedentary lifestyle", "Impaired insulin secretion or resistance", "soon"),
    ("cardiovascular", "Coronary Artery Disease", "chest pain shortness of breath fatigue", "hypertension diabetes smoking high cholesterol", "Plaque buildup in coronary arteries", "urgent"),
    ("metabolic", "Metabolic Syndrome", "abdominal obesity high BP elevated glucose", "obesity inactivity poor diet", "Cluster of conditions increasing heart disease risk", "soon"),
    ("hematology", "Iron Deficiency Anemia", "fatigue weakness pale skin low hemoglobin", "poor diet pregnancy blood loss", "Insufficient iron for red blood cell production", "soon"),
    ("pulmonary", "COPD", "chronic cough shortness of breath wheezing", "smoking air pollution occupational dust", "Obstructive airflow limitation in lungs", "urgent"),
    ("cardiovascular", "Heart Failure", "shortness of breath leg swelling fatigue", "coronary disease hypertension diabetes", "Heart unable to pump sufficient blood", "urgent"),
    ("endocrine", "Hypothyroidism", "fatigue weight gain cold intolerance constipation", "autoimmune disease iodine deficiency", "Insufficient thyroid hormone production", "soon"),
]

async def seed():
    async with Session() as db:
        # Admin user
        admin = User(email="admin@healthai.com", full_name="Dr. Admin", hashed_password=hash_password("Admin@123"), role="admin")
        db.add(admin)

        # Doctor
        doctor = User(email="doctor@healthai.com", full_name="Dr. Priya Sharma", hashed_password=hash_password("Doctor@123"), role="doctor")
        db.add(doctor)
        await db.flush()

        # Patients
        names = ["Rahul Gupta", "Sita Devi", "Amit Kumar", "Priya Singh", "Rajesh Patel",
                 "Meena Bose", "Suresh Roy", "Anita Das", "Vikram Joshi", "Kavita Nair"]

        patients = []
        for i, name in enumerate(names):
            p = Patient(
                patient_code=f"PT-{1000+i}",
                age=random.randint(25, 75),
                gender=random.choice(["male", "female"]),
                blood_group=random.choice(["A+", "B+", "O+", "AB+", "A-", "B-"]),
                contact_number=f"9{random.randint(100000000, 999999999)}",
                medical_history={"conditions": random.sample(["hypertension", "diabetes", "none"], k=1)},
            )
            patients.append(p)
            db.add(p)

        await db.flush()

        # Health records
        for p in patients:
            for _ in range(3):
                r = HealthRecord(
                    patient_id=p.id,
                    recorded_by=doctor.id,
                    blood_pressure_systolic=random.randint(110, 175),
                    blood_pressure_diastolic=random.randint(70, 110),
                    heart_rate=random.randint(60, 100),
                    blood_glucose=round(random.uniform(75, 220), 1),
                    bmi=round(random.uniform(18, 38), 1),
                    cholesterol_total=round(random.uniform(150, 260), 1),
                    cholesterol_hdl=round(random.uniform(35, 80), 1),
                    cholesterol_ldl=round(random.uniform(70, 190), 1),
                    hemoglobin=round(random.uniform(9, 17), 1),
                    temperature=round(random.uniform(36.1, 37.8), 1),
                    oxygen_saturation=round(random.uniform(93, 99), 1),
                    notes="Routine checkup",
                )
                db.add(r)

        # Medical knowledge base
        for cat, cond, symp, risk, desc, urg in MEDICAL_KB:
            kb = MedicalKnowledge(
                category=cat, condition_name=cond, symptoms=symp,
                risk_factors=risk, description=desc, urgency_guidelines=urg
            )
            db.add(kb)

        await db.commit()

    print("Database seeded successfully!")
    print("Admin: admin@healthai.com / Admin@123")
    print("Doctor: doctor@healthai.com / Doctor@123")


if __name__ == "__main__":
    asyncio.run(seed())
