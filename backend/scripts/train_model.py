"""
Train the stacked ensemble model:
  Base learners : Random Forest + XGBoost
  Meta-learner  : Logistic Regression
  Explainability: SHAP (TreeExplainer on XGBoost)

Run: python -m scripts.train_model
Outputs saved to: ml_pipeline/models/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent.parent / "ml_pipeline" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "age", "blood_pressure_systolic", "blood_pressure_diastolic",
    "heart_rate", "blood_glucose", "bmi", "cholesterol_total",
    "cholesterol_hdl", "cholesterol_ldl", "hemoglobin",
    "oxygen_saturation", "pulse_pressure", "cholesterol_ratio",
    "glucose_bmi_index",
]


def generate_synthetic_data(n=2000, seed=42) -> pd.DataFrame:
    """Generate clinically plausible synthetic health data with realistic risk labels."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 85, n)
    sbp = rng.integers(90, 200, n)
    dbp = rng.integers(60, 120, n)
    hr = rng.integers(50, 120, n)
    glucose = rng.uniform(70, 280, n)
    bmi = rng.uniform(16, 45, n)
    chol_total = rng.uniform(130, 300, n)
    chol_hdl = rng.uniform(25, 90, n)
    chol_ldl = rng.uniform(50, 220, n)
    hgb = rng.uniform(8, 18, n)
    o2_sat = rng.uniform(88, 100, n)

    pulse_pressure = sbp - dbp
    chol_ratio = chol_total / np.maximum(chol_hdl, 1)
    gluc_bmi_idx = (glucose * bmi) / 100.0

    # Clinical risk scoring for labels. Real-world clinical outcomes have
    # substantial unexplained variance even given the same measured vitals —
    # unmeasured confounders, individual variation, diagnostic uncertainty.
    # A purely deterministic threshold formula on the same features the
    # model sees makes the task artificially easy (near-perfect AUC is a
    # red flag, not a good sign, for a task like this). Two changes below
    # aim to make the synthetic task's difficulty resemble a real one:
    #   1. Symmetric Gaussian noise (not one-directional) in the risk score
    #   2. Explicit random label flips, simulating diagnostic/labeling
    #      uncertainty independent of the underlying formula
    risk = (
        (sbp >= 140).astype(float) * 0.3 +
        (glucose >= 126).astype(float) * 0.25 +
        (bmi >= 30).astype(float) * 0.15 +
        (o2_sat <= 94).astype(float) * 0.2 +
        (age >= 60).astype(float) * 0.1 +
        (chol_total >= 240).astype(float) * 0.1 +
        rng.normal(0, 0.18, n)  # symmetric noise, larger than before
    )
    y = (risk >= 0.55).astype(int)

    # Simulate real-world label uncertainty: flip ~8% of labels at random,
    # independent of the risk formula (measurement error, borderline calls,
    # diagnostic disagreement — all real sources of label noise a model
    # can't be expected to perfectly resolve).
    flip_mask = rng.random(n) < 0.08
    y = np.where(flip_mask, 1 - y, y)

    df = pd.DataFrame({
        "age": age, "blood_pressure_systolic": sbp, "blood_pressure_diastolic": dbp,
        "heart_rate": hr, "blood_glucose": glucose, "bmi": bmi,
        "cholesterol_total": chol_total, "cholesterol_hdl": chol_hdl,
        "cholesterol_ldl": chol_ldl, "hemoglobin": hgb,
        "oxygen_saturation": o2_sat, "pulse_pressure": pulse_pressure,
        "cholesterol_ratio": chol_ratio, "glucose_bmi_index": gluc_bmi_idx,
        "risk_label": y,
    })
    return df


def train():
    print("=" * 60)
    print("Health AI — Model Training Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Generating synthetic training data...")
    df = generate_synthetic_data(n=3000)

    # Save for reference
    df.to_csv(OUTPUT_DIR.parent / "data" / "synthetic_health_data.csv", index=False)
    print(f"    Dataset: {len(df)} records | Positive rate: {df['risk_label'].mean():.1%}")

    X = df[FEATURES].values
    y = df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Preprocessing + SMOTE
    print("\n[2/5] Preprocessing + SMOTE balancing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    print(f"    After SMOTE: {len(X_train_bal)} samples")

    # 3. Train base models
    print("\n[3/5] Training base learners...")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    from sklearn.metrics import f1_score

    rf_test_pred = rf.predict(X_test_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])
    rf_acc = accuracy_score(y_test, rf_test_pred)
    xgb_acc = accuracy_score(y_test, xgb_test_pred)
    rf_f1 = f1_score(y_test, rf_test_pred)
    xgb_f1 = f1_score(y_test, xgb_test_pred)
    print(f"    RF  — AUC: {rf_auc:.4f}  ACC: {rf_acc:.4f}  F1: {rf_f1:.4f}")
    print(f"    XGB — AUC: {xgb_auc:.4f}  ACC: {xgb_acc:.4f}  F1: {xgb_f1:.4f}")

    # 4. Stacking meta-learner
    print("\n[4/5] Training stacked meta-learner...")
    rf_train_proba = rf.predict_proba(X_train_bal)[:, 1]
    xgb_train_proba = xgb_model.predict_proba(X_train_bal)[:, 1]
    meta_train = np.column_stack([rf_train_proba, xgb_train_proba])

    rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_test_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    meta_test = np.column_stack([rf_test_proba, xgb_test_proba])

    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(meta_train, y_train_bal)

    ensemble_preds = meta_model.predict_proba(meta_test)[:, 1]
    ensemble_pred_labels = meta_model.predict(meta_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_preds)
    ensemble_acc = accuracy_score(y_test, ensemble_pred_labels)
    ensemble_f1 = f1_score(y_test, ensemble_pred_labels)

    print(f"    Ensemble AUC: {ensemble_auc:.4f}")
    print(f"    Ensemble ACC: {ensemble_acc:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_test, meta_model.predict(meta_test), target_names=["Low Risk", "High Risk"]))

    # 5. SHAP
    print("\n[5/5] Computing SHAP values + saving artifacts...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test_scaled[:100])

    # Save artifacts
    joblib.dump(rf, OUTPUT_DIR / "rf_model.pkl")
    joblib.dump(xgb_model, OUTPUT_DIR / "xgb_model.pkl")
    joblib.dump(meta_model, OUTPUT_DIR / "meta_model.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    # Save metrics
    metrics = {
        "random_forest": {"auc": rf_auc, "accuracy": rf_acc, "f1": rf_f1},
        "xgboost": {"auc": xgb_auc, "accuracy": xgb_acc, "f1": xgb_f1},
        "stacked_ensemble": {"auc": ensemble_auc, "accuracy": ensemble_acc, "f1": ensemble_f1},
        "model_version": "v1.2.0",
        "features": FEATURES,
        "training_samples": len(X_train_bal),
        "test_samples": len(X_test),
        "note": (
            "Synthetic data with symmetric noise + 8% random label flips "
            "(see generate_synthetic_data) to approximate real-world label "
            "uncertainty rather than a cleanly-separable formula on the "
            "same features the model is given."
        ),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n    Models saved to: {OUTPUT_DIR}")
    print(f"    RF AUC: {rf_auc:.4f} | XGB AUC: {xgb_auc:.4f} | Ensemble AUC: {ensemble_auc:.4f}")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
