"""
PyTorch Neural Network Baseline + MLflow Experiment Tracking
================================================================

Two things this script adds on top of the existing RF/XGBoost/Stacking
pipeline in 01_eda_and_training.py:

1. A PyTorch MLP baseline, trained on the SAME synthetic data/split, so it's
   a genuine apples-to-apples comparison against the tree-based ensemble --
   not a separate, incomparable exercise.
2. MLflow experiment tracking for ALL four models (RF, XGBoost, Stacked
   Ensemble, PyTorch MLP), logging params, metrics, and model artifacts, so
   "the ensemble improved AUC from X to Y" becomes a checkable claim backed
   by a real run history instead of a one-off printed number.

Deliberately lives in ml_pipeline/, not backend/ -- installing
tensorflow-cpu here forced numpy to 2.5.2, which broke backend's LangChain/
ChromaDB pin (numpy<2.0). Keeping torch/mlflow isolated to this training
environment avoids that conflict entirely, and is also why this uses
PyTorch only, not both PyTorch and TensorFlow: two deep-learning frameworks
in one project doesn't add signal over one done well, and TensorFlow's
numpy requirement makes it actively hostile to this repo's existing RAG
dependencies.

Run:
    cd ml_pipeline
    python notebooks/02_pytorch_and_mlflow.py

    # Then view the tracked runs:
    mlflow ui --backend-store-uri ./mlruns
    # open http://localhost:5000
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

MLFLOW_EXPERIMENT = "meditrack-risk-prediction"
RANDOM_STATE = 42


def generate_synthetic_data(n=3000, seed=RANDOM_STATE):
    """Same generator as 01_eda_and_training.py, factored out so this script
    can run standalone without re-running the notebook. Ground truth is a
    synthetic clinical-rule label, not real patient outcomes -- worth saying
    plainly, since it caps how much "predictive power" claims here should be
    read as: this measures whether models can recover a known rule, not
    whether they'd generalize to real-world clinical risk."""
    np.random.seed(seed)
    age = np.random.randint(18, 85, n)
    sbp = np.random.randint(90, 200, n)
    dbp = np.random.randint(60, 120, n)
    hr = np.random.randint(50, 120, n)
    glucose = np.random.uniform(70, 280, n)
    bmi = np.random.uniform(16, 45, n)
    chol_total = np.random.uniform(130, 300, n)
    chol_hdl = np.random.uniform(25, 90, n)
    chol_ldl = np.random.uniform(50, 220, n)
    hgb = np.random.uniform(8, 18, n)
    o2sat = np.random.uniform(88, 100, n)

    pulse_pressure = sbp - dbp
    chol_ratio = chol_total / np.maximum(chol_hdl, 1)
    gluc_bmi_idx = (glucose * bmi) / 100.0

    risk = (
        (sbp >= 140).astype(float) * 0.30
        + (glucose >= 126).astype(float) * 0.25
        + (bmi >= 30).astype(float) * 0.15
        + (o2sat <= 94).astype(float) * 0.20
        + (age >= 60).astype(float) * 0.10
        + (chol_total >= 240).astype(float) * 0.10
        + np.random.uniform(0, 0.15, n)
    )
    y = (risk >= 0.4).astype(int)

    df = pd.DataFrame({
        "age": age, "bp_systolic": sbp, "bp_diastolic": dbp, "heart_rate": hr,
        "blood_glucose": glucose, "bmi": bmi, "chol_total": chol_total,
        "chol_hdl": chol_hdl, "chol_ldl": chol_ldl, "hemoglobin": hgb,
        "o2_saturation": o2sat, "pulse_pressure": pulse_pressure,
        "chol_ratio": chol_ratio, "glucose_bmi_idx": gluc_bmi_idx,
        "risk_label": y,
    })
    return df


class RiskMLP(nn.Module):
    """Small 3-layer MLP -- deliberately simple. A dataset this size (3000
    rows, 14 features) doesn't justify a deep network; the point is showing
    you can implement and train a PyTorch model correctly and evaluate it
    honestly against simpler baselines, not that a neural net is the right
    tool for this specific dataset size."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_pytorch_mlp(X_train, y_train, X_test, y_test, epochs=60, lr=1e-3):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = RiskMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_proba = torch.sigmoid(model(X_test_t)).numpy().flatten()

    return model, test_proba, losses


def evaluate(y_test, proba, threshold=0.5) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = generate_synthetic_data()
    features = ["age", "bp_systolic", "bp_diastolic", "heart_rate", "blood_glucose", "bmi",
                "chol_total", "chol_hdl", "chol_ldl", "hemoglobin", "o2_saturation",
                "pulse_pressure", "chol_ratio", "glucose_bmi_idx"]
    X = df[features].values
    y = df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = sm.fit_resample(X_train_s, y_train)

    results = {}

    # ---------- Random Forest ----------
    with mlflow.start_run(run_name="random_forest"):
        params = {"n_estimators": 200, "max_depth": 12, "class_weight": "balanced"}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        rf = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_bal, y_bal)
        metrics = evaluate(y_test, rf.predict_proba(X_test_s)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "model")
        results["Random Forest"] = metrics
        print(f"Random Forest: {metrics}")

    # ---------- XGBoost ----------
    with mlflow.start_run(run_name="xgboost"):
        params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBClassifier")
        xgb_model = xgb.XGBClassifier(**params, eval_metric="logloss",
                                       random_state=RANDOM_STATE, n_jobs=-1)
        xgb_model.fit(X_bal, y_bal, eval_set=[(X_test_s, y_test)], verbose=False)
        metrics = evaluate(y_test, xgb_model.predict_proba(X_test_s)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgb_model, "model")
        results["XGBoost"] = metrics
        print(f"XGBoost: {metrics}")

    # ---------- Stacked Ensemble ----------
    with mlflow.start_run(run_name="stacked_ensemble"):
        mlflow.log_param("model_type", "RF+XGB -> LogisticRegression meta-learner")
        meta_X_train = np.column_stack([rf.predict_proba(X_bal)[:, 1], xgb_model.predict_proba(X_bal)[:, 1]])
        meta_X_test = np.column_stack([rf.predict_proba(X_test_s)[:, 1], xgb_model.predict_proba(X_test_s)[:, 1]])
        meta = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
        meta.fit(meta_X_train, y_bal)
        metrics = evaluate(y_test, meta.predict_proba(meta_X_test)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(meta, "model")
        results["Stacked Ensemble"] = metrics
        print(f"Stacked Ensemble: {metrics}")

    # ---------- PyTorch MLP ----------
    with mlflow.start_run(run_name="pytorch_mlp"):
        params = {"hidden_layers": "32-16-1", "epochs": 60, "lr": 1e-3, "dropout": 0.2}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "PyTorch MLP (BCEWithLogitsLoss)")
        model, test_proba, losses = train_pytorch_mlp(X_bal, y_bal, X_test_s, y_test)
        metrics = evaluate(y_test, test_proba)
        mlflow.log_metrics(metrics)
        for i, loss_val in enumerate(losses):
            mlflow.log_metric("train_loss", loss_val, step=i)
        example_input = X_test_s[:1].astype(np.float32)
        mlflow.pytorch.log_model(model, "model", input_example=example_input)
        results["PyTorch MLP"] = metrics
        print(f"PyTorch MLP: {metrics}")

    # ---------- Summary ----------
    summary = pd.DataFrame(results).T.round(4)
    print("\n=== Model comparison (test set) ===")
    print(summary.to_string())
    summary.to_csv("../data/model_comparison_with_pytorch.csv")
    print("\nSaved comparison to ml_pipeline/data/model_comparison_with_pytorch.csv")
    print(f"View tracked runs: mlflow ui --backend-store-uri ./mlruns")

    return results


if __name__ == "__main__":
    main()
