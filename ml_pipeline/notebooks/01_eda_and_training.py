# %% [markdown]
# # Health AI Platform — EDA & Model Training Notebook
# 
# This notebook walks through:
# 1. Synthetic data generation
# 2. Exploratory Data Analysis (EDA)
# 3. Feature engineering
# 4. Model training (RF + XGBoost + Stacking)
# 5. SHAP explainability analysis
# 6. Model evaluation

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import xgboost as xgb

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# %% [markdown]
# ## 1. Generate Synthetic Health Dataset

# %%
np.random.seed(42)
N = 3000

age = np.random.randint(18, 85, N)
sbp = np.random.randint(90, 200, N)
dbp = np.random.randint(60, 120, N)
hr  = np.random.randint(50, 120, N)
glucose    = np.random.uniform(70, 280, N)
bmi        = np.random.uniform(16, 45, N)
chol_total = np.random.uniform(130, 300, N)
chol_hdl   = np.random.uniform(25, 90, N)
chol_ldl   = np.random.uniform(50, 220, N)
hgb   = np.random.uniform(8, 18, N)
o2sat = np.random.uniform(88, 100, N)

# Derived features
pulse_pressure = sbp - dbp
chol_ratio     = chol_total / np.maximum(chol_hdl, 1)
gluc_bmi_idx   = (glucose * bmi) / 100.0

# Clinical risk labels
risk = (
    (sbp >= 140).astype(float) * 0.30 +
    (glucose >= 126).astype(float) * 0.25 +
    (bmi >= 30).astype(float) * 0.15 +
    (o2sat <= 94).astype(float) * 0.20 +
    (age >= 60).astype(float) * 0.10 +
    (chol_total >= 240).astype(float) * 0.10 +
    np.random.uniform(0, 0.15, N)
)
y = (risk >= 0.4).astype(int)

df = pd.DataFrame({
    'age': age, 'bp_systolic': sbp, 'bp_diastolic': dbp,
    'heart_rate': hr, 'blood_glucose': glucose, 'bmi': bmi,
    'chol_total': chol_total, 'chol_hdl': chol_hdl, 'chol_ldl': chol_ldl,
    'hemoglobin': hgb, 'o2_saturation': o2sat,
    'pulse_pressure': pulse_pressure,
    'chol_ratio': chol_ratio,
    'glucose_bmi_idx': gluc_bmi_idx,
    'risk_label': y,
})

print(f"Dataset shape: {df.shape}")
print(f"Risk positive rate: {y.mean():.1%}")
df.describe()

# %% [markdown]
# ## 2. EDA

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
features = ['age', 'bp_systolic', 'blood_glucose', 'bmi', 'chol_total', 'chol_hdl',
            'chol_ldl', 'hemoglobin', 'o2_saturation', 'pulse_pressure', 'chol_ratio', 'glucose_bmi_idx']

for ax, feat in zip(axes.flatten(), features):
    df.groupby('risk_label')[feat].plot.kde(ax=ax, legend=(feat == 'age'))
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel('')

plt.suptitle('Feature Distributions by Risk Label', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('../data/eda_distributions.png', dpi=120, bbox_inches='tight')
plt.show()

# %%
# Correlation heatmap
plt.figure(figsize=(12, 9))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('../data/correlation_matrix.png', dpi=120, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Model Training

# %%
FEATURES = ['age', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'blood_glucose', 'bmi',
            'chol_total', 'chol_hdl', 'chol_ldl', 'hemoglobin', 'o2_saturation',
            'pulse_pressure', 'chol_ratio', 'glucose_bmi_idx']

X = df[FEATURES].values
y = df['risk_label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_train_s, y_train)
print(f"After SMOTE: {X_bal.shape[0]} samples | Positive rate: {y_bal.mean():.1%}")

# %%
rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_bal, y_bal)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])
print(f"Random Forest AUC: {rf_auc:.4f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
)
xgb_model.fit(X_bal, y_bal, eval_set=[(X_test_s, y_test)], verbose=False)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test_s)[:, 1])
print(f"XGBoost AUC: {xgb_auc:.4f}")

# Stacking
meta_X_train = np.column_stack([rf.predict_proba(X_bal)[:, 1], xgb_model.predict_proba(X_bal)[:, 1]])
meta_X_test  = np.column_stack([rf.predict_proba(X_test_s)[:, 1], xgb_model.predict_proba(X_test_s)[:, 1]])

meta = LogisticRegression(C=1.0, random_state=42)
meta.fit(meta_X_train, y_bal)
ens_auc = roc_auc_score(y_test, meta.predict_proba(meta_X_test)[:, 1])
print(f"Ensemble AUC: {ens_auc:.4f}")
print(classification_report(y_test, meta.predict(meta_X_test), target_names=['Low Risk', 'High Risk']))

# %%
# ROC curves comparison
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, rf.predict_proba(X_test_s)[:, 1], ax=ax, name=f'Random Forest (AUC={rf_auc:.3f})')
RocCurveDisplay.from_predictions(y_test, xgb_model.predict_proba(X_test_s)[:, 1], ax=ax, name=f'XGBoost (AUC={xgb_auc:.3f})')
RocCurveDisplay.from_predictions(y_test, meta.predict_proba(meta_X_test)[:, 1], ax=ax, name=f'Ensemble (AUC={ens_auc:.3f})', color='green')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_title('ROC Curve Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('../data/roc_curves.png', dpi=120, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. SHAP Explainability

# %%
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_s[:200])

plt.figure()
shap.summary_plot(shap_values, pd.DataFrame(X_test_s[:200], columns=FEATURES),
                  plot_type='bar', show=False, max_display=14)
plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=12)
plt.tight_layout()
plt.savefig('../data/shap_bar.png', dpi=120, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values, pd.DataFrame(X_test_s[:200], columns=FEATURES),
                  show=False, max_display=12)
plt.title('SHAP Beeswarm Plot', fontsize=12)
plt.tight_layout()
plt.savefig('../data/shap_beeswarm.png', dpi=120, bbox_inches='tight')
plt.show()

# %%
# Single prediction explanation (SHAP waterfall)
idx = 5
shap.plots.waterfall(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test_s[idx],
    feature_names=FEATURES,
))

print(f"\nSample prediction: {meta.predict_proba(meta_X_test[idx:idx+1])[0, 1]:.3f} risk score")
print(f"True label: {'High Risk' if y_test[idx] == 1 else 'Low Risk'}")

# %% [markdown]
# ## 5. Save Artefacts

# %%
import joblib, json
from pathlib import Path

OUT = Path('../models')
OUT.mkdir(exist_ok=True)

joblib.dump(rf,        OUT / 'rf_model.pkl')
joblib.dump(xgb_model, OUT / 'xgb_model.pkl')
joblib.dump(meta,      OUT / 'meta_model.pkl')
joblib.dump(scaler,    OUT / 'scaler.pkl')

with open(OUT / 'metrics.json', 'w') as f:
    json.dump({
        'rf_auc': rf_auc, 'xgb_auc': xgb_auc, 'ensemble_auc': ens_auc,
        'features': FEATURES, 'model_version': 'v1.2.0',
    }, f, indent=2)

print("All artefacts saved to ml_pipeline/models/")
