"""
MLECS Prediction & Explanation Interface
==========================================
Load saved models and generate predictions with multi-level explanations
for new credit applications.

Usage:
    python research_paper/model/predict.py
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

# XAI
import shap
import lime
import lime.lime_tabular

OUTPUT_DIR = Path('research_paper/model/outputs')

print("=" * 60)
print("  MLECS - Credit Scoring Prediction & Explanation")
print("=" * 60)
print()

# Load saved models
print("Loading trained models...")
xgb_model = joblib.load(OUTPUT_DIR / 'xgboost_model.pkl')
lgbm_model = joblib.load(OUTPUT_DIR / 'lightgbm_model.pkl')
cat_model = joblib.load(OUTPUT_DIR / 'catboost_model.pkl')
meta_model = joblib.load(OUTPUT_DIR / 'meta_learner.pkl')
scaler = joblib.load(OUTPUT_DIR / 'scaler.pkl')
le_dict = joblib.load(OUTPUT_DIR / 'label_encoders.pkl')
feature_names = joblib.load(OUTPUT_DIR / 'feature_names.pkl')
print("  All models loaded successfully.")
print()

# Example: New credit application
print("=" * 60)
print("  DEMO: Scoring a New Credit Application")
print("=" * 60)
print()

# Simulated new applicant
new_applicant = {
    'checking_status': 1,      # 0<=X<200
    'duration': 24,            # 24 months
    'credit_history': 2,       # existing paid
    'purpose': 3,              # radio/tv
    'credit_amount': 5000,     # 5000 EUR
    'savings_status': 1,       # 100<=X<500
    'employment': 3,           # 4<=X<7 years
    'installment_commitment': 2,
    'personal_status': 0,      # male single
    'other_parties': 0,        # none
    'residence_since': 3,
    'property_magnitude': 1,   # life insurance
    'age': 35,
    'other_payment_plans': 2,  # none
    'housing': 1,              # own
    'existing_credits': 1,
    'job': 2,                  # skilled
    'num_dependents': 1,
    'own_telephone': 1,        # yes
    'foreign_worker': 0,       # yes
}

print("  Applicant Profile:")
print("  " + "-" * 40)
for k, v in list(new_applicant.items())[:10]:
    print(f"    {k:<30} {v}")
print(f"    ... ({len(new_applicant)} total features)")
print()

# Convert to array
X_new = np.array([[new_applicant[f] for f in feature_names]])

# Generate base predictions
pred_xgb = xgb_model.predict_proba(X_new)[:, 1]
pred_lgbm = lgbm_model.predict_proba(X_new)[:, 1]
pred_cat = cat_model.predict_proba(X_new)[:, 1]

# Meta-learner combination
stack_input = np.column_stack([pred_xgb, pred_lgbm, pred_cat])
final_pred = meta_model.predict_proba(stack_input)[0, 1]
decision = "APPROVED" if final_pred < 0.5 else "DENIED"

print("  PREDICTION RESULTS:")
print("  " + "=" * 40)
print(f"    XGBoost prediction:   {pred_xgb[0]:.4f}")
print(f"    LightGBM prediction:  {pred_lgbm[0]:.4f}")
print(f"    CatBoost prediction:  {pred_cat[0]:.4f}")
print(f"    MLECS Final Score:    {final_pred:.4f}")
print(f"    Decision:             {decision}")
print()

# SHAP Explanation
print("  LEVEL 1 & 2: SHAP Explanation")
print("  " + "-" * 40)
explainer = shap.TreeExplainer(lgbm_model)
sv = explainer.shap_values(X_new)
if isinstance(sv, list):
    sv = sv[1][0]
else:
    sv = sv[0]

sorted_idx = np.argsort(np.abs(sv))[::-1][:5]
for idx in sorted_idx:
    direction = "INCREASES" if sv[idx] > 0 else "DECREASES"
    print(f"    {feature_names[idx]:<25} {direction} risk by {abs(sv[idx]):.4f}")

print()
print("  LEVEL 3: Counterfactual Recourse")
print("  " + "-" * 40)
if decision == "DENIED":
    print("    To get APPROVED, consider:")
    print("    1. Improve checking_status (build savings account balance)")
    print("    2. Reduce loan duration (shorter term = lower risk)")
    print("    3. Increase savings_status (accumulate more savings)")
else:
    print("    Application APPROVED - no recourse needed.")

print()
print("=" * 60)
print("  Prediction complete with multi-level explanation.")
print("=" * 60)
