# MLECS: Multi-Level Explainable Credit Scoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: UCI German Credit](https://img.shields.io/badge/Dataset-UCI%20German%20Credit-orange.svg)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

> **Official implementation** for the paper: *"MLECS: A Multi-Level Explainable Framework for Credit Scoring Using Hybrid Ensemble Learning with SHAP, LIME, and Counterfactual Explanations"*

---

## Overview

MLECS is a framework that combines a **stacking ensemble** of gradient boosting models (XGBoost, LightGBM, CatBoost) with a **three-tier explainability architecture**:

| Level | Method | Purpose |
|-------|--------|---------|
| Level 1: Global | SHAP | Population-level feature importance |
| Level 2: Local | SHAP + LIME | Instance-level decision explanation with dual validation |
| Level 3: Counterfactual | DiCE | Actionable recourse for denied applicants |

The framework also introduces the **Explanation Consistency Score (ECS)** — a novel metric quantifying agreement between SHAP and LIME explanations.

---

## Results

Trained on the **German Credit Dataset (UCI)** | seed=42 | 5-fold stratified CV:

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.8052** |
| AUC-PR | 0.6692 |
| KS Statistic | 0.4667 |
| F1-Score | 0.5962 |
| ECS (Overall) | **0.6927** |
| Faithfulness Ratio | **4.09x** vs random |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the full model (downloads data automatically)
python train_mlecs.py

# Run inference demo
python predict.py
```

---

## Repository Structure

```
MLECS/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── train_mlecs.py              # Full training pipeline
├── predict.py                  # Inference demo
├── outputs/
│   ├── xgboost_model.pkl       # Trained XGBoost
│   ├── lightgbm_model.pkl      # Trained LightGBM
│   ├── catboost_model.pkl      # Trained CatBoost
│   ├── meta_learner.pkl        # Trained meta-learner
│   ├── scaler.pkl              # Fitted scaler
│   ├── label_encoders.pkl      # Fitted encoders
│   ├── feature_names.pkl       # Feature list
│   ├── feature_importance.csv  # SHAP rankings
│   └── results.json            # All metrics
├── figures/
│   ├── shap_summary_plot.png   # SHAP beeswarm
│   ├── shap_bar_plot.png       # SHAP bar chart
│   ├── shap_waterfall_local.png# SHAP waterfall
│   ├── lime_local_explanation.png # LIME plot
│   ├── roc_curve_comparison.png# ROC curves
│   ├── model_comparison_bar.png# AUC comparison
│   └── ecs_scores.png          # ECS visualization
├── diagrams/
│   ├── mlecs_architecture.puml # System architecture (PlantUML)
│   ├── mlecs_activity_diagram.puml # Activity diagram
│   └── mlecs_sequence_diagram.puml # Sequence diagram
└── paper/
    ├── MLECS_Full_Paper.md     # Full paper (Markdown)
    ├── MLECS_FINAL_CLEAN.docx  # Final paper (Word)
    ├── paper_ieee_latex.tex    # LaTeX version
    └── references.bib          # Bibliography
```

---

## Key Equations

**Stacking Ensemble:**
```
P(default|x) = σ(w₁·f_XGB(x) + w₂·f_LGBM(x) + w₃·f_CAT(x) + b)
```

**Explanation Consistency Score:**
```
ECS = α·RankCorr(SHAP,LIME) + β·SignAgree(SHAP,LIME) + γ·TopK_Overlap(SHAP,LIME,k)
```

**Faithfulness:**
```
Faithfulness(k) = (1/N)·Σᵢ |f(xᵢ) - f(xᵢ \ TopK_features)|
```

---

## Datasets Supported

| Dataset | Instances | Features | Default Rate | Source |
|---------|-----------|----------|-------------|--------|
| German Credit | 1,000 | 20 | 30.0% | [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) |
| LendingClub | 2,260,668 | 151 | 14.5% | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Home Credit | 307,511 | 122 | 8.1% | [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) |

---

## Citation

```bibtex
@article{mlecs2026,
  title={MLECS: A Multi-Level Explainable Framework for Credit Scoring Using Hybrid Ensemble Learning with SHAP, LIME, and Counterfactual Explanations},
  author={Kaushik, Hardik and others},
  journal={Expert Systems with Applications},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
