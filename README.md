# Startup Success Prediction Using Machine Learning

**Team**: Qiyuan Zhu, Zella Yu  
**Course**: 7180 Final Project  
**GitHub**: https://github.com/vincentvicente/7180_final_project

---

## Overview

Predict whether startups will succeed (Active/Acquired/IPO) or fail (Inactive) using machine learning on **4,974 Y Combinator companies** (2005-2024) integrated with Crunchbase funding data.

**Key Achievement**: Addressed all instructor feedback - class imbalance handling, company age feature engineering, text processing, and confusion matrix evaluation.

---

## Quick Start

🚀 **Easiest**: Double-click `run_app.bat` → Opens at http://localhost:8501

**Manual**:
```bash
.\venv\Scripts\activate
streamlit run app/app.py
```

---

## Data

**Sources**:
- **Y Combinator**: 4,974 companies (2005-2024) - 100% real data
- **Crunchbase**: 66,368 companies - matched 780 (15.7%) for real funding data
- **Remaining 84.3%**: Industry-specific median imputation

**Class Distribution**:
- Success (Active/Acquired/Public): 82.9%
- Failure (Inactive): 17.1%

**Features** (19 total):
- **Engineered**: `company_age` (from year_founded), funding ratios
- **Categorical**: industry (60), region (401), team_size
- **Text**: TF-IDF vectors from tags & descriptions

---

## Models & Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 63.9% | 0.47 | 0.61 | 0.58 |
| Random Forest | 67.3% | 0.57 | 0.65 | 0.62 |
| XGBoost | 67.3% | 0.55 | 0.66 | 0.64 |
| LightGBM | 68.1% | 0.58 | 0.67 | 0.65 |

**Evaluation**: Confusion matrix as primary metric (instructor requirement)  
**Imbalance Handling**: SMOTE + class weighting

---

## Application (5 Pages)

1. **Home** - Dataset statistics & class distribution
2. **Data Explorer** - Filter by industry/region, interactive EDA
3. **Model Performance** - Confusion matrices, metric comparison, feature importance
4. **Interactive Prediction** - Real-time success probability calculator
5. **Regional Analysis** - Geographic success patterns

---

## Project Structure

```
7180_final_project/
├── app/
│   ├── app.py              # Streamlit application (5 pages)
│   └── data_config.py      # Data loading & integration
├── src/
│   ├── data/               # Preprocessing & cleaning
│   ├── features/           # Feature engineering & text processing
│   ├── models/             # Training & evaluation (SMOTE, confusion matrix)
│   └── visualization/      # Plotting functions
├── data/raw/               # Datasets (not in git)
├── tests/sample_test.py    # Quick test
├── run_app.bat             # Launcher
└── requirements.txt        # Dependencies
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `run_app.bat` | One-click launcher (double-click to start) |
| `app/app.py` | Streamlit UI - all 5 pages |
| `app/data_config.py` | Loads & merges YC + Crunchbase data |
| `src/features/feature_engineering.py` | Creates `company_age` & other features |
| `src/features/text_processing.py` | TF-IDF, keyword extraction |
| `src/models/train_model.py` | SMOTE, model training |
| `src/models/evaluate_model.py` | Confusion matrix, metrics |
| `src/visualization/plots.py` | All charts & graphs |
| `tests/sample_test.py` | Pipeline validation test |

---

## Addressing Instructor Feedback ✅

**Feedback #1**: Pre-curated metrics for users  
→ ✅ Interactive dashboards with success rates by industry/region

**Feedback #2**: Class imbalance (82.9% success)  
→ ✅ SMOTE + class weighting + **confusion matrix** as primary metric

**Feedback #3**: "Year founded" not useful alone  
→ ✅ Engineered `company_age` feature (became top-3 most important)

**Feedback #4**: How are text features used?  
→ ✅ TF-IDF vectorization + keyword extraction from tags/descriptions

---

## Presentation Guide (10 Minutes)

### Slide Structure:
1. **Problem** (1.5min) - Why predict startup success? Importance & difficulty
2. **Data** (1.5min) - 4,974 YC companies, 82.9% class imbalance
3. **Findings** (2min) - Success rates by industry/region, company age impact
4. **Methods** (1.5min) - SMOTE, confusion matrix, company_age feature, TF-IDF
5. **Performance** (1.5min) - Model comparison, confusion matrices
6. **Demo** (2min) - Live Streamlit app walkthrough

### Key Talking Points:
- ✅ Confusion matrix shows model handles minority class (not just guessing "success")
- ✅ Company age (engineered from year_founded) is top-3 predictor
- ✅ 15.7% real Crunchbase funding + 84.3% industry-median imputation
- ✅ Multi-strategy matching (name normalization + domain) for data integration

---

## Technical Highlights

**Data Integration**:
- Multi-strategy matching: name normalization → domain matching → fuzzy matching (0.85 threshold)
- Result: 780 companies (15.7%) with verified funding data
- Quality-first approach: <5% false positive rate

**Feature Engineering**:
- `company_age` = 2024 - year_founded (addresses instructor feedback)
- Funding ratios, temporal features, location indicators
- Text: TF-IDF (100 features), keyword flags

**Handling Imbalance**:
- SMOTE: Synthetic minority over-sampling
- Class weighting in all models
- Stratified train-test split

---

## Installation (First Time Setup)

```bash
# Clone repository
git clone https://github.com/vincentvicente/7180_final_project.git
cd 7180_final_project

# Create virtual environment (already done if venv/ exists)
python -m venv venv

# Install dependencies
.\venv\Scripts\activate
pip install -r requirements.txt

# Run app
streamlit run app/app.py
```

---

## Contact

**Team**: Qiyuan Zhu, Zella Yu  
**GitHub**: https://github.com/vincentvicente/7180_final_project  
**Course**: 7180 Final Project  
**Instructor Score**: 85/100
