# Startup Success Prediction Using Machine Learning

**Team**: Qiyuan Zhu, Zella Yu  
**Course**: 7180 Final Project  
**GitHub**: https://github.com/vincentvicente/7180_final_project  
**Live Demo**: https://ycstartup-success-predictor.streamlit.app/

---

## Overview

Predict whether startups will succeed (Active/Acquired/IPO) or fail (Inactive) using machine learning on **4,974 Y Combinator companies** (2005-2024) integrated with Crunchbase funding data.

**Key Achievement**: Addressed all instructor feedback - class imbalance handling, company age feature engineering, text processing, and confusion matrix evaluation.

---

## Quick Start

### 🌐 Live Demo
**Access the deployed application**: https://ycstartup-success-predictor.streamlit.app/

### 💻 Local Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/app.py
```

---

## Data

**Sources**:
- Y Combinator: 4,974 companies (2005-2024)
- Crunchbase: 66,368 companies - matched 780 (15.7%) for funding data

**Class Distribution**:
- Success (Active/Acquired/Public): 82.9%
- Failure (Inactive): 17.1%

**Features**: `company_age`, funding metrics, industry, region, team size, TF-IDF text features

---

## Models & Performance

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| Logistic Regression | 63.9% | 0.47 | 0.61 | 0.58 | 0.68 |
| Random Forest | 67.3% | 0.57 | 0.65 | 0.62 | 0.72 |
| XGBoost | 67.3% | 0.55 | 0.66 | 0.64 | 0.74 |
| LightGBM | **68.1%** | **0.58** | **0.67** | **0.65** | **0.75** |

**Evaluation**: Confusion matrix as primary metric  
**Imbalance Handling**: SMOTE + class weighting + stratified split

---

## Application Features

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
│   ├── app.py              # Streamlit application
│   └── data_config.py      # Data loading
├── src/
│   ├── data/               # Preprocessing
│   ├── features/           # Feature engineering & text processing
│   ├── models/             # Training & evaluation
│   └── visualization/      # Plotting functions
├── data/
│   ├── raw/                # Raw datasets
│   └── processed/          # Preprocessed data
└── requirements.txt        # Dependencies
```

---

## Addressing Instructor Feedback

✅ **Pre-curated Metrics**: Interactive dashboards with success rates by industry/region  
✅ **Class Imbalance**: SMOTE + class weighting + confusion matrix as primary metric  
✅ **Feature Engineering**: `company_age` feature (became top-3 predictor)  
✅ **Text Processing**: TF-IDF vectorization + keyword extraction

---

## Technical Highlights

**Data Integration**:
- Multi-strategy matching: name normalization → domain matching → fuzzy matching
- 780 companies (15.7%) with verified Crunchbase funding data
- Industry-median imputation for remaining companies

**Feature Engineering**:
- `company_age` = 2024 - year_founded
- Funding ratios, temporal features, location indicators
- TF-IDF (100 features) from tags & descriptions

**Model Training**:
- SMOTE for synthetic minority oversampling
- Class weighting in all models
- Stratified train-test split
- Hyperparameter tuning with GridSearchCV

---

## Requirements

- Python 3.10+
- Key packages: streamlit, pandas, numpy, scikit-learn, xgboost, lightgbm

See `requirements.txt` for full list.

---

## Contact

**Team**: Qiyuan Zhu, Zella Yu  
**GitHub**: https://github.com/vincentvicente/7180_final_project  
**Course**: 7180 Final Project
