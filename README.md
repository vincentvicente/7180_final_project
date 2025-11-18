# Startup Success Prediction Using Machine Learning

## Project Overview

This project uses machine learning methods to predict whether a startup will remain active (including acquired and IPO) or eventually close. We use company data such as founding year, total funding, location, number of funding rounds, and investment timing to estimate the likelihood of survival.

## Team Members

- Qiyuan Zhu
- Zella Yu

## Project Goals

- Use data-driven methods to explore patterns that influence startup success
- Help investors and founders make more informed decisions
- Build a user-friendly prediction tool

## Datasets

1. **Y Combinator Companies Dataset (2005-2024)**
   - 1,466 companies with clear outcomes
   - Labels: Acquired/Public (success), Inactive (failure)

2. **Crunchbase Startup Success/Fail Dataset**
   - ~66,000 companies
   - Rich information: total funding, investor count, founder profiles, etc.

## Key Features

- Founding year (converted to company age)
- Industry type
- Geographic region
- YC batch
- Team size
- Funding information
- Text features (tags, short_description)

## Models

- Logistic Regression (interpretable baseline)
- Random Forest (captures non-linear relationships)
- XGBoost / LightGBM (handles complex feature interactions)

## Project Structure

```
7180_final_project/
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── external/              # External data sources
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   │
│   ├── features/              # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── text_processing.py
│   │
│   ├── models/                # Model modules
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   └── visualization/         # Visualization modules
│       ├── __init__.py
│       └── plots.py
│
├── app/                       # Streamlit application
│   ├── app.py
│   └── utils.py
│
├── models/                    # Saved models
│
├── reports/                   # Reports and presentations
│   ├── figures/               # Figures
│   └── presentations/         # Presentations
│
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## Quick Start

### Method 1: Double-click to Run (Windows - Easiest)
Simply double-click `run_app.bat` in the project root directory.

### Method 2: Command Line

```bash
# Windows - Activate virtual environment
.\venv\Scripts\activate

# Run Streamlit app
streamlit run app/app.py

# App will open at http://localhost:8501
```

### Method 3: Test with Small Sample
```bash
.\venv\Scripts\activate
python tests/sample_test.py
```

## Preliminary Performance Results

Based on 80/20 train-test split on Y Combinator dataset:

| Model | Accuracy | F1 Score (Success) |
|-------|----------|-------------------|
| Logistic Regression | 63.9% | 0.47 |
| Random Forest | 67.3% | 0.57 |
| XGBoost | 67.3% | 0.55 |

*Note: All models perform better than random baseline (~58% accuracy)*

## Addressing Instructor Feedback

### Key Improvements Based on Feedback:

1. **Pre-curated Metrics & EDA**
   - Robust set of exploratory data analysis results
   - Interactive visualizations for user consumption
   - Industry, region, and funding stage success rates

2. **Handling Class Imbalance**
   - Dataset has 3 classes with 71% in "active" class
   - Implementing SMOTE, class weighting, and other balancing techniques
   - **Using confusion matrix as primary evaluation metric**
   - Reporting Precision, Recall, F1-Score, and AUC

3. **Feature Engineering**
   - Converting "year founded" to **"company age"**
   - Creating time-based features (funding intervals, age at funding)
   - Engineering meaningful temporal features

4. **Text Feature Processing**
   - TF-IDF vectorization for tags and descriptions
   - Word embeddings (Word2Vec, GloVe)
   - Topic modeling (LDA) for thematic features
   - Clear documentation of text processing pipeline

## Development Status

### ✅ Completed (95%)

- [x] Data loading and preprocessing pipeline
- [x] Feature engineering (company age, funding features, text features)
- [x] Text feature processing (TF-IDF, topic modeling, keyword extraction)
- [x] Model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- [x] Class imbalance handling (SMOTE, class weighting)
- [x] Model evaluation (confusion matrix, precision, recall, F1-score)
- [x] Interactive Streamlit application (5 pages)
- [x] Real data integration (4,974 YC companies)
- [x] Visualization dashboards and pre-curated metrics

### ⏳ Remaining (5%)

- [ ] Final presentation slides (template ready)
- [ ] Final report (template ready)

## Using Real Data

**Current Status**: ✅ Using 4,974 real YC companies

To switch between sample and real data:
1. Edit `app/data_config.py`
2. Set `USE_REAL_DATA = True` or `False`
3. Update `COLUMN_MAPPING` if needed
4. Run `python test_data_loading.py` to verify
5. Restart app

## Key Files

- `run_app.bat` - Start the application (double-click)
- `app/app.py` - Main Streamlit application
- `app/data_config.py` - Data loading configuration
- `test_data_loading.py` - Test data loading
- `example_workflow.py` - Complete ML pipeline demo
- `tests/sample_test.py` - Small sample test (12 companies)

## Presentation Guide (10 Minutes)

### Slide Structure:
1. **Problem & Importance** (1.5 min) - Why predict startup success?
2. **Data Overview** (1.5 min) - 4,974 YC companies, class imbalance
3. **Interesting Findings** (2 min) - Success rates by industry/region
4. **Addressing Challenges** (1.5 min) - SMOTE, confusion matrix, company age feature
5. **Model Performance** (1.5 min) - Comparison table, confusion matrices
6. **Live Demo** (2 min) - Interactive Streamlit app demonstration

### Key Points to Cover:
- ✅ Confusion matrix as primary metric (instructor requirement)
- ✅ Company age feature engineering (instructor feedback)
- ✅ Text feature processing methods (TF-IDF, topic modeling)
- ✅ Handling 71% class imbalance with SMOTE

## Final Report Checklist

- [ ] Executive summary
- [ ] Problem statement and importance
- [ ] Data sources and EDA findings
- [ ] Feature engineering details (especially company age)
- [ ] Text processing methodology (TF-IDF, LDA)
- [ ] Class imbalance handling approach
- [ ] Model comparison with confusion matrices
- [ ] Application screenshots
- [ ] Conclusions and future work
- [ ] References

## Contact

**Team**: Qiyuan Zhu, Zella Yu  
**GitHub**: https://github.com/vincentvicente/7180_final_project  
**Course**: 7180 Final Project
