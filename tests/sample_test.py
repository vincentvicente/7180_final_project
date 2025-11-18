"""
Small-sample test script for the startup success prediction pipeline.

This script fabricates a compact dataset (12 startups) so that we can
verify the preprocessing, feature engineering, and a simple classifier
end-to-end without the full datasets. Run with:

    python tests/sample_test.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer


def build_tiny_dataset() -> pd.DataFrame:
    """Create a small, diverse sample dataset for smoke testing."""
    return pd.DataFrame(
        [
            {"company_name": "AlphaAI", "year_founded": 2016, "status": "Acquired", "industry": "Tech",
             "region": "San Francisco", "team_size": 12, "total_funding": 80_000_000, "funding_rounds": 5,
             "investor_count": 12, "tags": "AI, machine learning", "short_description": "Enterprise AI platform"},
            {"company_name": "BetaHealth", "year_founded": 2014, "status": "Inactive", "industry": "Healthcare",
             "region": "Boston", "team_size": 30, "total_funding": 35_000_000, "funding_rounds": 4,
             "investor_count": 8, "tags": "healthtech, devices", "short_description": "Wearable medical devices"},
            {"company_name": "GammaPay", "year_founded": 2018, "status": "Active", "industry": "Finance",
             "region": "New York", "team_size": 45, "total_funding": 120_000_000, "funding_rounds": 6,
             "investor_count": 15, "tags": "fintech, payments", "short_description": "Cross-border payment rail"},
            {"company_name": "DeltaShop", "year_founded": 2019, "status": "Inactive", "industry": "E-commerce",
             "region": "London", "team_size": 18, "total_funding": 8_000_000, "funding_rounds": 2,
             "investor_count": 3, "tags": "ecommerce, marketplace", "short_description": "Luxury resale marketplace"},
            {"company_name": "EpsilonBio", "year_founded": 2013, "status": "Public", "industry": "Healthcare",
             "region": "San Francisco", "team_size": 60, "total_funding": 250_000_000, "funding_rounds": 7,
             "investor_count": 20, "tags": "biotech, genomics", "short_description": "Genomics diagnostics"},
            {"company_name": "ZetaLogistics", "year_founded": 2015, "status": "Active", "industry": "Other",
             "region": "Seattle", "team_size": 22, "total_funding": 15_000_000, "funding_rounds": 3,
             "investor_count": 5, "tags": "supply chain, SaaS", "short_description": "Supply chain visibility"},
            {"company_name": "EtaCloud", "year_founded": 2012, "status": "Acquired", "industry": "Tech",
             "region": "Austin", "team_size": 40, "total_funding": 60_000_000, "funding_rounds": 4,
             "investor_count": 10, "tags": "cloud, DevOps", "short_description": "Cloud automation tools"},
            {"company_name": "ThetaFoods", "year_founded": 2017, "status": "Inactive", "industry": "Other",
             "region": "Chicago", "team_size": 10, "total_funding": 5_000_000, "funding_rounds": 2,
             "investor_count": 4, "tags": "foodtech, delivery", "short_description": "Meal kit marketplace"},
            {"company_name": "IotaSecurity", "year_founded": 2020, "status": "Active", "industry": "Tech",
             "region": "San Francisco", "team_size": 25, "total_funding": 55_000_000, "funding_rounds": 3,
             "investor_count": 6, "tags": "cybersecurity, SaaS", "short_description": "Zero-trust security"},
            {"company_name": "KappaMobility", "year_founded": 2011, "status": "Public", "industry": "Other",
             "region": "Berlin", "team_size": 150, "total_funding": 400_000_000, "funding_rounds": 8,
             "investor_count": 25, "tags": "mobility, platform", "short_description": "Urban mobility platform"},
            {"company_name": "LambdaEnergy", "year_founded": 2010, "status": "Inactive", "industry": "Other",
             "region": "Houston", "team_size": 55, "total_funding": 90_000_000, "funding_rounds": 5,
             "investor_count": 9, "tags": "energy, hardware", "short_description": "Renewable storage tech"},
            {"company_name": "MuEdTech", "year_founded": 2018, "status": "Active", "industry": "Other",
             "region": "Toronto", "team_size": 14, "total_funding": 12_000_000, "funding_rounds": 3,
             "investor_count": 4, "tags": "edtech, AI", "short_description": "Adaptive learning platform"},
        ]
    )


def run_small_sample_test() -> None:
    """Run pipeline on the sample dataset and print metrics."""
    df = build_tiny_dataset()
    print(f"Sample size: {len(df)} companies")

    # Preprocess
    preprocessor = DataPreprocessor(target_column="status")
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_target(
        df,
        success_labels=["Acquired", "Public", "Active"],
        failure_labels=["Inactive"],
    )

    # Feature engineering (focus on company age per instructor feedback)
    engineer = FeatureEngineer(reference_year=2024)
    df = engineer.create_company_age(df, "year_founded")
    df = engineer.create_funding_features(df)
    df = engineer.create_team_features(df)

    feature_cols = [
        "company_age",
        "total_funding",
        "funding_rounds",
        "team_size",
        "investor_count",
        "log_total_funding",
    ]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    run_small_sample_test()

