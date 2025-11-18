"""
Quick test script for data loading
Run this before starting the app to verify data loads correctly
"""

from app.data_config import load_data, USE_REAL_DATA

print("=" * 80)
print("DATA LOADING TEST")
print("=" * 80)
print(f"\nMode: {'REAL DATA' if USE_REAL_DATA else 'SAMPLE DATA'}\n")

try:
    df = load_data()

    print("Data loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    print("First 3 rows:")
    print(df.head(3))
    print()

    print("Data types:")
    print(df.dtypes)
    print()

    if "target" in df.columns:
        print("Target distribution:")
        print(df["target"].value_counts())
        print()

    print("=" * 80)
    print("TEST PASSED - Ready to start app!")
    print("=" * 80)

except Exception as e:
    print(f"ERROR: {e}\n")
    print("Please check:")
    print("1. File path in app/data_config.py")
    print("2. Column mappings")
    print("3. File format (CSV or Excel)")

