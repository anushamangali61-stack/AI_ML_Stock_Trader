import os
import pandas as pd
import joblib


print("=" * 70)
print("AI/ML STOCK TRADING PROJECT VALIDATION")
print("=" * 70)

errors = 0


def check_file(path):
    global errors

    if os.path.exists(path):
        print(f"[PASS] {path}")
    else:
        print(f"[FAIL] {path} not found")
        errors += 1


# ---------------------------------------------------------
# Required files
# ---------------------------------------------------------

print("\n1. Checking required files...")

required_files = [
    "data/sample_data.csv",
    "data/processed_data.csv",
    "data/paper_trading_results.csv",
    "data/basic_smma_results.csv",
    "data/ai_ml_results.csv",
    "models/trade_filter_model.pkl",
]

for file in required_files:
    check_file(file)


# ---------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------

print("\n2. Checking dataset...")

try:
    df = pd.read_csv("data/sample_data.csv")

    print(f"[PASS] Dataset loaded")
    print(f"       Rows: {len(df)}")
    print(f"       Columns: {len(df.columns)}")

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ltq",
        "bid_price",
        "ask_price",
        "bid_quantity",
        "ask_quantity",
        "bid_ask_imbalance",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(f"[FAIL] Missing columns: {missing_columns}")
        errors += 1
    else:
        print("[PASS] Required market-data columns present")

except Exception as e:
    print(f"[FAIL] Dataset validation error: {e}")
    errors += 1


# ---------------------------------------------------------
# ML model validation
# ---------------------------------------------------------

print("\n3. Checking ML model...")

try:
    model_path = "models/trade_filter_model.pkl"

    model = joblib.load(model_path)

    print("[PASS] ML model loaded successfully")
    print(f"       Model type: {type(model).__name__}")

except Exception as e:
    print(f"[FAIL] ML model could not be loaded: {e}")
    errors += 1


# ---------------------------------------------------------
# Paper trading report
# ---------------------------------------------------------

print("\n4. Checking paper trading results...")

try:
    results = pd.read_csv(
        "data/paper_trading_results.csv"
    )

    print("[PASS] Paper trading report loaded")
    print(f"       Trades: {len(results)}")

except Exception as e:
    print(f"[FAIL] Paper trading report error: {e}")
    errors += 1


# ---------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------

print("\n5. Checking strategy comparison reports...")

try:
    basic = pd.read_csv(
        "data/basic_smma_results.csv"
    )

    ai_ml = pd.read_csv(
        "data/ai_ml_results.csv"
    )

    print("[PASS] Basic SMMA results loaded")
    print(f"       Basic trades: {len(basic)}")

    print("[PASS] AI/ML results loaded")
    print(f"       AI/ML trades: {len(ai_ml)}")

except Exception as e:
    print(f"[FAIL] Comparison report error: {e}")
    errors += 1


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 70)

if errors == 0:
    print("PROJECT VALIDATION PASSED")
    print("All required files, dataset, model and reports are valid.")
else:
    print("PROJECT VALIDATION FAILED")
    print(f"Total errors: {errors}")

print("=" * 70)