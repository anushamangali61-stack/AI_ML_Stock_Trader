import pandas as pd

from src.features import (
    create_features,
    create_trade_labels,
    get_ml_features,
)

from src.ml_model import (
    load_model,
    predict_trade,
    explain_trade,
)


# =========================================================
# 1. Load Data
# =========================================================

df = pd.read_csv(
    "data/sample_data.csv"
)


# =========================================================
# 2. Create Features
# =========================================================

df = create_features(df)


# =========================================================
# 3. Create Historical Labels
# =========================================================

df = create_trade_labels(
    df,
    lookahead=10,
    target_pct=0.002
)


# =========================================================
# 4. Load Trained Model
# =========================================================

model = load_model(
    "models/trade_filter_model.pkl"
)


# =========================================================
# 5. Get ML Features
# =========================================================

feature_columns = get_ml_features()


# =========================================================
# 6. Find Crossover Signals
# =========================================================

signals = df[
    df["signal"] != "NONE"
].copy()


if signals.empty:

    print("No crossover signal found.")

    raise SystemExit


# Get latest crossover
latest_signal = signals.iloc[-1]


# =========================================================
# 7. Prepare Feature Row
# =========================================================

feature_row = latest_signal[
    feature_columns
].to_frame().T


# =========================================================
# 8. AI Prediction
# =========================================================

result = predict_trade(
    model,
    feature_row,
    threshold=0.60
)


probability = result["probability"]

decision = result["decision"]


# =========================================================
# 9. Generate Explanation
# =========================================================

reason = explain_trade(
    latest_signal,
    probability,
    decision
)


# =========================================================
# 10. Display Result
# =========================================================

print()
print("=" * 60)
print("AI/ML TRADE FILTER")
print("=" * 60)

print(
    f"Timestamp        : "
    f"{latest_signal['timestamp']}"
)

print(
    f"Price            : "
    f"{latest_signal['close']:.2f}"
)

print(
    f"Signal            : "
    f"{latest_signal['signal']}"
)

print(
    f"SMMA 20           : "
    f"{latest_signal['smma_20']:.4f}"
)

print(
    f"SMMA 120          : "
    f"{latest_signal['smma_120']:.4f}"
)

print(
    f"LTQ               : "
    f"{latest_signal['ltq']}"
)

print(
    f"Bid Quantity      : "
    f"{latest_signal['bid_quantity']}"
)

print(
    f"Ask Quantity      : "
    f"{latest_signal['ask_quantity']}"
)

print(
    f"Bid/Ask Imbalance : "
    f"{latest_signal['bid_ask_imbalance']:.4f}"
)

print(
    f"AI Probability    : "
    f"{probability * 100:.2f}%"
)

print(
    f"Decision          : "
    f"{decision}"
)

print(
    f"Reason            : "
    f"{reason}"
)

print("=" * 60)