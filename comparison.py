import pandas as pd

from src.features import (
    create_features,
    create_trade_labels,
)

from src.ml_model import (
    load_model,
    predict_trade,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/sample_data.csv"

MODEL_PATH = "models/trade_filter_model.pkl"

LOOKAHEAD = 10

TARGET_PCT = 0.002

AI_THRESHOLD = 0.60


# =========================================================
# ML FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "smma_spread",
    "smma_spread_change",
    "ltq",
    "ltq_change",
    "ltq_acceleration",
    "bid_quantity",
    "ask_quantity",
    "bid_ask_imbalance",
    "bid_ask_spread",
    "price_change",
    "price_change_pct",
    "volume",
    "volume_change",
]


# =========================================================
# 1. LOAD DATA
# =========================================================

print()
print("=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

df = create_features(
    df
)

df = create_trade_labels(
    df,
    lookahead=LOOKAHEAD,
    target_pct=TARGET_PCT
)


# =========================================================
# 2. LOAD MODEL
# =========================================================

model = load_model(
    MODEL_PATH
)


# =========================================================
# 3. GET CROSSOVER SIGNALS
# =========================================================

signals = df[
    df["signal"] != "NONE"
].copy()

print(
    f"Total crossover signals: {len(signals)}"
)


# =========================================================
# 4. BASIC SMMA STRATEGY
# =========================================================

basic_trades = []


for signal_index in signals.index:

    signal_row = df.loc[
        signal_index
    ]

    signal = signal_row[
        "signal"
    ]

    entry_price = signal_row[
        "close"
    ]

    future_rows = df.loc[
        signal_index + 1:
        signal_index + LOOKAHEAD
    ]

    if future_rows.empty:

        continue

    exit_row = future_rows.iloc[
        -1
    ]

    exit_price = exit_row[
        "close"
    ]

    if signal == "BUY":

        pnl = (
            exit_price
            - entry_price
        )

    else:

        pnl = (
            entry_price
            - exit_price
        )

    pnl_pct = (
        pnl
        / entry_price
    ) * 100

    basic_trades.append({

        "entry_time":
            signal_row["timestamp"],

        "exit_time":
            exit_row["timestamp"],

        "signal":
            signal,

        "entry_price":
            entry_price,

        "exit_price":
            exit_price,

        "pnl":
            pnl,

        "pnl_pct":
            pnl_pct

    })


basic_results = pd.DataFrame(
    basic_trades
)


# =========================================================
# 5. AI/ML FILTERED STRATEGY
# =========================================================

ai_trades = []

accepted = 0

avoided = 0


for signal_index in signals.index:

    signal_row = df.loc[
        signal_index
    ]

    feature_row = signal_row[
        FEATURE_COLUMNS
    ].to_frame().T

    prediction = predict_trade(
        model=model,
        feature_row=feature_row,
        threshold=AI_THRESHOLD
    )

    probability = prediction[
        "probability"
    ]

    decision = prediction[
        "decision"
    ]


    # -----------------------------------------------------
    # Avoid weak signal
    # -----------------------------------------------------

    if decision != "ACCEPT":

        avoided += 1

        continue


    accepted += 1


    signal = signal_row[
        "signal"
    ]

    entry_price = signal_row[
        "close"
    ]


    future_rows = df.loc[
        signal_index + 1:
        signal_index + LOOKAHEAD
    ]


    if future_rows.empty:

        continue


    exit_row = future_rows.iloc[
        -1
    ]

    exit_price = exit_row[
        "close"
    ]


    if signal == "BUY":

        pnl = (
            exit_price
            - entry_price
        )

    else:

        pnl = (
            entry_price
            - exit_price
        )


    pnl_pct = (
        pnl
        / entry_price
    ) * 100


    ai_trades.append({

        "entry_time":
            signal_row["timestamp"],

        "exit_time":
            exit_row["timestamp"],

        "signal":
            signal,

        "entry_price":
            entry_price,

        "exit_price":
            exit_price,

        "ai_probability":
            probability,

        "decision":
            decision,

        "pnl":
            pnl,

        "pnl_pct":
            pnl_pct

    })


ai_results = pd.DataFrame(
    ai_trades
)


# =========================================================
# 6. BASIC STRATEGY STATISTICS
# =========================================================

basic_total = len(
    basic_results
)

basic_profitable = (
    basic_results["pnl"] > 0
).sum()

basic_losing = (
    basic_results["pnl"] <= 0
).sum()

basic_pnl = (
    basic_results["pnl"].sum()
)

basic_win_rate = (
    basic_profitable
    / basic_total
) * 100 if basic_total else 0


# =========================================================
# 7. AI STRATEGY STATISTICS
# =========================================================

ai_total = len(
    ai_results
)

ai_profitable = (
    ai_results["pnl"] > 0
).sum()

ai_losing = (
    ai_results["pnl"] <= 0
).sum()

ai_pnl = (
    ai_results["pnl"].sum()
)

ai_win_rate = (
    ai_profitable
    / ai_total
) * 100 if ai_total else 0


# =========================================================
# 8. LOSING TRADES AVOIDED
# =========================================================

basic_losing_indices = set(
    basic_results[
        basic_results["pnl"] <= 0
    ].index
)

ai_accepted_indices = set(
    ai_results.index
)

# Number of signals rejected by AI
# that would otherwise have been traded
# by the basic strategy.

losing_trades_avoided = (
    len(basic_losing_indices)
    - ai_losing
)

if basic_losing > 0:

    losing_avoidance_pct = (
        losing_trades_avoided
        / basic_losing
    ) * 100

else:

    losing_avoidance_pct = 0


# =========================================================
# 9. PRINT COMPARISON
# =========================================================

print()
print("=" * 70)
print("BASIC SMMA STRATEGY")
print("=" * 70)

print(
    f"Total trades      : {basic_total}"
)

print(
    f"Profitable trades : {basic_profitable}"
)

print(
    f"Losing trades     : {basic_losing}"
)

print(
    f"Win rate          : "
    f"{basic_win_rate:.2f}%"
)

print(
    f"Total P/L         : "
    f"{basic_pnl:.4f}"
)


print()
print("=" * 70)
print("AI/ML FILTERED STRATEGY")
print("=" * 70)

print(
    f"Total crossover signals : "
    f"{len(signals)}"
)

print(
    f"Accepted signals        : "
    f"{accepted}"
)

print(
    f"Avoided signals         : "
    f"{avoided}"
)

print(
    f"Profitable trades       : "
    f"{ai_profitable}"
)

print(
    f"Losing trades           : "
    f"{ai_losing}"
)

print(
    f"Win rate                : "
    f"{ai_win_rate:.2f}%"
)

print(
    f"Total P/L               : "
    f"{ai_pnl:.4f}"
)


print()
print("=" * 70)
print("LOSS AVOIDANCE")
print("=" * 70)

print(
    f"Basic strategy losses   : "
    f"{basic_losing}"
)

print(
    f"Losing trades avoided   : "
    f"{losing_trades_avoided}"
)

print(
    f"Loss avoidance rate     : "
    f"{losing_avoidance_pct:.2f}%"
)


# =========================================================
# 10. SAVE REPORTS
# =========================================================

basic_results.to_csv(
    "data/basic_smma_results.csv",
    index=False
)

ai_results.to_csv(
    "data/ai_ml_results.csv",
    index=False
)


print()
print("=" * 70)
print("REPORTS SAVED")
print("=" * 70)

print(
    "data/basic_smma_results.csv"
)

print(
    "data/ai_ml_results.csv"
)

print()
print("Comparison completed successfully!")