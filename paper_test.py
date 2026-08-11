import pandas as pd

from src.features import (
    create_features,
    create_trade_labels,
)

from src.ml_model import (
    load_model,
    predict_trade,
)

from src.paper_trading import (
    check_trade_deterioration,
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
# ML FEATURE COLUMNS
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
print("LOADING MARKET DATA")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Rows loaded: {len(df)}"
)


# =========================================================
# 2. CREATE FEATURES
# =========================================================

df = create_features(
    df
)

print(
    "Features created successfully."
)


# =========================================================
# 3. CREATE HISTORICAL LABELS
# =========================================================

df = create_trade_labels(
    df,
    lookahead=LOOKAHEAD,
    target_pct=TARGET_PCT
)

print(
    "Historical labels created."
)


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

print()
print(
    "Loading trained ML model..."
)

model = load_model(
    MODEL_PATH
)

print(
    f"Model loaded: {MODEL_PATH}"
)


# =========================================================
# 5. FIND CROSSOVER SIGNALS
# =========================================================

signals = df[
    df["signal"] != "NONE"
].copy()

print()
print("=" * 70)
print("CROSSOVER ANALYSIS")
print("=" * 70)

print(
    f"Total crossover signals: {len(signals)}"
)


if signals.empty:

    print(
        "No crossover signals found."
    )

    raise SystemExit


# =========================================================
# 6. PAPER TRADING VARIABLES
# =========================================================

paper_trades = []

total_signals = len(signals)

accepted_signals = 0

avoided_signals = 0


# =========================================================
# 7. PROCESS EVERY CROSSOVER
# =========================================================

for signal_index in signals.index:

    signal_row = df.loc[
        signal_index
    ]

    signal = signal_row[
        "signal"
    ]

    # -----------------------------------------------------
    # Prepare ML input
    # -----------------------------------------------------

    feature_row = signal_row[
        FEATURE_COLUMNS
    ].to_frame().T

    # -----------------------------------------------------
    # AI prediction
    # -----------------------------------------------------

    result = predict_trade(
        model=model,
        feature_row=feature_row,
        threshold=AI_THRESHOLD
    )

    probability = result[
        "probability"
    ]

    decision = result[
        "decision"
    ]

    # -----------------------------------------------------
    # AVOID
    # -----------------------------------------------------

    if decision == "AVOID":

        avoided_signals += 1

        continue

    # -----------------------------------------------------
    # ACCEPT
    # -----------------------------------------------------

    accepted_signals += 1

    entry_price = signal_row[
        "close"
    ]

    entry_time = signal_row[
        "timestamp"
    ]

    # -----------------------------------------------------
    # Monitor next candles
    # -----------------------------------------------------

    future_rows = df.loc[
        signal_index + 1:
        signal_index + LOOKAHEAD
    ]

    trade_status = "ACCEPT"

    exit_price = entry_price

    exit_index = signal_index

    reasons = []

    # -----------------------------------------------------
    # Monitor trade
    # -----------------------------------------------------

    for current_index, current_row in future_rows.iterrows():

        monitoring = check_trade_deterioration(

            entry_row=signal_row,

            current_row=current_row,

            signal=signal

        )

        current_status = monitoring[
            "status"
        ]

        current_reasons = monitoring[
            "reasons"
        ]

        # ---------------------------------------------
        # EXIT
        # ---------------------------------------------

        if current_status == "EXIT":

            trade_status = "EXIT"

            exit_price = current_row[
                "close"
            ]

            exit_index = current_index

            reasons = current_reasons

            break

        # ---------------------------------------------
        # DETERIORATING
        # ---------------------------------------------

        elif current_status == "DETERIORATING":

            trade_status = "DETERIORATING"

            reasons = current_reasons

    # =====================================================
    # If no EXIT happened
    # =====================================================

    if exit_index == signal_index:

        if len(future_rows) > 0:

            final_row = future_rows.iloc[
                -1
            ]

            exit_price = final_row[
                "close"
            ]

            exit_index = future_rows.index[
                -1
            ]


    # =====================================================
    # Calculate P/L
    # =====================================================

    if signal == "BUY":

        pnl = (
            exit_price
            - entry_price
        )

    elif signal == "SELL":

        pnl = (
            entry_price
            - exit_price
        )

    else:

        pnl = 0


    # =====================================================
    # P/L Percentage
    # =====================================================

    pnl_pct = (
        pnl
        / entry_price
    ) * 100


    # =====================================================
    # EXIT TIME
    # =====================================================

    exit_time = df.loc[
        exit_index,
        "timestamp"
    ]


    # =====================================================
    # SAVE PAPER TRADE
    # =====================================================

    paper_trades.append({

        "entry_time":
            entry_time,

        "exit_time":
            exit_time,

        "signal":
            signal,

        "entry_price":
            round(
                entry_price,
                4
            ),

        "exit_price":
            round(
                exit_price,
                4
            ),

        "ai_probability":
            round(
                probability,
                4
            ),

        "decision":
            decision,

        "status":
            trade_status,

        "pnl":
            round(
                pnl,
                4
            ),

        "pnl_pct":
            round(
                pnl_pct,
                4
            ),

        "reason":
            "; ".join(
                reasons
            )

    })


# =========================================================
# 8. CREATE RESULTS DATAFRAME
# =========================================================

results = pd.DataFrame(
    paper_trades
)


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

print()
print("=" * 70)
print("PAPER TRADING RESULTS")
print("=" * 70)


if results.empty:

    print(
        "No trades were accepted by the AI filter."
    )

else:

    print(
        results.to_string(
            index=False
        )
    )


# =========================================================
# 10. PERFORMANCE STATISTICS
# =========================================================

print()
print("=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

print(
    f"Total crossover signals : "
    f"{total_signals}"
)

print(
    f"Accepted signals        : "
    f"{accepted_signals}"
)

print(
    f"Avoided signals         : "
    f"{avoided_signals}"
)


if not results.empty:

    profitable_trades = (
        results["pnl"] > 0
    ).sum()

    losing_trades = (
        results["pnl"] <= 0
    ).sum()

    total_pnl = (
        results["pnl"].sum()
    )

    average_pnl = (
        results["pnl"].mean()
    )

    win_rate = (
        profitable_trades
        / len(results)
    ) * 100

    print(
        f"Profitable trades       : "
        f"{profitable_trades}"
    )

    print(
        f"Losing trades           : "
        f"{losing_trades}"
    )

    print(
        f"Win rate                : "
        f"{win_rate:.2f}%"
    )

    print(
        f"Total simulated P/L     : "
        f"{total_pnl:.4f}"
    )

    print(
        f"Average trade P/L       : "
        f"{average_pnl:.4f}"
    )


# =========================================================
# 11. SAVE PAPER TRADING REPORT
# =========================================================

if not results.empty:

    results.to_csv(
        "data/paper_trading_results.csv",
        index=False
    )

    print()
    print(
        "Paper trading report saved:"
    )

    print(
        "data/paper_trading_results.csv"
    )


# =========================================================
# 12. FINAL STATUS
# =========================================================

print()
print("=" * 70)
print("PAPER TRADING TEST COMPLETED")
print("=" * 70)