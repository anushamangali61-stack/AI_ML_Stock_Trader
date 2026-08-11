import pandas as pd
import numpy as np


# =========================================================
# Configuration
# =========================================================

np.random.seed(42)

rows = 5000


# =========================================================
# Generate Price Movement
# =========================================================

time = np.arange(rows)

# Multiple trends to create more SMMA crossovers
trend_1 = 3.0 * np.sin(time / 80)
trend_2 = 2.0 * np.sin(time / 25)

noise = np.random.normal(
    0,
    0.25,
    rows
)

prices = (
    100
    + trend_1
    + trend_2
    + np.cumsum(noise)
)


# =========================================================
# OHLCV Data
# =========================================================

df = pd.DataFrame({

    "timestamp": pd.date_range(
        "2026-08-11 09:15",
        periods=rows,
        freq="min"
    ),

    "open": (
        prices
        + np.random.normal(
            0,
            0.1,
            rows
        )
    ),

    "high": (
        prices
        + np.random.uniform(
            0,
            0.5,
            rows
        )
    ),

    "low": (
        prices
        - np.random.uniform(
            0,
            0.5,
            rows
        )
    ),

    "close": prices,

    "volume": np.random.randint(
        100000,
        500000,
        rows
    )
})


# =========================================================
# Simulated LTQ
# =========================================================

df["ltq"] = np.random.randint(
    1000,
    10000,
    rows
)


# =========================================================
# Bid / Ask Prices
# =========================================================

df["bid_price"] = (
    df["close"]
    - np.random.uniform(
        0.01,
        0.10,
        rows
    )
)

df["ask_price"] = (
    df["close"]
    + np.random.uniform(
        0.01,
        0.10,
        rows
    )
)


# =========================================================
# Bid / Ask Quantities
# =========================================================

df["bid_quantity"] = np.random.randint(
    500000,
    2000000,
    rows
)

df["ask_quantity"] = np.random.randint(
    500000,
    2000000,
    rows
)


# =========================================================
# Bid / Ask Imbalance
# =========================================================

df["bid_ask_imbalance"] = (
    (
        df["bid_quantity"]
        - df["ask_quantity"]
    )
    /
    (
        df["bid_quantity"]
        + df["ask_quantity"]
    )
)


# =========================================================
# Additional Market Features
# =========================================================

df["ltq_change"] = (
    df["ltq"].diff()
)

df["price_change"] = (
    df["close"].diff()
)

df["volume_change"] = (
    df["volume"].diff()
)


# =========================================================
# Save
# =========================================================

df.to_csv(
    "data/sample_data.csv",
    index=False
)


# =========================================================
# Confirmation
# =========================================================

print(
    "Sample data created successfully!"
)

print(
    f"Rows: {len(df)}"
)

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nFirst 5 rows:")

print(
    df.head()
)