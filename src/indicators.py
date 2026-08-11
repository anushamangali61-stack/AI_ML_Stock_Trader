import pandas as pd


def calculate_smma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Smoothed Moving Average (SMMA).

    SMMA:
    First value = Simple Moving Average
    Next values = ((Previous SMMA * (period - 1)) + Current Price) / period
    """

    smma = pd.Series(index=series.index, dtype=float)

    if len(series) < period:
        return smma

    # Initial SMA
    smma.iloc[period - 1] = series.iloc[:period].mean()

    # Remaining SMMA values
    for i in range(period, len(series)):
        smma.iloc[i] = (
            (smma.iloc[i - 1] * (period - 1)) + series.iloc[i]
        ) / period

    return smma


def add_smma_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMMA 20 and SMMA 120 columns."""

    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    df["smma_20"] = calculate_smma(df["close"], 20)
    df["smma_120"] = calculate_smma(df["close"], 120)

    return df


def detect_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect SMMA crossovers.

    BUY  = SMMA 20 crosses above SMMA 120
    SELL = SMMA 20 crosses below SMMA 120
    """

    df = df.copy()

    previous_20 = df["smma_20"].shift(1)
    previous_120 = df["smma_120"].shift(1)

    df["signal"] = "NONE"

    buy_condition = (
        (previous_20 <= previous_120)
        & (df["smma_20"] > df["smma_120"])
    )

    sell_condition = (
        (previous_20 >= previous_120)
        & (df["smma_20"] < df["smma_120"])
    )

    df.loc[buy_condition, "signal"] = "BUY"
    df.loc[sell_condition, "signal"] = "SELL"

    return df