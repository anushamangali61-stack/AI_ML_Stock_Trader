import pandas as pd

from src.indicators import add_smma_indicators, detect_crossovers


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical and market-depth features
    for the AI/ML trade-filtering model.
    """

    df = df.copy()

    # -------------------------------------------------
    # SMMA indicators
    # -------------------------------------------------

    df = add_smma_indicators(df)

    # -------------------------------------------------
    # SMMA-based features
    # -------------------------------------------------

    df["smma_spread"] = (
        df["smma_20"] - df["smma_120"]
    )

    df["smma_spread_change"] = (
        df["smma_spread"].diff()
    )

    # -------------------------------------------------
    # LTQ features
    # -------------------------------------------------

    df["ltq_change"] = df["ltq"].diff()

    df["ltq_acceleration"] = (
        df["ltq_change"].diff()
    )

    # -------------------------------------------------
    # Price features
    # -------------------------------------------------

    df["price_change"] = (
        df["close"].diff()
    )

    df["price_change_pct"] = (
        df["close"].pct_change() * 100
    )

    # -------------------------------------------------
    # Volume features
    # -------------------------------------------------

    df["volume_change"] = (
        df["volume"].diff()
    )

    # -------------------------------------------------
    # Bid / Ask features
    # -------------------------------------------------

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

    df["bid_ask_spread"] = (
        df["ask_price"]
        - df["bid_price"]
    )

    # -------------------------------------------------
    # Detect SMMA crossover
    # -------------------------------------------------

    df = detect_crossovers(df)

    return df


def get_ml_features():
    """
    Features that will be given to the ML model.
    """

    return [
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
def create_trade_labels(
    df: pd.DataFrame,
    lookahead: int = 10,
    target_pct: float = 0.002
) -> pd.DataFrame:
    """
    Create training labels based on future price movement.
    """

    df = df.copy()

    # Future price after the lookahead period
    df["future_close"] = (
        df["close"].shift(-lookahead)
    )

    # Future percentage return
    df["future_return"] = (
        (df["future_close"] - df["close"])
        / df["close"]
    )

    # Default: unsuccessful trade
    df["target"] = 0

    # Successful BUY
    buy_success = (
        (df["signal"] == "BUY")
        & (df["future_return"] >= target_pct)
    )

    # Successful SELL
    sell_success = (
        (df["signal"] == "SELL")
        & (df["future_return"] <= -target_pct)
    )

    # Mark successful trades
    df.loc[
        buy_success | sell_success,
        "target"
    ] = 1

    # Remove rows without future data
    df = df.dropna(
        subset=["future_close"]
    )

    return df