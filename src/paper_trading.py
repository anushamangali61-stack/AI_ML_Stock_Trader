import pandas as pd


def check_trade_deterioration(
    entry_row: pd.Series,
    current_row: pd.Series,
    signal: str
):
    """
    Monitor an accepted paper trade.

    Returns:
        ACCEPT
        DETERIORATING
        EXIT
    """

    reasons = []

    deterioration_count = 0

    # =====================================================
    # 1. LTQ deterioration
    # =====================================================

    if (
        current_row["ltq"]
        < entry_row["ltq"]
    ):
        deterioration_count += 1

        reasons.append(
            "LTQ decreased"
        )

    # =====================================================
    # 2. Bid/Ask deterioration
    # =====================================================

    if signal == "BUY":

        if (
            current_row["bid_ask_imbalance"]
            < entry_row["bid_ask_imbalance"]
        ):
            deterioration_count += 1

            reasons.append(
                "Bid support weakened"
            )

    elif signal == "SELL":

        if (
            current_row["bid_ask_imbalance"]
            > entry_row["bid_ask_imbalance"]
        ):
            deterioration_count += 1

            reasons.append(
                "Ask pressure weakened"
            )

    # =====================================================
    # 3. Price deterioration
    # =====================================================

    if signal == "BUY":

        if (
            current_row["close"]
            < entry_row["close"]
        ):
            deterioration_count += 1

            reasons.append(
                "Price moved below entry"
            )

    elif signal == "SELL":

        if (
            current_row["close"]
            > entry_row["close"]
        ):
            deterioration_count += 1

            reasons.append(
                "Price moved above entry"
            )

    # =====================================================
    # 4. SMMA deterioration
    # =====================================================

    current_spread = (
        current_row["smma_20"]
        - current_row["smma_120"]
    )

    entry_spread = (
        entry_row["smma_20"]
        - entry_row["smma_120"]
    )

    if signal == "BUY":

        if current_spread < entry_spread:

            deterioration_count += 1

            reasons.append(
                "SMMA bullish spread weakened"
            )

    elif signal == "SELL":

        if current_spread > entry_spread:

            deterioration_count += 1

            reasons.append(
                "SMMA bearish spread weakened"
            )

    # =====================================================
    # Final Decision
    # =====================================================

    if deterioration_count >= 3:

        status = "EXIT"

    elif deterioration_count >= 2:

        status = "DETERIORATING"

    else:

        status = "ACCEPT"

    return {
        "status": status,
        "deterioration_count": deterioration_count,
        "reasons": reasons
    }