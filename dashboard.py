import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from src.features import create_features

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI/ML Stock Trade Filter",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI/ML Real-Time Stock Screening & Trade Filter")
st.caption(
    "SMMA 20/120 + LTQ + Bid/Ask Market Depth + Machine Learning "
    "Paper Trading Dashboard"
)

# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/sample_data.csv"
MODEL_FILE = "models/trade_filter_model.pkl"

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = create_features(df)

    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


df = load_data()
model = load_model()

# ============================================================
# CHECK DATA
# ============================================================

if df.empty:
    st.error("No market data available.")
    st.stop()

latest = df.iloc[-1]

# ============================================================
# AI PREDICTION
# ============================================================

feature_columns = [
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
    "volume_change"
]

probability = 0.50

if model is not None:

    try:

        X_latest = latest[feature_columns].to_frame().T

        X_latest = X_latest.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        probability = float(
            model.predict_proba(X_latest)[0][1]
        )

    except Exception as e:

        st.warning(
            f"ML prediction unavailable: {e}"
        )

# ============================================================
# SIGNAL
# ============================================================

signal = latest.get("signal", "NONE")

# ============================================================
# DECISION LOGIC
# ============================================================

reasons = []

if probability >= 0.65:

    decision = "ACCEPT"

else:

    decision = "AVOID"

if latest["ltq_change"] < 0:
    reasons.append("LTQ decreased")

if latest["bid_ask_imbalance"] < -0.10:
    reasons.append("Strong ask-side pressure")

if latest["bid_ask_imbalance"] > 0.10:
    reasons.append("Strong bid-side support")

if latest["price_change"] < 0:
    reasons.append("Price moving downward")

if latest["price_change"] > 0:
    reasons.append("Price moving upward")

if not reasons:
    reasons.append("Market conditions stable")

reason_text = "; ".join(reasons)

# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "LTP",
        f"₹{latest['close']:.2f}"
    )

with col2:
    st.metric(
        "Signal",
        signal
    )

with col3:
    st.metric(
        "AI Probability",
        f"{probability * 100:.2f}%"
    )

with col4:
    st.metric(
        "Decision",
        decision
    )

with col5:
    st.metric(
        "LTQ",
        f"{int(latest['ltq']):,}"
    )

# ============================================================
# MARKET DEPTH
# ============================================================

st.subheader("📊 Market Depth")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Bid Price",
        f"₹{latest['bid_price']:.2f}"
    )

with c2:
    st.metric(
        "Bid Quantity",
        f"{int(latest['bid_quantity']):,}"
    )

with c3:
    st.metric(
        "Ask Price",
        f"₹{latest['ask_price']:.2f}"
    )

with c4:
    st.metric(
        "Ask Quantity",
        f"{int(latest['ask_quantity']):,}"
    )

with c5:
    st.metric(
        "Bid/Ask Imbalance",
        f"{latest['bid_ask_imbalance']:.3f}"
    )

# ============================================================
# DECISION
# ============================================================

st.subheader("🤖 AI/ML Decision")

if decision == "ACCEPT":
    st.success(
        f"ACCEPT — AI probability: "
        f"{probability * 100:.2f}%"
    )
else:
    st.error(
        f"AVOID — AI probability: "
        f"{probability * 100:.2f}%"
    )

st.info(
    f"Reason: {reason_text}"
)

# ============================================================
# SMMA CHART
# ============================================================

st.subheader("📈 SMMA 20 / SMMA 120")

chart_data = df.tail(300)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["close"],
        mode="lines",
        name="LTP"
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["smma_20"],
        mode="lines",
        name="SMMA 20"
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["smma_120"],
        mode="lines",
        name="SMMA 120"
    )
)

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# LTQ CHART
# ============================================================

st.subheader("📊 LTQ")

fig_ltq = go.Figure()

fig_ltq.add_trace(
    go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["ltq"],
        mode="lines",
        name="LTQ"
    )
)

fig_ltq.update_layout(
    height=350,
    xaxis_title="Time",
    yaxis_title="LTQ"
)

st.plotly_chart(
    fig_ltq,
    use_container_width=True
)

# ============================================================
# BID / ASK IMBALANCE
# ============================================================

st.subheader("⚖️ Bid / Ask Imbalance")

fig_depth = go.Figure()

fig_depth.add_trace(
    go.Scatter(
        x=chart_data["timestamp"],
        y=chart_data["bid_ask_imbalance"],
        mode="lines",
        name="Bid/Ask Imbalance"
    )
)

fig_depth.add_hline(
    y=0,
    line_dash="dash"
)

fig_depth.update_layout(
    height=350,
    xaxis_title="Time",
    yaxis_title="Imbalance"
)

st.plotly_chart(
    fig_depth,
    use_container_width=True
)

# ============================================================
# RECENT SIGNALS
# ============================================================

st.subheader("🔔 Recent Crossover Signals")

signals = df[df["signal"].isin(["BUY", "SELL"])].copy()

if len(signals) > 0:

    display_columns = [
        "timestamp",
        "close",
        "smma_20",
        "smma_120",
        "signal",
        "ltq",
        "bid_quantity",
        "ask_quantity",
        "bid_ask_imbalance"
    ]

    st.dataframe(
        signals[display_columns].tail(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("No crossover signals detected.")

# ============================================================
# PAPER TRADING RESULTS
# ============================================================

st.subheader("💰 Paper Trading")

RESULT_FILE = "data/paper_trading_results.csv"

try:

    results = pd.read_csv(RESULT_FILE)

    if not results.empty:

        total_trades = len(results)

        profitable = (
            results["pnl"] > 0
        ).sum()

        losing = (
            results["pnl"] <= 0
        ).sum()

        total_pnl = results["pnl"].sum()

        win_rate = (
            profitable / total_trades * 100
            if total_trades > 0
            else 0
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Trades",
                total_trades
            )

        with p2:
            st.metric(
                "Profitable",
                profitable
            )

        with p3:
            st.metric(
                "Losing",
                losing
            )

        with p4:
            st.metric(
                "Win Rate",
                f"{win_rate:.2f}%"
            )

        st.metric(
            "Total Simulated P/L",
            f"{total_pnl:.4f}"
        )

        st.dataframe(
            results.tail(20),
            use_container_width=True,
            hide_index=True
        )

except FileNotFoundError:

    st.warning(
        "Paper trading results file not found."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ Prototype uses synthetic market data for testing. "
    "No real-money trading is performed."
)