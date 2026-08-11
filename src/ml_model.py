import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_model(
    df: pd.DataFrame,
    feature_columns: list,
    model_path: str = "models/trade_filter_model.pkl"
):
    """
    Train Random Forest model to predict
    whether a crossover is likely to succeed.
    """

    # Only crossover rows are useful for training
    training_df = df[
        df["signal"] != "NONE"
    ].copy()

    # Remove missing values
    training_df = training_df.dropna(
        subset=feature_columns + ["target"]
    )

    if len(training_df) < 10:
        raise ValueError(
            "Not enough crossover samples for ML training."
        )

    # Need both target classes
    if training_df["target"].nunique() < 2:
        raise ValueError(
            "Training data contains only one target class."
        )

    X = training_df[feature_columns]
    y = training_df["target"]

    # Time-based split
    split_index = int(
        len(training_df) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training portion contains only one target class."
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    print("\nModel Evaluation")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to: {model_path}"
    )

    return model


def load_model(
    model_path: str = "models/trade_filter_model.pkl"
):
    """
    Load trained ML model.
    """

    return joblib.load(
        model_path
    )


def predict_trade(
    model,
    feature_row: pd.DataFrame,
    threshold: float = 0.60
):
    """
    Predict ACCEPT / AVOID.
    """

    probabilities = model.predict_proba(
        feature_row
    )[0]

    class_probabilities = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    success_probability = class_probabilities.get(
        1,
        0.0
    )

    decision = (
        "ACCEPT"
        if success_probability >= threshold
        else "AVOID"
    )

    return {
        "probability": float(
            success_probability
        ),
        "decision": decision
    }


def explain_trade(row, probability, decision):
    """
    Generate human-readable explanation
    for the ACCEPT / AVOID decision.
    """

    reasons = []

    # SMMA
    if row["smma_spread"] > 0:
        reasons.append(
            "SMMA 20 is above SMMA 120"
        )
    else:
        reasons.append(
            "SMMA 20 is below SMMA 120"
        )

    # LTQ
    if row["ltq_change"] > 0:
        reasons.append(
            "LTQ is increasing"
        )
    else:
        reasons.append(
            "LTQ is decreasing"
        )

    # Bid / Ask imbalance
    if row["bid_ask_imbalance"] > 0.10:
        reasons.append(
            "Strong bid-side support"
        )
    elif row["bid_ask_imbalance"] < -0.10:
        reasons.append(
            "Strong ask-side pressure"
        )
    else:
        reasons.append(
            "Bid/Ask balance is relatively neutral"
        )

    # Price movement
    if row["price_change"] > 0:
        reasons.append(
            "Price is moving upward"
        )
    elif row["price_change"] < 0:
        reasons.append(
            "Price is moving downward"
        )

    # Decision
    if decision == "ACCEPT":
        action_reason = (
            f"AI probability is "
            f"{probability * 100:.1f}%"
        )
    else:
        action_reason = (
            f"AI probability is only "
            f"{probability * 100:.1f}%"
        )

    return action_reason + ". " + "; ".join(
        reasons
    )