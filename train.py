import pandas as pd

from src.features import (
    create_features,
    create_trade_labels,
    get_ml_features,
)

from src.ml_model import train_model


# =========================================================
# 1. Load Market Data
# =========================================================

df = pd.read_csv(
    "data/sample_data.csv"
)


# =========================================================
# 2. Create Technical + Market Features
# =========================================================

df = create_features(df)


# =========================================================
# 3. Create Historical Trade Labels
# =========================================================

df = create_trade_labels(
    df,
    lookahead=10,
    target_pct=0.002
)


# =========================================================
# 4. Get ML Feature Columns
# =========================================================

feature_columns = get_ml_features()


# =========================================================
# 5. Display Dataset Information
# =========================================================

print("\nTraining Dataset")
print("=" * 60)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total ML features: {len(feature_columns)}"
)


print("\nFeature columns:")

for feature in feature_columns:
    print(f"- {feature}")


# =========================================================
# 6. Target Distribution
# =========================================================

print("\nTarget distribution:")

print(
    df["target"].value_counts()
)


# =========================================================
# 7. Display Crossover Signals
# =========================================================

signals = df[
    df["signal"] != "NONE"
]

print("\nCrossover Signals:")

if len(signals) > 0:

    print(
        signals[
            [
                "timestamp",
                "close",
                "signal",
                "future_return",
                "target",
            ]
        ]
    )

else:

    print("No crossover signals found.")


# =========================================================
# 8. Train ML Model
# =========================================================

print("\nStarting ML model training...")


try:

    model = train_model(
        df=df,
        feature_columns=feature_columns,
        model_path="models/trade_filter_model.pkl"
    )

    print(
        "\nML model training completed successfully!"
    )

    print(
        "Model saved at:"
    )

    print(
        "models/trade_filter_model.pkl"
    )


except ValueError as error:

    print(
        "\nML model was not trained."
    )

    print(
        f"Reason: {error}"
    )


# =========================================================
# 9. Final Status
# =========================================================

print("\nTraining pipeline completed.")