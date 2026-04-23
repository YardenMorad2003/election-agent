import pickle
import pandas as pd

from build_dataset import build_dataset


def load_model():
    with open("src/prediction/model.pkl", "rb") as f:
        return pickle.load(f)


def predict_next():
    model = load_model()
    df = build_dataset()

    # Use most recent year as proxy for next prediction
    latest = df.sort_values("year").iloc[-1:]

    X_latest = latest.drop(columns=["year", "dem_vote_share"])

    prediction = model.predict(X_latest)[0]

    return prediction


if __name__ == "__main__":
    pred = predict_next()
    print(f"Predicted Democratic vote share (next election): {pred:.3f}")