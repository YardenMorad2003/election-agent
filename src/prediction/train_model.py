import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.linear_model import Ridge


from build_dataset import build_dataset


def train():
    df = build_dataset()

    # Features (X) and target (y)
    X = df.drop(columns=["year", "dem_vote_share"])
    y = df["dem_vote_share"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Save model
    with open("src/prediction/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained successfully!")
    print("\nCoefficients:")
    for col, coef in zip(X.columns, model.coef_):
        print(f"{col}: {coef:.4f}")


if __name__ == "__main__":
    train()