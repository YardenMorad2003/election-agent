from langchain_core.tools import tool
from src.prediction.predict import predict_next

@tool
def predict_vote_share(_: str) -> str:
    """Predict Democratic vote share for the next U.S. presidential election."""
    pred = predict_next()
    return f"Predicted Democratic vote share: {pred:.3f} ({pred*100:.1f}%)"