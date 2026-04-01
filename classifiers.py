"""
Question classifiers for Config 3 (Fixed Routing).

Module 3: Zero-shot classification using BART-MNLI (facebook/bart-large-mnli)
Module 7: Fine-tuned DistilBERT classifier (trained on benchmark questions)
"""
import os
from transformers import pipeline

# ── Zero-Shot Classifier (BART-MNLI) ──
_zero_shot = None

# Descriptive labels for zero-shot classification — BART-MNLI checks
# entailment between the input question and each label as a hypothesis.
_CANDIDATE_LABELS = [
    "looking up election results, vote counts, percentages, party performance, or turnout statistics",
    "calculating which party combinations can form a coalition government with a majority of seats",
    "background context about election systems, definitions, and data coverage",
]
_LABEL_TO_TOOL = {
    _CANDIDATE_LABELS[0]: "data_query",
    _CANDIDATE_LABELS[1]: "coalition",
    _CANDIDATE_LABELS[2]: "context_search",
}


def _get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        _zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _zero_shot


def classify_question_zeroshot(question: str) -> str:
    """Classify a question to a tool using zero-shot NLI (BART-MNLI).

    The model checks entailment between the question and each candidate label,
    returning the label with the highest entailment score.
    """
    classifier = _get_zero_shot()
    result = classifier(question, _CANDIDATE_LABELS)
    return _LABEL_TO_TOOL[result["labels"][0]]


# ── Fine-Tuned DistilBERT Classifier (added after training) ──
_distilbert_model = None
_distilbert_tokenizer = None

ID_TO_TOOL = {0: "data_query", 1: "coalition", 2: "context_search"}


def classify_question_finetuned(question: str) -> str:
    """Classify a question using a fine-tuned DistilBERT model.

    The model is trained on labeled benchmark questions and saved to
    models/distilbert-router/. Returns one of: data_query, coalition, context_search.
    """
    global _distilbert_model, _distilbert_tokenizer
    import torch

    if _distilbert_model is None:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        model_path = os.path.join(os.path.dirname(__file__), "models", "distilbert-router")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. "
                "Run 'python train_classifier.py' first."
            )
        _distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        _distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        _distilbert_model.eval()

    inputs = _distilbert_tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = _distilbert_model(**inputs).logits
    predicted = torch.argmax(logits, dim=-1).item()
    return ID_TO_TOOL[predicted]
