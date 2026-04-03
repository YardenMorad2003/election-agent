"""
Fine-tune DistilBERT for question routing (Module 7).

Trains a 3-class classifier on benchmark questions to route user queries
to the correct tool: data_query (0), coalition_calculator (1), context_search (2).

Handles extreme class imbalance (62/6/2) via:
  - Synonym-based data augmentation for minority classes
  - Weighted cross-entropy loss
  - Stratified train/val split

Usage: python train_classifier.py
Output: models/distilbert-router/
"""
import json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Label mapping (matches classifiers.py ID_TO_TOOL) ──
TOOL_TO_ID = {"data_query": 0, "coalition_calculator": 1, "rag_retrieval": 2}
ID_TO_TOOL = {v: k for k, v in TOOL_TO_ID.items()}

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "distilbert-router")


# ── Data augmentation for minority classes ──
def augment_question(question: str) -> list[str]:
    """Generate paraphrases via synonym substitution."""
    variants = []

    swaps = [
        ("form a government", "form a ruling coalition"),
        ("form a government", "create a majority government"),
        ("form a coalition", "build a coalition"),
        ("form a coalition", "assemble a coalition"),
        ("coalition", "governing coalition"),
        ("possible", "feasible"),
        ("reaching", "that reach"),
        ("reaching", "achieving"),
        ("without", "excluding"),
        ("smallest", "minimum-size"),
        ("What are all", "List all"),
        ("What are all", "Show all"),
        ("How many", "What number of"),
        ("Can ", "Is it possible for "),
        ("What are the", "Describe the"),
        ("What are the", "Explain the"),
        ("What datasets", "Which datasets"),
        ("available in this system", "included in the database"),
        ("available in this system", "covered by this tool"),
        ("urban-rural", "metro-nonmetro"),
        ("classification codes", "category codes"),
        ("right bloc", "right-wing bloc"),
        ("center-left", "center-left-arab"),
        ("3-party", "three-party"),
        ("right-bloc coalitions", "coalitions from the right bloc"),
    ]

    for old, new in swaps:
        if old.lower() in question.lower():
            idx = question.lower().find(old.lower())
            variant = question[:idx] + new + question[idx + len(old):]
            variants.append(variant)

    return variants


def load_and_augment_data():
    """Load benchmark questions and augment minority classes."""
    questions_path = os.path.join(os.path.dirname(__file__), "benchmark", "questions.json")
    with open(questions_path) as f:
        questions = json.load(f)

    texts, labels = [], []
    for q in questions:
        tool = q["expected_tool"]
        label = TOOL_TO_ID[tool]
        texts.append(q["question"])
        labels.append(label)

        # Augment minority classes
        if tool in ("coalition_calculator", "rag_retrieval"):
            for variant in augment_question(q["question"]):
                texts.append(variant)
                labels.append(label)

    print(f"Dataset: {len(texts)} samples after augmentation")
    for tool, tid in TOOL_TO_ID.items():
        count = labels.count(tid)
        print(f"  {tool}: {count}")

    return texts, labels


class QuestionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_class_weights(labels):
    """Inverse frequency weighting."""
    counts = np.bincount(labels, minlength=3)
    # Avoid division by zero for classes with no samples
    counts = np.maximum(counts, 1)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    texts, labels = load_and_augment_data()

    # Stratified split — use larger val set since dataset is small
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    train_dataset = QuestionDataset(train_texts, train_labels, tokenizer)
    val_dataset = QuestionDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Weighted loss for class imbalance
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 20
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["labels"].tolist())

        val_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {avg_loss:.4f}, val_acc: {val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # ── Final report ──
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print("\nClassification report (final epoch):")
    target_names = [ID_TO_TOOL[i] for i in range(3)]
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
