"""
BERT Training — Scoliosis Multimodal Project
=============================================
Trains BERT on clinical notes from dataset_metadata.csv
to predict scoliosis severity: mild / moderate / severe.

Fits directly into the late-fusion pipeline alongside ResNet50.

Requirements:
    pip install transformers torch scikit-learn pandas tqdm

Expected file from generate_metadata.py:
    dataset_metadata.csv  →  columns: image_path, label, clinical_note
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG — matches your project structure exactly
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data ─────────────────────────────────────────────────
    "metadata_csv":   "dataset_metadata.csv",   # from generate_metadata.py
    "text_column":    "clinical_note",
    "label_column":   "label",
    "split_column":   "image_path",             # used to filter train/val/test rows

    # ── Classes (must match generate_metadata.py) ────────────
    "label_names":    ["mild", "moderate", "severe"],
    "num_labels":     3,

    # ── Model ────────────────────────────────────────────────
    "model_name":     "bert-base-uncased",
    "max_length":     64,                       # clinical notes are short; 128 is enough

    # ── Training ─────────────────────────────────────────────
    "batch_size":     4,
    "epochs":         3,
    "learning_rate":  2e-5,
    "accumulation_steps": 2,
    "warmup_ratio":   0.1,

    # ── Output ───────────────────────────────────────────────
    "output_dir":     "./bert_scoliosis_model",  # saved model for fusion

    # ── Device ───────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Maps string labels → integers
LABEL2ID = {label: idx for idx, label in enumerate(CONFIG["label_names"])}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class ScoliosisNotesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# Reads dataset_metadata.csv and splits by train/val/test
# using the image_path prefix (e.g. "train/mild/img_1.jpg")
# ─────────────────────────────────────────────────────────────
def load_split(df, split_name, config):
    """Filter rows belonging to a split using the image_path prefix."""
    mask = df[config["split_column"]].str.startswith(split_name + "/")
    subset = df[mask].copy()
    texts = subset[config["text_column"]].astype(str).tolist()
    labels = subset[config["label_column"]].map(LABEL2ID).tolist()
    return texts, labels


def load_all_splits(config):
    print(f"\n📂  Loading metadata: {config['metadata_csv']}")
    df = pd.read_csv(config["metadata_csv"])
    df = df.dropna(subset=[config["text_column"], config["label_column"]])

    print(f"    Total rows : {len(df)}")
    print(f"    Label dist : {df[config['label_column']].value_counts().to_dict()}")

    train_texts, train_labels = load_split(df, "train", config)
    val_texts,   val_labels   = load_split(df, "val",   config)
    test_texts,  test_labels  = load_split(df, "test",  config)

    print(f"\n    Train : {len(train_texts)} samples")
    print(f"    Val   : {len(val_texts)} samples")
    print(f"    Test  : {len(test_texts)} samples")

    return (
        train_texts, train_labels,
        val_texts,   val_labels,
        test_texts,  test_labels,
    )


# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, config):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss = loss / config["accumulation_steps"]
        loss.backward()

        if (step + 1) % config["accumulation_steps"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += out.loss.item()
            all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — for late fusion with ResNet50
# Returns [CLS] embeddings: shape (N, 768)
# Call this after training to get text vectors for fusion layer
# ─────────────────────────────────────────────────────────────
def extract_bert_features(texts, model, tokenizer, config):
    """
    Extract BERT [CLS] token embeddings for multimodal fusion.

    Usage in fusion pipeline:
        text_features = extract_bert_features(notes, model, tokenizer, CONFIG)
        # shape: (N, 768)
        # concatenate with ResNet50 features → pass to fusion classifier
    """
    model.eval()
    dataset = ScoliosisNotesDataset(texts, [0] * len(texts), tokenizer, config["max_length"])
    loader  = DataLoader(dataset, batch_size=config["batch_size"])
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting BERT features"):
            out = model.bert(
                input_ids      = batch["input_ids"].to(config["device"]),
                attention_mask = batch["attention_mask"].to(config["device"]),
                token_type_ids = batch["token_type_ids"].to(config["device"]),
            )
            cls = out.last_hidden_state[:, 0, :]   # [CLS] token → (batch, 768)
            embeddings.append(cls.cpu().numpy())

    return np.vstack(embeddings)   # (N, 768)


# ─────────────────────────────────────────────────────────────
# INFERENCE — predict severity from a list of notes
# ─────────────────────────────────────────────────────────────
def predict(texts, model, tokenizer, config):
    """
    Predict scoliosis severity from raw clinical note strings.
    Returns (label_ids, label_names)
    """
    model.eval()
    dataset = ScoliosisNotesDataset(texts, [0] * len(texts), tokenizer, config["max_length"])
    loader  = DataLoader(dataset, batch_size=config["batch_size"])
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids      = batch["input_ids"].to(config["device"]),
                attention_mask = batch["attention_mask"].to(config["device"]),
                token_type_ids = batch["token_type_ids"].to(config["device"]),
            )
            all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())

    return all_preds, [ID2LABEL[p] for p in all_preds]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    config = CONFIG
    torch.manual_seed(42)
    print(f"\n🖥️   Device : {config['device']}")

    # ── 1. Load data ──────────────────────────────────────────
    (train_texts, train_labels,
     val_texts,   val_labels,
     test_texts,  test_labels) = load_all_splits(config)

    # ── 2. Tokenizer & model ──────────────────────────────────
    print(f"\n🤖  Loading {config['model_name']} ...")
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    model = BertForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(config["device"])

    # ── 3. DataLoaders ────────────────────────────────────────
    train_ds = ScoliosisNotesDataset(train_texts, train_labels, tokenizer, config["max_length"])
    val_ds   = ScoliosisNotesDataset(val_texts,   val_labels,   tokenizer, config["max_length"])
    test_ds  = ScoliosisNotesDataset(test_texts,  test_labels,  tokenizer, config["max_length"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"])

    # ── 4. Optimizer & scheduler ──────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── 5. Training ───────────────────────────────────────────
    print(f"\n🚀  Training for {config['epochs']} epochs...\n")
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        print(f"── Epoch {epoch}/{config['epochs']} " + "─" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config["device"], config
        )
        val_loss, val_preds, val_true = evaluate(model, val_loader, config["device"])
        val_acc = (val_preds == val_true).mean()

        print(f"   Train  →  loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"   Val    →  loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config["output_dir"], exist_ok=True)
            model.save_pretrained(config["output_dir"])
            tokenizer.save_pretrained(config["output_dir"])
            print(f"   ✅  Best model saved → {config['output_dir']}")

    # ── 6. Final test-set evaluation ──────────────────────────
    print("\n📊  Test Set Evaluation (best checkpoint):")
    model = BertForSequenceClassification.from_pretrained(
        config["output_dir"]
    ).to(config["device"])
    _, test_preds, test_true = evaluate(model, test_loader, config["device"])

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=config["label_names"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(test_true, test_preds)
    print(pd.DataFrame(cm, index=config["label_names"], columns=config["label_names"]))

    # ── 7. Save BERT features for fusion ──────────────────────
    print("\n💾  Saving BERT feature embeddings for late fusion ...")
    os.makedirs(config["output_dir"], exist_ok=True)

    for split_name, texts, labels in [
        ("train", train_texts, train_labels),
        ("val",   val_texts,   val_labels),
        ("test",  test_texts,  test_labels),
    ]:
        feats = extract_bert_features(texts, model, tokenizer, config)
        np.save(os.path.join(config["output_dir"], f"bert_features_{split_name}.npy"), feats)
        np.save(os.path.join(config["output_dir"], f"bert_labels_{split_name}.npy"), np.array(labels))
        print(f"   {split_name}: {feats.shape} → bert_features_{split_name}.npy")

    print("\n✅  Done.")
    print(f"   Model     : {config['output_dir']}/")
    print(f"   Features  : bert_features_{{train,val,test}}.npy  (shape: N × 768)")
    print(f"   Labels    : bert_labels_{{train,val,test}}.npy")
    print(f"\n   Next step : load these .npy files in your fusion script,")
    print(f"               concatenate with ResNet50 features, train classifier.")


if __name__ == "__main__":
    main()