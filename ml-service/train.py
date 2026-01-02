# train.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------- CONFIG ----------
DATA_CSV = "datasets/WELFake_Dataset.csv"
MODELS_DIR = "models"
BERT_MODEL = "bert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 4         # Lower for CPU
LR = 2e-5
MAX_LEN = 256
SEED = 42
# ----------------------------

os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- 1) LOAD CSV ----------
print("Loading CSV:", DATA_CSV)
df = pd.read_csv(DATA_CSV)

# Sanity checks
if "text" not in df.columns:
    raise RuntimeError("CSV must contain a 'text' column.")
if "label" not in df.columns:
    raise RuntimeError("CSV must contain a 'label' column (FAKE/REAL or 0/1).")

df = df.rename(columns={c: c.strip() for c in df.columns})

# Handle social columns
for col in ("shares", "likes", "comments"):
    if col not in df.columns:
        print(f"Column '{col}' missing — synthesizing proxy from text length.")
        df[col] = df["text"].apply(lambda t: int((len(t.split()) % 50) * (1 + (len(t) % 7)/10)))

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(df["label"].astype(str))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
print("Saved label encoder ->", os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Scale social features
social_cols = ["shares", "likes", "comments"]
social = df[social_cols].astype(float).values
scaler = StandardScaler()
social_scaled = scaler.fit_transform(social)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Saved scaler ->", os.path.join(MODELS_DIR, "scaler.pkl"))

# ---------- 2) TOKENIZER ----------
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# ---------- 3) DATASET ----------
class FakeNewsDataset(Dataset):
    def __init__(self, texts, social_features, labels, tokenizer, max_len=256):
        self.texts = texts
        self.social = social_features.astype(np.float32)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        enc = self.tokenizer(txt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "social": torch.tensor(self.social[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Train/val split
texts = df["text"].values
X_train_text, X_val_text, X_train_social, X_val_social, y_train, y_val = train_test_split(
    texts, social_scaled, labels, test_size=0.15, random_state=SEED, stratify=labels
)

train_dataset = FakeNewsDataset(X_train_text, X_train_social, y_train, tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(X_val_text, X_val_social, y_val, tokenizer, MAX_LEN)

# ---------- 4) HYBRID MODEL ----------
class BertSocialClassifier(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL, social_dim=3, social_hidden=32, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        self.social_mlp = nn.Sequential(
            nn.Linear(social_dim, social_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(social_hidden, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, social=None, labels=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = bert_out.pooler_output
        social_emb = self.social_mlp(social)
        combined = torch.cat([cls, social_emb], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

model = BertSocialClassifier().to(device)

# ---------- 5) TRAINER ----------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0)
    }

training_args = TrainingArguments(
    output_dir=os.path.join(MODELS_DIR, "hf-checkpoint"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_dir=os.path.join(MODELS_DIR, "logs"),
    logging_steps=50,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------- 6) TRAIN ----------
print("Starting training...")
trainer.train()

# Save model and tokenizer
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "combined_model.pt"))
tokenizer.save_pretrained(os.path.join(MODELS_DIR, "tokenizer"))
print("Saved model and tokenizer.")

print("Training complete.")
