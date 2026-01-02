# app.py
import os
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# ---------- CONFIG ----------
MODELS_DIR = "models"
BERT_MODEL = "bert-base-uncased"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

app = FastAPI(title="Fake News Detection API")

# ---------- INPUT SCHEMA ----------
class PredictionInput(BaseModel):
    text: str
    shares: float = 0
    likes: float = 0
    comments: float = 0

# ---------- LOAD SCALER & ENCODER ----------
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# ---------- LOAD TOKENIZER ----------
tokenizer = BertTokenizer.from_pretrained(os.path.join(MODELS_DIR, "tokenizer"))

# ---------- MODEL DEFINITION ----------
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

# ---------- LOAD MODEL ----------
model = BertSocialClassifier().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "combined_model.pt"), map_location=DEVICE))
model.eval()

# ---------- PREDICTION FUNCTION ----------
def predict_with_probs(text, shares, likes, comments):
    # Tokenize text
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Scale social features
    social_scaled = scaler.transform(np.array([[shares, likes, comments]]).astype(float))
    social_tensor = torch.tensor(social_scaled, dtype=torch.float).to(DEVICE)

    # Run model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, social=social_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Map probabilities to class labels
    prob_dict = {label: float(prob) for label, prob in zip(label_encoder.classes_, probs)}

    return pred_label, prob_dict

# ---------- API ROUTES ----------
@app.post("/predict")
def make_prediction(payload: PredictionInput):
    pred_label, prob_dict = predict_with_probs(payload.text, payload.shares, payload.likes, payload.comments)
    return {
        "prediction": pred_label,
        "probs": prob_dict
    }

