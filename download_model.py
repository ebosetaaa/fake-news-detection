from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "dhruvpal/fake-news-bert"  # public fake-news model

# create folders
os.makedirs("ml-service/models/hf-checkpoint/checkpoint-3189", exist_ok=True)

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("ml-service/models/hf-checkpoint/checkpoint-3189")

print("Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained("ml-service/models/hf-checkpoint/checkpoint-3189")

print("✅ Done! Files are in ml-service/models/hf-checkpoint/checkpoint-3189")
