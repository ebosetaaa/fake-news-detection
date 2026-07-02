from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "ml-service/models/hf-checkpoint/checkpoint-3189"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")
