"""
Classifier step of the chatbot pipeline.

Loads the fine-tuned DistilBERT model and provides a function that takes
a raw user message and returns the predicted label, confidence, and full
probability distribution across all 7 labels.

Run this file directly to test it against a few example messages:
    python classifier.py
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = '../models/distilbert_mental_health_final'
LABEL_MAP_PATH = '../models/distilbert_mental_health_final/label_mapping.json'

print("Loading classifier model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(LABEL_MAP_PATH) as f:
    label_mapping = json.load(f)

print("Classifier ready.\n")


def classify_message(text: str) -> dict:
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors='pt')

    # DistilBERT doesn't use token_type_ids (that's a BERT sentence-pair
    # input) - some tokenizer/library version combos generate it anyway,
    # which the model's forward() then rejects. Strip it out defensively.
    inputs.pop('token_type_ids', None)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]

    all_probabilities = {
        label_mapping[str(i)]: round(prob.item(), 4)
        for i, prob in enumerate(probabilities)
    }

    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = label_mapping[str(predicted_idx)]
    confidence = all_probabilities[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "all_probabilities": all_probabilities
    }


if __name__ == '__main__':
    test_messages = [
        "my heart is racing and I can't stop worrying something bad will happen",
        "I have way too much to do and no idea where to even start",
        "I don't want to talk to anyone lately, I just want to disappear",
        "what's the weather like today",
    ]

    for msg in test_messages:
        result = classify_message(msg)
        print(f'Message: "{msg}"')
        print(f'  -> Predicted: {result["label"]} (confidence: {result["confidence"]:.2%})')
        top3 = sorted(result["all_probabilities"].items(), key=lambda x: -x[1])[:3]
        print(f'  Top 3: {top3}')
        print()