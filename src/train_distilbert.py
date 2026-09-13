"""
Fine-tune DistilBERT on the mental health status classification task.

WHY THIS SCRIPT EXISTS SEPARATELY:
This needs a real internet connection to download pretrained DistilBERT weights
from Hugging Face, and (ideally) a GPU to train in reasonable time. Run this on
Google Colab (free GPU: Runtime -> Change runtime type -> T4 GPU) or your own
machine with a GPU.

WHAT TO UPLOAD TO COLAB BEFORE RUNNING:
- train.csv
- test.csv
(the ones we built together - must have 'statement_light' and 'status' columns)

INSTALL FIRST (run this in a Colab cell):
!pip install transformers datasets scikit-learn torch accelerate -q
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
# Using statement_light here, not statement_heavy - DistilBERT was pretrained
# on natural, cased, punctuated text, so we keep that intact (unlike TF-IDF,
# which needed the heavily-cleaned version). See our earlier discussion on why.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = train_df.dropna(subset=['statement_light'])
test_df = test_df.dropna(subset=['statement_light'])

# ---------------------------------------------------------------
# 2. ENCODE LABELS
# ---------------------------------------------------------------
# DistilBERT needs integer labels (0-6), not text labels. LabelEncoder
# handles that mapping and we save it, since we need the SAME mapping
# later when interpreting predictions.
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['status'])
test_df['label'] = label_encoder.transform(test_df['status'])

label_names = label_encoder.classes_
print("Label mapping:", dict(zip(range(len(label_names)), label_names)))

# ---------------------------------------------------------------
# 3. CLASS WEIGHTS (same fix as Logistic Regression's class_weight='balanced')
# ---------------------------------------------------------------
# Transformers don't have a built-in class_weight parameter like sklearn does,
# so we compute weights manually and pass them into a weighted loss function
# in a custom Trainer below. This addresses the same Personality
# disorder / Stress under-representation problem we found in the baseline.
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", dict(zip(label_names, class_weights.tolist())))

# ---------------------------------------------------------------
# 4. TOKENIZE
# ---------------------------------------------------------------
# max_length=256: we saw in EDA that median length varies a lot by class
# (10 words for Normal, 137 for Personality disorder), and some posts run
# into the thousands of words. BERT-family models cap at 512 tokens max,
# but 256 keeps training fast while still covering the vast majority of
# posts without truncating away the important part. Longer posts get cut
# off at 256 tokens - worth knowing as a limitation, not hiding it.
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch['statement_light'], truncation=True, padding='max_length', max_length=256)

train_dataset = Dataset.from_pandas(train_df[['statement_light', 'label']])
test_dataset = Dataset.from_pandas(test_df[['statement_light', 'label']])

train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.remove_columns(['statement_light'])
test_dataset = test_dataset.remove_columns(['statement_light'])
train_dataset.set_format('torch')
test_dataset.set_format('torch')

# ---------------------------------------------------------------
# 5. MODEL
# ---------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names)
)

# ---------------------------------------------------------------
# 6. CUSTOM TRAINER WITH WEIGHTED LOSS
# ---------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': macro_f1}

# ---------------------------------------------------------------
# 7. TRAINING ARGUMENTS
# ---------------------------------------------------------------
# 2-3 epochs is usually enough for fine-tuning (unlike training from scratch) -
# the model already knows English, it just needs to adapt to this task.
# More epochs risks overfitting on a 40k-row dataset.
training_args = TrainingArguments(
    output_dir='./distilbert_mental_health',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    logging_steps=100,
    report_to='none',
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------
# 8. TRAIN
# ---------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------
# 9. FINAL EVALUATION - this is the first and only time test.csv is used
#    for scoring, matching the discipline we set up earlier
# ---------------------------------------------------------------
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_df['label'].values

report = classification_report(y_true, y_pred, target_names=label_names, digits=2)
print(report)

with open('distilbert_classification_report.txt', 'w') as f:
    f.write(report)

# Confusion matrix, same style as the Logistic Regression one for direct comparison
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (row-normalized) - DistilBERT fine-tuned')
plt.tight_layout()
plt.savefig('distilbert_confusion_matrix.png', dpi=150)

# Save the model + tokenizer for later use in the actual chatbot pipeline
trainer.save_model('./distilbert_mental_health_final')
tokenizer.save_pretrained('./distilbert_mental_health_final')

# Save label mapping - the chatbot code will need this to turn predictions back into readable labels
with open('label_mapping.json', 'w') as f:
    json.dump({str(i): label for i, label in enumerate(label_names)}, f)

print("\nDone. Download these files and bring them back:")
print("- distilbert_classification_report.txt")
print("- distilbert_confusion_matrix.png")
print("- label_mapping.json")