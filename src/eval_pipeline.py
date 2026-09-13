"""
Full pipeline evaluation - goes beyond classifier-only accuracy to measure
the RAG retrieval step in isolation, so we know exactly which part of the
pipeline to blame for any given failure.

Three separate numbers:
1. Classifier accuracy - does the classifier pick the right category?
2. Retrieval accuracy GIVEN the correct category - isolates RAG quality
   from classifier quality, by searching using the TRUE label.
3. End-to-end accuracy - using the classifier's ACTUAL predicted label.

Run from inside src/: python eval_pipeline.py
"""

import os
import csv
import json
import faiss
from sentence_transformers import SentenceTransformer
from classifier import classify_message
from safety_gate import evaluate_message

EVAL_PATH = '../data/pipeline_eval.csv'
INDEX_DIR = '../knowledge_base_index'

print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.\n")


def retrieve_document(category: str, message: str):
    safe_name = category.lower().replace(' ', '_')
    index_path = f'{INDEX_DIR}/{safe_name}.index'
    metadata_path = f'{INDEX_DIR}/{safe_name}_metadata.json'
    if not os.path.exists(index_path):
        return None
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    query_vec = embed_model.encode([message], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    _, indices = index.search(query_vec, 1)
    return metadata[indices[0][0]]


with open(EVAL_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} test messages\n")

classifier_correct = 0
retrieval_given_true_label_correct = 0
end_to_end_correct = 0
retrieval_applicable_count = 0

suicidal_total = 0
suicidal_caught = 0

for row in rows:
    text = row['message']
    true_label = row['true_label']
    expected_doc = row['expected_document_id'] or None

    predicted = classify_message(text)['label']
    gate = evaluate_message(text)

    if predicted == true_label:
        classifier_correct += 1

    if true_label == 'Suicidal':
        suicidal_total += 1
        if gate['route'] == 'crisis':
            suicidal_caught += 1

    if expected_doc:
        retrieval_applicable_count += 1

        true_label_doc = retrieve_document(true_label, text)
        if true_label_doc and true_label_doc['id'] == expected_doc:
            retrieval_given_true_label_correct += 1

        if gate['route'] == 'normal' and predicted not in ('Normal', 'Suicidal'):
            predicted_label_doc = retrieve_document(predicted, text)
            if predicted_label_doc and predicted_label_doc['id'] == expected_doc:
                end_to_end_correct += 1

print("=== Results ===\n")
print(f"1. Classifier accuracy (raw label match): "
      f"{classifier_correct}/{len(rows)} ({classifier_correct/len(rows):.0%})")

print(f"\n2. Retrieval accuracy GIVEN correct category (isolates RAG quality): "
      f"{retrieval_given_true_label_correct}/{retrieval_applicable_count} "
      f"({retrieval_given_true_label_correct/retrieval_applicable_count:.0%})")

print(f"\n3. End-to-end accuracy (realistic - classifier's own label used): "
      f"{end_to_end_correct}/{retrieval_applicable_count} "
      f"({end_to_end_correct/retrieval_applicable_count:.0%})")

print(f"\n4. Safety-critical: Suicidal messages correctly routed to crisis: "
      f"{suicidal_caught}/{suicidal_total} ({suicidal_caught/suicidal_total:.0%})")

print("\n=== What this tells us ===")
print("If (2) is high and (3) is low, retrieval itself works well - the")
print("bottleneck is the classifier's category prediction, not RAG.")