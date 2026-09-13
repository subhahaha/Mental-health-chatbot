"""
Evaluate the classifier (and the full safety gate) against short,
chat-style messages - the kind real users actually type, as opposed to
the long-form posts in the original training/test data.

This measures the real-world gap we found by accident (the "want to
disappear" and "too much to do" examples) more systematically.

Run from inside src/: python eval_short_messages.py
"""

import csv
from classifier import classify_message
from safety_gate import evaluate_message

EVAL_PATH = '../data/short_message_eval.csv'

with open(EVAL_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} short test messages\n")

correct = 0
per_category_total = {}
per_category_correct = {}
crisis_route_for_suicidal = 0
suicidal_total = 0
mismatches = []

for row in rows:
    text = row['message']
    true_label = row['true_label']

    result = classify_message(text)
    gate_result = evaluate_message(text)

    predicted_label = result['label']
    is_correct = predicted_label == true_label

    per_category_total[true_label] = per_category_total.get(true_label, 0) + 1
    if is_correct:
        correct += 1
        per_category_correct[true_label] = per_category_correct.get(true_label, 0) + 1
    else:
        mismatches.append((text, true_label, predicted_label, result['confidence']))

    # Separately - for actual Suicidal messages, what matters most isn't
    # whether the LABEL was exactly right, it's whether the GATE correctly
    # routed it to "crisis". This is the number that actually matters for safety.
    if true_label == 'Suicidal':
        suicidal_total += 1
        if gate_result['route'] == 'crisis':
            crisis_route_for_suicidal += 1

print("=== Per-category accuracy (raw classifier label match) ===")
for category in per_category_total:
    acc = per_category_correct.get(category, 0) / per_category_total[category]
    print(f"  {category:22s} {acc:.0%}  ({per_category_correct.get(category, 0)}/{per_category_total[category]})")

print(f"\nOverall raw accuracy: {correct}/{len(rows)} ({correct/len(rows):.0%})")

print(f"\n=== The number that actually matters for safety ===")
print(f"Of {suicidal_total} genuinely Suicidal short messages, "
      f"{crisis_route_for_suicidal} correctly routed to 'crisis' by the FULL GATE "
      f"({crisis_route_for_suicidal/suicidal_total:.0%})")

print(f"\n=== Misclassifications (raw classifier, for reference) ===")
for text, true_label, predicted_label, confidence in mismatches:
    print(f'  "{text}"')
    print(f'    true: {true_label}  ->  predicted: {predicted_label} ({confidence:.0%})')