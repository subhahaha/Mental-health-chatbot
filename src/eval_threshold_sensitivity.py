"""
Threshold sensitivity analysis for the safety gate.

safety_gate.py currently uses SUICIDAL_PROBABILITY_THRESHOLD = 0.30, chosen
by feel, never actually tested against alternatives. This script sweeps
that threshold across a range and measures two things that trade off
against each other:

- Recall: of messages that ARE genuinely Suicidal, what fraction get
  correctly routed to crisis? (want this HIGH - missing a real crisis
  is the worst failure mode)
- False positive rate: of messages that are NOT Suicidal, what fraction
  get incorrectly routed to crisis anyway? (want this reasonably LOW -
  too many false alarms make the safety net feel broken/annoying)

Note: the keyword check and "Suicidal" being the classifier's TOP label
both trigger crisis routing independent of this threshold - so this sweep
only affects the "moderate but not top-ranked Suicidal probability" cases.
That's expected; the threshold is a secondary safety net, not the only one.

Run from inside src/: python eval_threshold_sensitivity.py
Requires: data/pipeline_eval.csv (has true_label ground truth)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classifier import classify_message
from safety_check import check_crisis_keywords

EVAL_PATH = '../data/pipeline_eval.csv'
OUTPUT_PATH = '../results/threshold_sensitivity.png'

CURRENT_DEFAULT = 0.30
THRESHOLDS_TO_TEST = np.arange(0.05, 1.00, 0.05)

with open(EVAL_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} test messages\n")
print("Classifying all messages once (reused across every threshold)...")

# Classify every message ONCE and cache results - re-classifying inside
# the threshold loop would be wasteful and slow, since the classifier's
# output doesn't change based on the threshold we're testing.
cached_results = []
for row in rows:
    text = row['message']
    true_label = row['true_label']
    classifier_result = classify_message(text)
    keyword_flagged = check_crisis_keywords(text)
    cached_results.append({
        'true_label': true_label,
        'predicted_label': classifier_result['label'],
        'suicidal_probability': classifier_result['all_probabilities'].get('Suicidal', 0),
        'keyword_flagged': keyword_flagged,
    })

print("Done.\n")

recalls = []
false_positive_rates = []

suicidal_rows = [r for r in cached_results if r['true_label'] == 'Suicidal']
non_suicidal_rows = [r for r in cached_results if r['true_label'] != 'Suicidal']

for threshold in THRESHOLDS_TO_TEST:
    def is_crisis(r, t=threshold):
        return (
            r['keyword_flagged']
            or r['predicted_label'] == 'Suicidal'
            or r['suicidal_probability'] >= t
        )

    true_positives = sum(1 for r in suicidal_rows if is_crisis(r))
    recall = true_positives / len(suicidal_rows) if suicidal_rows else 0

    false_positives = sum(1 for r in non_suicidal_rows if is_crisis(r))
    fpr = false_positives / len(non_suicidal_rows) if non_suicidal_rows else 0

    recalls.append(recall)
    false_positive_rates.append(fpr)

print("Threshold | Recall (catch real crises) | False Positive Rate")
for t, r, f in zip(THRESHOLDS_TO_TEST, recalls, false_positive_rates):
    marker = " <-- current default" if abs(t - CURRENT_DEFAULT) < 0.025 else ""
    print(f"  {t:.2f}    |  {r:.0%}                       |  {f:.0%}{marker}")

plt.figure(figsize=(9, 6))
plt.plot(THRESHOLDS_TO_TEST, recalls, 'o-', label='Recall (catches real crises)', color='#5B8A80')
plt.plot(THRESHOLDS_TO_TEST, false_positive_rates, 'o-', label='False positive rate', color='#D9A05B')
plt.axvline(x=CURRENT_DEFAULT, color='gray', linestyle='--', alpha=0.6, label=f'Current default ({CURRENT_DEFAULT})')
plt.xlabel('Suicidal probability threshold')
plt.ylabel('Rate')
plt.title('Safety Gate Threshold Sensitivity')
plt.legend()
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nSaved: {OUTPUT_PATH}")