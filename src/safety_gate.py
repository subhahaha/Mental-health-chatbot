"""
Combined safety gate - the actual decision point in the pipeline.
See eval_threshold_sensitivity.py for the evidence behind these thresholds.
"""

from classifier import classify_message
from safety_check import check_crisis_keywords

# SUICIDAL_PROBABILITY_THRESHOLD updated 0.30 -> 0.45 based on
# eval_threshold_sensitivity.py: recall stayed at 100% across the entire
# tested range (0.05-0.95) in our eval set, while false positive rate
# dropped from 10% to 3% once the threshold reached ~0.40. Raising it
# reduces false alarms with no measured recall cost.
#
# Caveat: our eval set never isolated a case where THIS specific rule was
# the only thing catching a crisis (keyword/top-label caught all 5 test
# cases) - so this tunes false-positive rate with confidence, but doesn't
# fully validate this rule's recall contribution in isolation. Worth
# adding a targeted test case for that scenario specifically.
SUICIDAL_PROBABILITY_THRESHOLD = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.40


def evaluate_message(text: str) -> dict:
    classifier_result = classify_message(text)
    keyword_flagged = check_crisis_keywords(text)

    suicidal_probability = classifier_result["all_probabilities"].get("Suicidal", 0)

    is_crisis = (
        keyword_flagged
        or classifier_result["label"] == "Suicidal"
        or suicidal_probability >= SUICIDAL_PROBABILITY_THRESHOLD
    )

    if is_crisis:
        route = "crisis"
    elif classifier_result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        route = "low_confidence"
    else:
        route = "normal"

    return {
        "route": route,
        "label": classifier_result["label"],
        "confidence": classifier_result["confidence"],
        "keyword_flagged": keyword_flagged,
        "suicidal_probability": suicidal_probability,
    }


if __name__ == '__main__':
    test_messages = [
        "my heart is racing and I can't stop worrying something bad will happen",
        "I have way too much to do and no idea where to even start",
        "I don't want to talk to anyone lately, I just want to disappear",
        "what's the weather like today",
    ]

    for msg in test_messages:
        result = evaluate_message(msg)
        print(f'"{msg}"')
        print(f'  -> route: {result["route"]}')
        print(f'  -> classifier label: {result["label"]} (confidence: {result["confidence"]:.2%})')
        print(f'  -> keyword flagged: {result["keyword_flagged"]}, suicidal_probability: {result["suicidal_probability"]:.2%}')
        print()