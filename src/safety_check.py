"""
Rule-based safety layer - independent of the ML classifier.

WHY THIS EXISTS: we found that DistilBERT, despite 83% test accuracy,
misclassifies short, casual, conversational messages (like real chat input)
because it was trained on long-form Reddit/Twitter-style posts. A phrase
like "I just want to disappear" got classified as "Normal" with 80%
confidence - which would have completely bypassed the safety override.

This module does NOT replace the classifier. It's a second, simpler,
much more literal check that runs independently. If EITHER this keyword
check OR the classifier flags crisis risk, we treat it as a crisis.
We deliberately err toward false positives (flagging something that
turns out fine) over false negatives (missing something real) - a false
positive just means someone sees a helpline message they didn't strictly
need; a false negative could mean someone in real crisis gets a chatty
AI response instead.

This is intentionally simple pattern matching, not a modelnot trying to
be clever - simplicity here is a feature, not a limitation. It's easy to
audit, easy to extend, and doesn't depend on the ML model's blind spots.
"""

import re

# Phrases and patterns associated with suicidal ideation or crisis risk.
# This list is NOT exhaustive - it should grow over time as you find gaps
# (like the "want to disappear" example we just found). Keep entries as
# lowercase, and prefer short, high-signal phrases over single common
# words (a word like "die" alone is too broad - "hate 2 die laughing"
# shouldn't trigger this, but "want to die" should).
CRISIS_PATTERNS = [
    r"\bwant(ed)? to die\b",
    r"\bwant(ed)? to disappear\b",
    r"\bdon'?t want to (be here|exist|live)\b",
    r"\bno (reason|point) (to|in) (live|living|continuing|going on)\b",
    r"\bdon'?t see (the |any )?point\b",
    r"\bbetter off without me\b",
    r"\bkill myself\b",
    r"\bend(ing)? (it all|my life|things|everything)\b",
    r"\bthinking about ending\b",
    r"\bsuicid(e|al)\b",
    r"\bcan'?t (do this|go on) anymore\b",
    r"\bnot worth living\b",
    r"\bhurt(ing)? myself\b",
    r"\bself[\s-]?harm\b",
    r"\bwish i (was|were) dead\b",
    r"\bgive up on (life|everything)\b",
    r"\bwant it (all )?to stop\b",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]


def check_crisis_keywords(text: str) -> bool:
    """
    Returns True if the message matches any known crisis-risk pattern.
    Simple regex matching - deliberately literal and easy to audit rather
    than 'smart', since predictability matters more than sophistication
    for a safety check like this.
    """
    return any(pattern.search(text) for pattern in _compiled_patterns)


if __name__ == '__main__':
    test_messages = [
        "I don't want to talk to anyone lately, I just want to disappear",
        "my heart is racing and I can't stop worrying",
        "I have way too much to do and no idea where to even start",
        "I hate this traffic, I could just die of boredom",  # should NOT trigger - common phrase, not literal
        "I don't want to be here anymore",
    ]

    for msg in test_messages:
        flagged = check_crisis_keywords(msg)
        print(f'"{msg}"')
        print(f'  -> Crisis keyword match: {flagged}\n')