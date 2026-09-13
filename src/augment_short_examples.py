"""
Data augmentation to fix the classifier's biggest known weakness: it
learned to associate SHORT text with "Normal" purely because Normal's
training examples happened to be short (median 10 words) while every
other category was long (60-140+ words) - a shortcut, not real
understanding. See eval_pipeline.py's results for the proof.

The fix: extract individual SHORT sentences from the existing long
training posts for every non-Normal category, so short length stops
being a reliable signal for "Normal" specifically. This gives real,
in-domain short examples at scale, without writing hundreds by hand.

Run from inside src/: python augment_short_examples.py
Expects: mental_health_statements_cleaned_v1.csv (from data_cleaning.py)
Produces: augmented_train.csv (ready to swap into train_distilbert.py)
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split

CLEANED_PATH = '../data/mental_health_statements_cleaned_v1.csv'
OUTPUT_TRAIN = '../data/augmented_train.csv'
OUTPUT_TEST = '../data/augmented_test.csv'

# How many short excerpts to pull per category - roughly matching what
# we found useful in testing, enough to meaningfully shift the pattern
# without overwhelming the original long-form data entirely.
TARGET_SHORT_EXAMPLES_PER_CATEGORY = 2000

# A "short, chat-style" sentence: not a fragment, not still a paragraph.
MIN_WORDS = 3
MAX_WORDS = 20


def extract_short_sentences(text: str) -> list:
    """Split a long post into individual sentences, keep only the
    ones that read like a realistic short chat message."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    short_ones = []
    for s in sentences:
        s = s.strip()
        word_count = len(s.split())
        if MIN_WORDS <= word_count <= MAX_WORDS:
            short_ones.append(s)
    return short_ones


df = pd.read_csv(CLEANED_PATH)
df = df.dropna(subset=['statement_light'])

print("Original class balance:")
print(df['status'].value_counts())
print()

augmented_rows = []

for category in df['status'].unique():
    if category == 'Normal':
        continue  # Normal is already short - it's not the class that needs augmenting

    category_df = df[df['status'] == category]
    all_short_sentences = []

    for text in category_df['statement_light']:
        all_short_sentences.extend(extract_short_sentences(text))

    all_short_sentences = list(set(all_short_sentences))

    n_to_take = min(TARGET_SHORT_EXAMPLES_PER_CATEGORY, len(all_short_sentences))
    sampled = pd.Series(all_short_sentences).sample(n=n_to_take, random_state=42)

    print(f"{category}: extracted {len(all_short_sentences)} candidate short sentences, "
          f"using {n_to_take}")

    for sentence in sampled:
        augmented_rows.append({
            'statement': sentence,
            'status': category,
            'statement_light': sentence,
            'statement_heavy': sentence.lower(),
        })

augmented_df = pd.DataFrame(augmented_rows)
print(f"\nTotal short examples added: {len(augmented_df)}")

combined_df = pd.concat([df, augmented_df], ignore_index=True)

print("\nNew class balance (original + short augmented):")
print(combined_df['status'].value_counts())

train_df, test_df = train_test_split(
    combined_df, test_size=0.2, stratify=combined_df['status'], random_state=42
)

train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)
print(f"\nSaved {OUTPUT_TRAIN} ({train_df.shape[0]} rows)")
print(f"Saved {OUTPUT_TEST} ({test_df.shape[0]} rows)")