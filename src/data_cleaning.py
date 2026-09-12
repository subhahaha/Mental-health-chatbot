"""
Data cleaning pipeline for the mental health status dataset.

Takes the raw Kaggle CSV and produces:
- A cleaned dataset with two text versions (light-cleaned for transformers,
  heavy-cleaned for TF-IDF/classical models)
- A stratified train/test split

Run: python data_cleaning.py
Expects: data/CombinedData.csv (the raw Kaggle download) in the same folder,
         or edit RAW_PATH below.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split

RAW_PATH = '..\dataset\CombinedData.csv'
OUTPUT_CLEANED = 'mental_health_statements_cleaned_v1.csv'
OUTPUT_TRAIN = 'train.csv'
OUTPUT_TEST = 'test.csv'

# Common English stopwords - used to flag likely gibberish/spam rows.
# Real sentences have plenty of these mixed in; keyword-spam or non-English
# text usually doesn't.
STOPWORDS = {
    'the', 'is', 'i', 'my', 'a', 'to', 'and', 'of', 'in', 'feel', 'it', 'you',
    'me', 'was', 'for', 'that', 'have', 'be', 'not', 'with', 'on', 'this',
    'but', 'so', 'am', 'as', 'are', 'just', 'like', 'how', 'do', 'what',
    'can', 'no', 'if', 'all', 'been', 'has', 'get', 'know', 'out'
}


def stopword_ratio(text: str) -> float:
    """Fraction of words in `text` that are common English stopwords.
    Low ratio on a long piece of text usually means gibberish, spam,
    or non-English content that slipped into the scrape."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in STOPWORDS)
    return hits / len(words)


def is_likely_junk(text: str, word_count: int, ratio_threshold: float = 0.05,
                    min_words: int = 15) -> bool:
    """Flags long text with an unusually low stopword ratio - catches
    ads, non-English posts, and garbled unicode spam without needing
    a language-detection library."""
    return word_count >= min_words and stopword_ratio(text) < ratio_threshold


def light_clean(text: str) -> str:
    """Minimal cleaning for transformer models - strip URLs and normalize
    whitespace only. Keeps casing and punctuation intact since transformers
    were pretrained on natural text and can use that signal (e.g. '!!' as
    an intensity cue)."""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def heavy_clean(text: str) -> str:
    """Aggressive cleaning for TF-IDF/classical models - lowercase, strip
    HTML, punctuation, and numbers. TF-IDF treats 'Feel' and 'feel' as
    different tokens unless normalized, wasting vocabulary on formatting
    rather than meaning."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataset(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    before = df.shape[0]
    df = df.dropna(subset=['statement'])
    print(f'Dropped {before - df.shape[0]} rows with missing statements')

    before = df.shape[0]
    df = df.drop_duplicates(subset=['statement', 'status'])
    print(f'Dropped {before - df.shape[0]} duplicate rows')

    df['word_count'] = df['statement'].str.split().str.len()
    junk_mask = df.apply(
        lambda row: is_likely_junk(row['statement'], row['word_count']), axis=1
    )
    print(f'Dropped {junk_mask.sum()} likely junk/gibberish/spam rows')
    df = df[~junk_mask].drop(columns=['word_count'])

    df['statement_light'] = df['statement'].apply(light_clean)
    df['statement_heavy'] = df['statement'].apply(heavy_clean)

    before = df.shape[0]
    df = df[df['statement_heavy'].str.len() > 0]
    print(f'Dropped {before - df.shape[0]} rows that became empty after heavy cleaning')

    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split - preserves each class's proportion in both train
    and test. Necessary here because the smallest class (Personality
    disorder) has under 1,000 examples; a random split risks leaving
    too few in test to evaluate reliably."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['status'], random_state=seed
    )
    return train_df, test_df


if __name__ == '__main__':
    print('=== Cleaning ===')
    df = clean_dataset(RAW_PATH)
    df.to_csv(OUTPUT_CLEANED, index=False)
    print(f'Saved {OUTPUT_CLEANED} ({df.shape[0]} rows)\n')

    print('=== Splitting ===')
    train_df, test_df = split_dataset(df)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)
    print(f'Saved {OUTPUT_TRAIN} ({train_df.shape[0]} rows)')
    print(f'Saved {OUTPUT_TEST} ({test_df.shape[0]} rows)')