"""
Train and evaluate the classical ML baseline (TF-IDF + Logistic Regression)
for mental health status classification.

Also runs SVM and Naive Bayes for comparison, since documenting *why*
Logistic Regression was chosen over alternatives is part of the point.

Run: python train_baseline.py
Expects: train.csv, test.csv (from data_cleaning.py) in the same folder.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ['Normal', 'Anxiety', 'Depression', 'Stress', 'Bipolar',
          'Personality disorder', 'Suicidal']


def build_vectorizer_and_features(train_df: pd.DataFrame):
    """Fit TF-IDF on TRAINING DATA ONLY - fitting on train+test would leak
    test vocabulary into the model before evaluation, inflating results
    in a way that won't hold up on genuinely new messages later."""
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(train_df['statement_heavy'])
    return vectorizer, X_train


def compare_models(X_train, y_train):
    """Cross-validated comparison of three classical algorithms.
    Uses f1_macro (not accuracy) because accuracy would be dominated by
    the big classes (Normal, Depression) and hide whether small classes
    (Personality disorder, Stress) are actually being learned."""
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Linear SVM': LinearSVC(class_weight='balanced'),
        'Naive Bayes': MultinomialNB(),  # no class_weight support - expected to underperform on imbalance
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
        results[name] = scores.mean()
        print(f'{name:20s} mean f1_macro: {scores.mean():.3f}')
    return results


def per_class_report(X_train, y_train):
    """Out-of-fold predictions on training data only (test.csv stays
    untouched until final evaluation) - gives an honest per-class view
    before we commit to this as the chosen baseline."""
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    y_pred = cross_val_predict(model, X_train, y_train, cv=3, n_jobs=-1)
    report = classification_report(y_train, y_pred, digits=2)
    print(report)

    cm = confusion_matrix(y_train, y_pred, labels=LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (row-normalized) - Logistic Regression baseline (train, cross-validated)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_baseline.png', dpi=150)

    return report


def train_final_model(X_train, y_train):
    """Fit on the FULL training set (no cross-validation held-out folds) -
    this is the actual model that gets saved and used downstream."""
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_on_test(model, vectorizer, test_df: pd.DataFrame):
    """The ONE time test.csv gets used - final, honest evaluation on
    data the model has never seen in any form."""
    X_test = vectorizer.transform(test_df['statement_heavy'])
    y_test = test_df['status']
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=2)
    print('=== FINAL TEST SET EVALUATION ===')
    print(report)
    with open('baseline_test_report.txt', 'w') as f:
        f.write(report)
    return report


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print('=== Building TF-IDF features ===')
    vectorizer, X_train = build_vectorizer_and_features(train_df)
    y_train = train_df['status']
    print(f'TF-IDF matrix shape: {X_train.shape}\n')

    print('=== Comparing algorithms (cross-validated, train data only) ===')
    compare_models(X_train, y_train)
    print()

    print('=== Per-class breakdown for Logistic Regression ===')
    per_class_report(X_train, y_train)
    print()

    print('=== Training final model on full training set ===')
    model = train_final_model(X_train, y_train)

    print('=== Evaluating on held-out test set ===')
    evaluate_on_test(model, vectorizer, test_df)

    joblib.dump(model, 'logistic_regression_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print('\nSaved logistic_regression_model.joblib and tfidf_vectorizer.joblib')