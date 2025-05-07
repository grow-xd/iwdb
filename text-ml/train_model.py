#!/usr/bin/env python3
"""
train_model.py

This script trains several machine learning models for hate speech detection.
It loads the dataset, preprocesses text, extracts TF-IDF features, trains multiple classifiers,
evaluates their accuracy, and saves the trained models and vectorizer to .pkl files.
"""

import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    """
    Clean and preprocess a string of text by:
    - Removing URLs, mentions, and non-letters
    - Converting to lowercase
    - Tokenizing, removing stopwords, and applying stemming.
    Returns the cleaned string.
    """
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # Extend stopwords with additional tokens used in tweets
    stop_words.update(["#ff", "ff", "rt"])
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove mentions (e.g., @username)
    text = re.sub(r'@[\w\-]+', '', text)
    # Remove non-letter characters (keep spaces)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Normalize whitespace and strip edges
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    # Tokenize by splitting on spaces
    tokens = text.split()
    # Remove stopwords and stem tokens
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
    # Re-join tokens into a single string
    return ' '.join(tokens)

def main():
    print("Loading dataset...")
    try:
        data = pd.read_csv("HateSpeechData.csv")
    except FileNotFoundError:
        print("Error: HateSpeechData.csv not found.")
        return

    # Ensure required columns are present
    if 'tweet' not in data.columns or 'class' not in data.columns:
        print("Dataset missing required columns.")
        return

    # Keep only tweet text and class label
    df = data[['tweet', 'class']].dropna()
    df['class'] = df['class'].astype(int)

    print("Preprocessing text...")
    df['processed_tweet'] = df['tweet'].apply(preprocess_text)

    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                       max_df=0.75, min_df=5,
                                       max_features=10000)
    X = tfidf_vectorizer.fit_transform(df['processed_tweet'])
    y = df['class'].values

    print("Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}
    accuracies = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred)
    print(f"  Logistic Regression accuracy: {acc_lr:.4f}")
    models['logistic_regression'] = lr
    accuracies['Logistic Regression'] = acc_lr

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred)
    print(f"  Random Forest accuracy: {acc_rf:.4f}")
    models['random_forest'] = rf
    accuracies['Random Forest'] = acc_rf

    # Gaussian Naive Bayes (requires dense array)
    print("Training Gaussian Naive Bayes...")
    nb = GaussianNB()
    X_train_nb = X_train.toarray()
    X_test_nb = X_test.toarray()
    nb.fit(X_train_nb, y_train)
    y_pred = nb.predict(X_test_nb)
    acc_nb = accuracy_score(y_test, y_pred)
    print(f"  Naive Bayes accuracy: {acc_nb:.4f}")
    models['naive_bayes'] = nb
    accuracies['Naive Bayes'] = acc_nb

    # Linear SVM (no probability)
    print("Training Linear SVM...")
    svm = LinearSVC(random_state=42, max_iter=10000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred)
    print(f"  Linear SVM accuracy: {acc_svm:.4f}")
    models['linear_svc'] = svm
    accuracies['Linear SVM'] = acc_svm

    # Identify best model
    best_name = max(accuracies, key=accuracies.get)
    print(f"Best model: {best_name} (accuracy {accuracies[best_name]:.4f})")

    print("Saving vectorizer and models to .pkl files...")
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    for name, model in models.items():
        filename = f"{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved {name} model to {filename}")

if __name__ == "__main__":
    main()
