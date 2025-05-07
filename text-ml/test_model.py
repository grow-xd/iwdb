#!/usr/bin/env python3
"""
test_model.py

This script loads a trained model and TF-IDF vectorizer, preprocesses input text,
and outputs the predicted label and confidence score.
"""

import pickle
import argparse
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Mapping of numeric labels to text labels (based on dataset encoding)
LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

def preprocess_text(text):
    """
    Clean and preprocess a string of text (same steps as training).
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(["#ff", "ff", "rt"])
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove mentions
    text = re.sub(r'@[\w\-]+', '', text)
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {path}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test a saved hate-speech detection model")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Path to the .pkl file of the trained model")
    parser.add_argument('-v', '--vectorizer', type=str, default="tfidf_vectorizer.pkl",
                        help="Path to the TF-IDF vectorizer .pkl file")
    parser.add_argument('-t', '--text', type=str, required=True,
                        help="Input text to classify")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_pickle(args.model)
    print(f"Loading vectorizer from {args.vectorizer}...")
    vectorizer = load_pickle(args.vectorizer)

    # Preprocess input text
    print(f"Preprocessing input text: {args.text}")
    processed = preprocess_text(args.text)

    # Transform text to TF-IDF vector
    X_input = vectorizer.transform([processed])

    # Predict label
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        label_confidences = {LABEL_MAP.get(i, str(i)): prob for i, prob in enumerate(probs)}
        idx = np.argmax(probs)
        label = LABEL_MAP.get(idx, str(idx))
        confidence = probs[idx]
    else:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_input)[0]
            label_confidences = {LABEL_MAP.get(i, str(i)): score for i, score in enumerate(scores)}
            idx = np.argmax(scores)
            label = LABEL_MAP.get(idx, str(idx))
            confidence = None
        else:
            idx = model.predict(X_input)[0]
            label = LABEL_MAP.get(idx, str(idx))
            confidence = None
            label_confidences = {label: confidence}

    # Output confidence for all labels
    print("Confidence for all labels:")
    for lbl, conf in label_confidences.items():
        print(f"  {lbl}: {conf:.4f}")

    # Output result
    print(f"Predicted label: {label}", end='')
    if confidence is not None:
        print(f"  (confidence: {confidence:.4f})")
    else:
        print()

if __name__ == "__main__":
    main()
