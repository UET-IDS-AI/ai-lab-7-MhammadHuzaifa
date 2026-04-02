"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Step 1: Tokenize
    tokenized = [text.lower().split() for text in texts]

    # Step 2: Build vocabulary
    vocab = set(word for doc in tokenized for word in doc)

    # Step 3: Class priors P(class)
    n_total = len(labels)
    classes = [0, 1]
    priors = {c: float(np.sum(labels == c)) / n_total for c in classes}

    # Step 4: Word probabilities using MLE (no smoothing)
    # P(word | class) = count(word in class docs) / total words in class docs
    word_probs = {c: {} for c in classes}

    for c in classes:
        # Get all words from docs of class c
        class_words = []
        for doc, label in zip(tokenized, labels):
            if label == c:
                class_words.extend(doc)

        total_words = len(class_words)

        # Count each word
        for word in vocab:
            count = class_words.count(word)
            word_probs[c][word] = count / total_words if total_words > 0 else 0.0

    # Step 5: Predict test_email
    test_tokens = test_email.lower().split()

    log_scores = {}
    for c in classes:
        # log P(class) + sum of log P(word | class)
        score = np.log(priors[c])
        for word in test_tokens:
            if word in word_probs[c] and word_probs[c][word] > 0:
                score += np.log(word_probs[c][word])
            else:
                # Zero probability word → log(0) = -inf → class impossible
                score = float('-inf')
                break
        log_scores[c] = score

    prediction = max(log_scores, key=lambda c: log_scores[c])

    return priors, word_probs, prediction


def knn_iris(k=3, test_size=0.2, seed=0):
    # Step 1: Load Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # Step 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Step 3 & 4: Euclidean distance + majority vote prediction
    def predict(X_train, y_train, X_query, k):
        preds = []
        for x in X_query:
            # Euclidean distances to all training points
            dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
            # k nearest neighbor indices
            nn_indices = np.argsort(dists)[:k]
            # Majority vote
            nn_labels = y_train[nn_indices]
            counts = np.bincount(nn_labels)
            preds.append(np.argmax(counts))
        return np.array(preds)

    # Step 5: Compute predictions
    test_predictions = predict(X_train, y_train, X_test, k)
    train_predictions = predict(X_train, y_train, X_train, k)

    # Step 6: Accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
