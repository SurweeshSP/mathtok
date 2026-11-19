"""
Lightweight Expression-Equivalence Classification Evaluation
============================================================

Compares Raw Text Representation vs MathTok Structural Representation
using TF-IDF + Logistic Regression for mathematical expression equivalence.

Enforces a strict leakage-resistant split: expressions belonging to the
same rewrite family are isolated into either Train or Test, never split.

Follows Ponytail clean-code principles: flat, explicit, standard library
+ scikit-learn, zero unnecessary abstractions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from mathtok.pipeline import MathTokPipeline


def get_equivalence_dataset() -> List[Dict[str, Any]]:
    """
    Constructs a diverse set of positive and negative expression pairs,
    tagged by rewrite family to ensure leakage-resistant splitting.
    """
    pairs = [
        # Family: distributive
        {"family": "distributive", "e1": "a*(b + c)", "e2": "a*b + a*c", "label": 1},
        {"family": "distributive", "e1": "2*(x + y)", "e2": "2*x + 2*y", "label": 1},
        {"family": "distributive", "e1": "x*(y + z)", "e2": "x*y + x*z", "label": 1},
        {"family": "distributive", "e1": "3*(a - b)", "e2": "3*a - 3*b", "label": 1},
        {"family": "distributive", "e1": "a*(b + c)", "e2": "a*b + c", "label": 0},
        {"family": "distributive", "e1": "2*(x + y)", "e2": "2*x + y", "label": 0},
        
        # Family: diff_squares
        {"family": "diff_squares", "e1": "x^2 - y^2", "e2": "(x - y)*(x + y)", "label": 1},
        {"family": "diff_squares", "e1": "a^2 - 4", "e2": "(a - 2)*(a + 2)", "label": 1},
        {"family": "diff_squares", "e1": "4*x^2 - 9", "e2": "(2*x - 3)*(2*x + 3)", "label": 1},
        {"family": "diff_squares", "e1": "x^2 - y^2", "e2": "(x - y)^2", "label": 0},
        {"family": "diff_squares", "e1": "a^2 - 4", "e2": "(a - 2)^2", "label": 0},

        # Family: binomial_sq
        {"family": "binomial_sq", "e1": "(x + 1)^2", "e2": "x^2 + 2*x + 1", "label": 1},
        {"family": "binomial_sq", "e1": "(a + b)^2", "e2": "a^2 + 2*a*b + b^2", "label": 1},
        {"family": "binomial_sq", "e1": "(x - 2)^2", "e2": "x^2 - 4*x + 4", "label": 1},
        {"family": "binomial_sq", "e1": "(2*x + 3)^2", "e2": "4*x^2 + 12*x + 9", "label": 1},
        {"family": "binomial_sq", "e1": "(x + 1)^2", "e2": "x^2 + 1", "label": 0},
        {"family": "binomial_sq", "e1": "(a + b)^2", "e2": "a^2 + b^2", "label": 0},
        {"family": "binomial_sq", "e1": "(x - 2)^2", "e2": "x^2 - 4", "label": 0},

        # Family: commutativity
        {"family": "commutativity", "e1": "x + 2", "e2": "2 + x", "label": 1},
        {"family": "commutativity", "e1": "a*b", "e2": "b*a", "label": 1},
        {"family": "commutativity", "e1": "3 + y", "e2": "y + 3", "label": 1},
        {"family": "commutativity", "e1": "x - 2", "e2": "2 - x", "label": 0},
        {"family": "commutativity", "e1": "a / b", "e2": "b / a", "label": 0},

        # Family: pythagorean_trig
        {"family": "pythagorean_trig", "e1": "sin(x)^2 + cos(x)^2", "e2": "1", "label": 1},
        {"family": "pythagorean_trig", "e1": "1 + tan(x)^2", "e2": "sec(x)^2", "label": 1},
        {"family": "pythagorean_trig", "e1": "1 - cos(x)^2", "e2": "sin(x)^2", "label": 1},
        {"family": "pythagorean_trig", "e1": "sin(x)^2 + cos(x)^2", "e2": "2", "label": 0},
        {"family": "pythagorean_trig", "e1": "1 + tan(x)^2", "e2": "cos(x)^2", "label": 0},

        # Family: double_angle_trig
        {"family": "double_angle_trig", "e1": "2*sin(x)*cos(x)", "e2": "sin(2*x)", "label": 1},
        {"family": "double_angle_trig", "e1": "cos(x)^2 - sin(x)^2", "e2": "cos(2*x)", "label": 1},
        {"family": "double_angle_trig", "e1": "2*cos(x)^2 - 1", "e2": "cos(2*x)", "label": 1},
        {"family": "double_angle_trig", "e1": "2*sin(x)*cos(x)", "e2": "cos(2*x)", "label": 0},
        {"family": "double_angle_trig", "e1": "cos(x)^2 - sin(x)^2", "e2": "sin(2*x)", "label": 0},

        # Family: log_rules
        {"family": "log_rules", "e1": "log(a*b)", "e2": "log(a) + log(b)", "label": 1},
        {"family": "log_rules", "e1": "log(x^2)", "e2": "2*log(x)", "label": 1},
        {"family": "log_rules", "e1": "log(a/b)", "e2": "log(a) - log(b)", "label": 1},
        {"family": "log_rules", "e1": "exp(log(x))", "e2": "x", "label": 1},
        {"family": "log_rules", "e1": "log(a + b)", "e2": "log(a) + log(b)", "label": 0},
        {"family": "log_rules", "e1": "log(x^2)", "e2": "log(x)^2", "label": 0},

        # Family: rational_cancellation
        {"family": "rational_cancellation", "e1": "(x^2 - 1)/(x - 1)", "e2": "x + 1", "label": 1},
        {"family": "rational_cancellation", "e1": "(x^2 - 9)/(x + 3)", "e2": "x - 3", "label": 1},
        {"family": "rational_cancellation", "e1": "(2*x + 4)/2", "e2": "x + 2", "label": 1},
        {"family": "rational_cancellation", "e1": "(x^2 - 1)/(x - 1)", "e2": "x - 1", "label": 0},
        {"family": "rational_cancellation", "e1": "(x^2 - 9)/(x + 3)", "e2": "x + 3", "label": 0},

        # Family: exponent_rules
        {"family": "exponent_rules", "e1": "x^a * x^b", "e2": "x^(a + b)", "label": 1},
        {"family": "exponent_rules", "e1": "(x^a)^b", "e2": "x^(a*b)", "label": 1},
        {"family": "exponent_rules", "e1": "x^a / x^b", "e2": "x^(a - b)", "label": 1},
        {"family": "exponent_rules", "e1": "x^a * x^b", "e2": "x^(a*b)", "label": 0},
        {"family": "exponent_rules", "e1": "(x^a)^b", "e2": "x^(a + b)", "label": 0},
    ]
    return pairs


def encode_mathtok_string(expr: str, pipeline: MathTokPipeline) -> str:
    """Encodes an expression into a space-separated MathTok structural token string."""
    try:
        out = pipeline.encode_math_only(expr)
        tokens = [t for t in out.tokens if t not in ("[MATH_START]", "[MATH_END]")]
        return " ".join(tokens)
    except Exception:
        return expr


def create_feature_vectors(
    data: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    use_mathtok: bool,
    pipeline: MathTokPipeline,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms pairs into joint difference/product feature vectors.
    """
    texts1 = []
    texts2 = []
    labels = []

    for item in data:
        e1 = encode_mathtok_string(item["e1"], pipeline) if use_mathtok else item["e1"]
        e2 = encode_mathtok_string(item["e2"], pipeline) if use_mathtok else item["e2"]
        texts1.append(e1)
        texts2.append(e2)
        labels.append(item["label"])

    if fit:
        vectorizer.fit(texts1 + texts2)

    v1 = vectorizer.transform(texts1).toarray()
    v2 = vectorizer.transform(texts2).toarray()

    # Pair features: absolute difference and elementwise product
    features = np.hstack([np.abs(v1 - v2), v1 * v2])
    return features, np.array(labels)


def run_classifier_evaluation(seed: int = 42) -> Dict[str, Any]:
    """
    Trains and evaluates TF-IDF + Logistic Regression on Raw Text vs MathTok.
    Enforces leakage-resistant train/test splitting by family.
    """
    data = get_equivalence_dataset()
    random.seed(seed)
    np.random.seed(seed)

    # 1. Leakage-resistant split by family
    all_families = sorted(list({item["family"] for item in data}))
    random.shuffle(all_families)
    
    # 60% train families, 40% test families
    split_idx = int(len(all_families) * 0.6)
    train_families = set(all_families[:split_idx])
    test_families = set(all_families[split_idx:])

    train_data = [item for item in data if item["family"] in train_families]
    test_data = [item for item in data if item["family"] in test_families]

    pipeline = MathTokPipeline()

    results = {
        "train_families": sorted(list(train_families)),
        "test_families": sorted(list(test_families)),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "models": {},
    }

    for name, use_mathtok in [("Raw_Text_TFIDF", False), ("MathTok_AST_TFIDF", True)]:
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[\+\-\*\/\^\(\)\=\<\>]", ngram_range=(1, 2))
        X_train, y_train = create_feature_vectors(train_data, vec, use_mathtok, pipeline, fit=True)
        X_test, y_test = create_feature_vectors(test_data, vec, use_mathtok, pipeline, fit=False)

        clf = LogisticRegression(random_state=seed, max_iter=500, C=1.0)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros_like(preds)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5

        results["models"][name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "num_features": X_train.shape[1],
        }

    output_path = Path(__file__).parent / "results" / "classifier_equivalence_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run_classifier_evaluation()
    print("Expression Equivalence Classification Results:")
    print(json.dumps(res, indent=2))
