"""
models.py
=========
Classifier definitions for IMDb Sentiment Classification.

Four classifiers, each returning a fresh instance with sensible defaults:
  1. Naive Bayes        — fast baseline (ComplementNB, better for binary)
  2. Logistic Regression — strong linear baseline
  3. LinearSVC          — best on sparse TF-IDF (wrapped for predict_proba)
  4. XGBoost            — ensemble, best on dense embeddings

Usage:
    from src.models import get_all_classifiers, get_classifier
    clfs = get_all_classifiers()
"""

from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Individual classifier builders
# ─────────────────────────────────────────────────────────────────────────────

def get_naive_bayes(alpha: float = 1.0) -> ComplementNB:
    """
    ComplementNB — outperforms MultinomialNB on imbalanced text tasks.
    Works only with non-negative features (TF-IDF safe; NOT for dense embeddings).

    Args:
        alpha : Laplace smoothing parameter.
    """
    return ComplementNB(alpha=alpha)


def get_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """
    Logistic Regression — strong linear classifier, works on both
    sparse (TF-IDF) and dense (W2V, SBERT) features.

    Args:
        C        : inverse regularization strength. Smaller = more regularization.
        max_iter : solver iteration limit (increase if convergence warning appears).
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )


def get_linearsvc(C: float = 1.0) -> CalibratedClassifierCV:
    """
    LinearSVC wrapped in CalibratedClassifierCV.

    Why CalibratedClassifierCV?
        LinearSVC does not support predict_proba natively.
        Wrapping with isotonic calibration adds probability estimates,
        which are needed for ROC-AUC computation and Streamlit confidence bar.

    Best suited for sparse TF-IDF features — typically fastest + highest F1.

    Args:
        C : regularization parameter.
    """
    return CalibratedClassifierCV(
        LinearSVC(C=C, max_iter=2000, random_state=42),
        cv=3,
        method="isotonic",
    )


def get_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBClassifier:
    """
    XGBoost — gradient boosted trees.
    Works best on dense embeddings (W2V, SBERT); slower on high-dim sparse.

    Args:
        n_estimators    : number of boosting rounds.
        max_depth       : tree depth — controls overfitting.
        learning_rate   : step size shrinkage.
        subsample       : row sampling ratio per tree.
        colsample_bytree: feature sampling ratio per tree.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry — returns all 4 classifiers as a dict
# ─────────────────────────────────────────────────────────────────────────────

def get_all_classifiers() -> dict:
    """
    Return a dict of {name: classifier_instance} for all 4 classifiers.

    Note: each call returns FRESH instances — safe to call multiple times
    without shared state between experiments.

    Returns:
        {
          'naive_bayes'         : ComplementNB,
          'logistic_regression' : LogisticRegression,
          'linearsvc'           : CalibratedClassifierCV(LinearSVC),
          'xgboost'             : XGBClassifier,
        }

    Usage:
        clfs = get_all_classifiers()
        for name, clf in clfs.items():
            pipeline = build_tfidf_pipeline(clf)
            pipeline.fit(X_train, y_train)
    """
    return {
        "naive_bayes":          get_naive_bayes(),
        "logistic_regression":  get_logistic_regression(),
        "linearsvc":            get_linearsvc(),
        "xgboost":              get_xgboost(),
    }


def get_classifier(name: str):
    """
    Return a single classifier by name.

    Args:
        name : one of 'naive_bayes', 'logistic_regression',
                      'linearsvc', 'xgboost'

    Raises:
        ValueError if name is not recognized.
    """
    registry = {
        "naive_bayes":         get_naive_bayes,
        "logistic_regression": get_logistic_regression,
        "linearsvc":           get_linearsvc,
        "xgboost":             get_xgboost,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search grids (used by tuner.py)
# ─────────────────────────────────────────────────────────────────────────────

TFIDF_PARAM_GRIDS = {
    "naive_bayes": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [50_000, 100_000],
        "clf__alpha": [0.1, 0.5, 1.0],
    },
    "logistic_regression": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [50_000, 100_000],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "linearsvc": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [50_000, 100_000],
        "clf__estimator__C": [0.1, 1.0, 10.0],
    },
    "xgboost": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
    },
}

DENSE_PARAM_GRIDS = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "linearsvc": {
        "clf__estimator__C": [0.01, 0.1, 1.0, 10.0],
    },
    "xgboost": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.7, 0.8],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (run: python -m src.models)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clfs = get_all_classifiers()
    print("Classifiers instantiated:")
    for name, clf in clfs.items():
        print(f"  {name:25s} → {type(clf).__name__}")

    print("\nSingle lookup:")
    clf = get_classifier("linearsvc")
    print(f"  linearsvc → {type(clf).__name__}")

    print("\nTF-IDF param grids:")
    for name, grid in TFIDF_PARAM_GRIDS.items():
        print(f"  {name}: {list(grid.keys())}")

    print("\n✓ models.py self-test passed")
