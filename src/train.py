"""
train.py
========
Trains all 12 pipeline variants and saves them to models/.

12 combinations: 3 vectorizers x 4 classifiers
  tfidf_naive_bayes, tfidf_logistic_regression, tfidf_linearsvc, tfidf_xgboost
  w2v_naive_bayes*,  w2v_logistic_regression,   w2v_linearsvc,   w2v_xgboost
  sbert_naive_bayes*,sbert_logistic_regression,  sbert_linearsvc, sbert_xgboost

  * NaiveBayes skipped for W2V/SBERT (requires non-negative features)
    → 10 valid pipelines total

Usage:
    python -m src.train                        # trains all
    python -m src.train --vectorizer tfidf     # trains tfidf only
    python -m src.train --model linearsvc      # trains one model across all vecs
"""

import os
import time
import argparse
import logging
import numpy as np
import joblib

from sklearn.pipeline import Pipeline

from src.vectorizers import (
    build_tfidf_pipeline,
    build_dense_pipeline,
    Word2VecTransformer,
    SBERTTransformer,
)
from src.models import get_all_classifiers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
DATA_DIR   = "data"

# NaiveBayes needs non-negative features — skip for dense vectorizers
SKIP_DENSE = {"naive_bayes"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits():
    """Load train/val/test splits saved by Phase 1 notebook."""
    logger.info("Loading data splits from %s/", DATA_DIR)
    X_train = np.load(f"{DATA_DIR}/X_train.npy", allow_pickle=True)
    X_val   = np.load(f"{DATA_DIR}/X_val.npy",   allow_pickle=True)
    X_test  = np.load(f"{DATA_DIR}/X_test.npy",  allow_pickle=True)
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    logger.info(
        "Loaded | train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Single pipeline trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train, y_train,
    X_val,   y_val,
    save: bool = True,
) -> dict:
    """
    Fit one pipeline, evaluate on val set, optionally save to disk.

    Args:
        name     : e.g. 'tfidf_linearsvc'
        pipeline : sklearn Pipeline (vectorizer + classifier)
        save     : if True, saves pipeline to models/{name}.pkl

    Returns:
        dict with keys: name, val_accuracy, val_f1, train_time_s, model_path
    """
    from sklearn.metrics import accuracy_score, f1_score

    logger.info("Training: %s", name)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    y_pred = pipeline.predict(X_val)
    val_acc = round(accuracy_score(y_val, y_pred), 4)
    val_f1  = round(f1_score(y_val, y_pred, average="macro"), 4)

    logger.info(
        "%-35s | val_acc: %.4f | val_f1: %.4f | time: %.1fs",
        name, val_acc, val_f1, train_time
    )

    model_path = None
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = f"{MODELS_DIR}/{name}.pkl"
        joblib.dump(pipeline, model_path)
        logger.info("Saved → %s", model_path)

    return {
        "name":        name,
        "val_accuracy": val_acc,
        "val_f1":       val_f1,
        "train_time_s": train_time,
        "model_path":   model_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all TF-IDF pipelines
# ─────────────────────────────────────────────────────────────────────────────

def train_tfidf_pipelines(X_train, y_train, X_val, y_val, save=True) -> list:
    """Train all 4 TF-IDF based pipelines."""
    classifiers = get_all_classifiers()
    results = []
    for clf_name, clf in classifiers.items():
        name     = f"tfidf_{clf_name}"
        pipeline = build_tfidf_pipeline(clf)
        result   = train_pipeline(name, pipeline, X_train, y_train, X_val, y_val, save)
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train all Word2Vec pipelines
# ─────────────────────────────────────────────────────────────────────────────

def train_w2v_pipelines(X_train, y_train, X_val, y_val, wv=None, save=True) -> list:
    """
    Train Word2Vec based pipelines (skips NaiveBayes).

    Args:
        wv : pre-loaded gensim KeyedVectors. If None, attempts to load
             'glove-wiki-gigaword-100' (fast) for development.
    """
    if wv is None:
        from src.vectorizers import load_word2vec
        wv = load_word2vec("glove-wiki-gigaword-100")

    classifiers = get_all_classifiers()
    results = []
    for clf_name, clf in classifiers.items():
        if clf_name in SKIP_DENSE:
            logger.info("Skipping w2v_%s (NaiveBayes needs non-negative features)", clf_name)
            continue
        name        = f"w2v_{clf_name}"
        transformer = Word2VecTransformer(wv=wv, dim=wv.vector_size)
        pipeline    = build_dense_pipeline(transformer, clf)
        result      = train_pipeline(name, pipeline, X_train, y_train, X_val, y_val, save)
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train all SBERT pipelines
# ─────────────────────────────────────────────────────────────────────────────

def train_sbert_pipelines(X_train, y_train, X_val, y_val, save=True) -> list:
    """
    Train Sentence-BERT based pipelines (skips NaiveBayes).

    Note: SBERT model downloads ~80 MB on first run, then cached locally.
    Encoding 50K reviews takes ~5-10 min on CPU, ~1 min on GPU.

    Optimization: encode all splits ONCE, then train classifiers on embeddings.
    """
    logger.info("Pre-encoding all splits with SBERT (runs once, reused for all classifiers)...")
    transformer = SBERTTransformer(model_name="all-MiniLM-L6-v2", batch_size=64)
    transformer.fit(X_train)

    t0 = time.time()
    X_train_emb = transformer.transform(X_train)
    X_val_emb   = transformer.transform(X_val)
    logger.info("SBERT encoding done in %.1fs | shape: %s", time.time() - t0, X_train_emb.shape)

    # Save embeddings to avoid recomputing
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(f"{DATA_DIR}/X_train_sbert.npy", X_train_emb)
    np.save(f"{DATA_DIR}/X_val_sbert.npy",   X_val_emb)
    logger.info("SBERT embeddings saved to data/")

    classifiers = get_all_classifiers()
    results = []
    for clf_name, clf in classifiers.items():
        if clf_name in SKIP_DENSE:
            logger.info("Skipping sbert_%s (NaiveBayes needs non-negative features)", clf_name)
            continue
        name   = f"sbert_{clf_name}"
        # For SBERT: pipeline wraps pre-fitted transformer (transform only, no re-encoding)
        pipeline = build_dense_pipeline(transformer, clf)
        result   = train_pipeline(name, pipeline, X_train, y_train, X_val, y_val, save)
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train everything + print summary table
# ─────────────────────────────────────────────────────────────────────────────

def train_all(vectorizer_filter=None, model_filter=None, save=True) -> list:
    """
    Train all (or filtered) pipeline variants.

    Args:
        vectorizer_filter : 'tfidf', 'w2v', 'sbert', or None (all)
        model_filter      : 'naive_bayes', 'logistic_regression',
                            'linearsvc', 'xgboost', or None (all)
        save              : save fitted pipelines to models/

    Returns:
        List of result dicts, sorted by val_f1 descending.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    all_results = []

    if vectorizer_filter in (None, "tfidf"):
        logger.info("═" * 50)
        logger.info("TFIDF PIPELINES")
        logger.info("═" * 50)
        all_results += train_tfidf_pipelines(X_train, y_train, X_val, y_val, save)

    if vectorizer_filter in (None, "w2v"):
        logger.info("═" * 50)
        logger.info("WORD2VEC PIPELINES")
        logger.info("═" * 50)
        all_results += train_w2v_pipelines(X_train, y_train, X_val, y_val, save=save)

    if vectorizer_filter in (None, "sbert"):
        logger.info("═" * 50)
        logger.info("SBERT PIPELINES")
        logger.info("═" * 50)
        all_results += train_sbert_pipelines(X_train, y_train, X_val, y_val, save)

    # Filter by model if specified
    if model_filter:
        all_results = [r for r in all_results if model_filter in r["name"]]

    # Sort by val_f1 descending
    all_results.sort(key=lambda x: x["val_f1"], reverse=True)

    # Print summary table
    print("\n" + "═" * 65)
    print(f"{'MODEL':<35} {'VAL_ACC':>8} {'VAL_F1':>8} {'TIME(s)':>8}")
    print("─" * 65)
    for r in all_results:
        print(f"{r['name']:<35} {r['val_accuracy']:>8.4f} {r['val_f1']:>8.4f} {r['train_time_s']:>8.1f}")
    print("═" * 65)
    print(f"Best model: {all_results[0]['name']}  (val_f1={all_results[0]['val_f1']})")

    return all_results


def load_pipeline(name: str) -> Pipeline:
    """
    Load a saved pipeline from models/{name}.pkl

    Args:
        name : e.g. 'tfidf_linearsvc'

    Returns:
        Fitted sklearn Pipeline ready for predict().
    """
    path = f"{MODELS_DIR}/{name}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run train_all() first to train and save pipelines."
        )
    logger.info("Loading pipeline from %s", path)
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IMDb sentiment pipelines")
    parser.add_argument("--vectorizer", choices=["tfidf", "w2v", "sbert"],
                        default=None, help="Train only this vectorizer group")
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression",
                                             "linearsvc", "xgboost"],
                        default=None, help="Filter to one classifier")
    parser.add_argument("--no-save", action="store_true", help="Skip saving .pkl files")
    args = parser.parse_args()

    train_all(
        vectorizer_filter=args.vectorizer,
        model_filter=args.model,
        save=not args.no_save,
    )
