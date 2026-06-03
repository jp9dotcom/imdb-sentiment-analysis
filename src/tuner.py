"""
tuner.py
========
Hyperparameter tuning for IMDb sentiment pipelines.

Two strategies:
  1. GridSearchCV   — exhaustive, best for TF-IDF (small search space)
  2. Optuna         — Bayesian optimization, best for dense + XGBoost

Usage:
    from src.tuner import tune_tfidf_pipeline, tune_dense_pipeline, tune_all
"""

import time
import logging
import numpy as np
import joblib
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models import TFIDF_PARAM_GRIDS, DENSE_PARAM_GRIDS

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
CV_FOLDS   = 5
SCORING    = "f1_macro"


# ─────────────────────────────────────────────────────────────────────────────
# 1. GridSearchCV — TF-IDF pipelines
# ─────────────────────────────────────────────────────────────────────────────

def tune_tfidf_pipeline(
    clf_name:  str,
    pipeline:  Pipeline,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    param_grid: dict = None,
    cv:        int   = CV_FOLDS,
    n_jobs:    int   = -1,
    save:      bool  = True,
) -> dict:
    """
    Tune a TF-IDF pipeline using GridSearchCV.

    Args:
        clf_name   : classifier name e.g. 'linearsvc'
        pipeline   : unfitted Pipeline([tfidf, clf])
        X_train    : training texts
        y_train    : training labels
        param_grid : override default grid from models.TFIDF_PARAM_GRIDS
        cv         : number of CV folds
        n_jobs     : parallel jobs (-1 = all cores)
        save       : save best pipeline to models/tfidf_{clf_name}_tuned.pkl

    Returns:
        dict with best_params, best_score, tuned_pipeline, model_path
    """
    grid = param_grid or TFIDF_PARAM_GRIDS.get(clf_name, {})
    if not grid:
        logger.warning("No param grid found for %s — skipping tuning", clf_name)
        return {"clf_name": clf_name, "best_params": {}, "best_score": None,
                "tuned_pipeline": pipeline, "model_path": None}

    name = f"tfidf_{clf_name}"
    logger.info("GridSearchCV tuning: %s | grid size: %d combinations",
                name, _grid_size(grid))

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=SCORING,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 1)

    logger.info(
        "%s | best_score: %.4f | best_params: %s | time: %.1fs",
        name, gs.best_score_, gs.best_params_, elapsed
    )

    model_path = None
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = f"{MODELS_DIR}/{name}_tuned.pkl"
        joblib.dump(gs.best_estimator_, model_path)
        logger.info("Saved tuned pipeline → %s", model_path)

    return {
        "name":            name,
        "best_params":     gs.best_params_,
        "best_cv_score":   round(gs.best_score_, 4),
        "cv_results":      gs.cv_results_,
        "tuned_pipeline":  gs.best_estimator_,
        "tune_time_s":     elapsed,
        "model_path":      model_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Optuna — dense pipelines (W2V / SBERT + classifiers)
# ─────────────────────────────────────────────────────────────────────────────

def tune_dense_pipeline(
    name:       str,
    clf_name:   str,
    X_train_emb: np.ndarray,
    y_train:    np.ndarray,
    X_val_emb:  np.ndarray,
    y_val:      np.ndarray,
    n_trials:   int  = 30,
    save:       bool = True,
) -> dict:
    """
    Tune a dense embedding pipeline using Optuna.

    Expects pre-computed embeddings (X_train_emb, X_val_emb) rather than raw
    text — avoids re-encoding with SBERT/W2V on every trial (huge time saving).

    Args:
        name         : full pipeline name e.g. 'sbert_xgboost'
        clf_name     : classifier name for param space selection
        X_train_emb  : pre-computed training embeddings (n, dim)
        y_train      : training labels
        X_val_emb    : pre-computed validation embeddings
        y_val        : validation labels
        n_trials     : number of Optuna trials
        save         : save best model to models/{name}_tuned.pkl

    Returns:
        dict with best_params, best_val_f1, best_model, model_path
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from sklearn.metrics import f1_score

    logger.info("Optuna tuning: %s | n_trials: %d", name, n_trials)

    def objective(trial):
        clf = _build_clf_from_trial(trial, clf_name)
        if clf is None:
            raise optuna.exceptions.TrialPruned()
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_val_emb)
        return f1_score(y_val, y_pred, average="macro")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = round(time.time() - t0, 1)

    best_params = study.best_params
    best_score  = round(study.best_value, 4)
    logger.info("%s | best_val_f1: %.4f | best_params: %s | time: %.1fs",
                name, best_score, best_params, elapsed)

    # Refit best model on full training set
    best_clf = _build_clf_from_params(clf_name, best_params)
    best_clf.fit(X_train_emb, y_train)

    model_path = None
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = f"{MODELS_DIR}/{name}_tuned.pkl"
        joblib.dump(best_clf, model_path)
        logger.info("Saved tuned model → %s", model_path)

    return {
        "name":           name,
        "best_params":    best_params,
        "best_val_f1":    best_score,
        "best_model":     best_clf,
        "tune_time_s":    elapsed,
        "model_path":     model_path,
        "study":          study,
    }


def _build_clf_from_trial(trial, clf_name: str):
    """Build a classifier with trial-suggested hyperparameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    if clf_name == "logistic_regression":
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        return LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                  n_jobs=-1, random_state=42)

    elif clf_name == "linearsvc":
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        return CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=2000, random_state=42), cv=3
        )

    elif clf_name == "xgboost":
        return XGBClassifier(
            n_estimators    = trial.suggest_int("n_estimators", 100, 500),
            max_depth       = trial.suggest_int("max_depth", 3, 10),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            eval_metric="logloss", use_label_encoder=False,
            n_jobs=-1, random_state=42,
        )
    return None


def _build_clf_from_params(clf_name: str, params: dict):
    """Rebuild a classifier from a flat params dict (post-study refit)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    if clf_name == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=1000,
                                   solver="lbfgs", n_jobs=-1, random_state=42)
    elif clf_name == "linearsvc":
        return CalibratedClassifierCV(
            LinearSVC(C=params["C"], max_iter=2000, random_state=42), cv=3
        )
    elif clf_name == "xgboost":
        p = {k: params[k] for k in params}
        return XGBClassifier(**p, eval_metric="logloss",
                              use_label_encoder=False, n_jobs=-1, random_state=42)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tune all TF-IDF pipelines in one call
# ─────────────────────────────────────────────────────────────────────────────

def tune_all_tfidf(X_train, y_train, save=True) -> list:
    """
    Tune all 4 TF-IDF pipelines with GridSearchCV.

    Returns list of tuning result dicts sorted by best_cv_score descending.
    """
    from src.vectorizers import build_tfidf_pipeline
    from src.models import get_all_classifiers

    classifiers = get_all_classifiers()
    results = []
    for clf_name, clf in classifiers.items():
        pipeline = build_tfidf_pipeline(clf)
        result   = tune_tfidf_pipeline(clf_name, pipeline, X_train, y_train, save=save)
        results.append(result)

    results.sort(key=lambda r: r["best_cv_score"] or 0, reverse=True)

    print("\n── TF-IDF Tuning Summary ──")
    for r in results:
        print(f"  {r['name']:<35} best_cv_f1: {r['best_cv_score']}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_size(param_grid: dict) -> int:
    """Calculate total number of grid combinations."""
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size


def load_tuned_pipeline(name: str):
    """Load a tuned pipeline from models/{name}_tuned.pkl"""
    path = f"{MODELS_DIR}/{name}_tuned.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tuned model not found: {path}")
    logger.info("Loading tuned pipeline: %s", path)
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from src.models import TFIDF_PARAM_GRIDS

    # Small synthetic data
    texts  = np.array(["great film loved it"] * 100 + ["terrible movie hated it"] * 100)
    labels = np.array([1] * 100 + [0] * 100)

    # Test GridSearchCV tuning
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=200))])
    small_grid = {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1.0]}
    result = tune_tfidf_pipeline("logistic_regression", pipe, texts, labels,
                                  param_grid=small_grid, cv=3, save=False)
    print(f"Best CV score : {result['best_cv_score']}")
    print(f"Best params   : {result['best_params']}")

    # Test Optuna tuning on dense features
    X_emb = np.random.randn(200, 64).astype(np.float32)
    result2 = tune_dense_pipeline(
        name="test_sbert_logreg", clf_name="logistic_regression",
        X_train_emb=X_emb[:160], y_train=labels[:160],
        X_val_emb=X_emb[160:],   y_val=labels[160:],
        n_trials=5, save=False,
    )
    print(f"Optuna best val_f1 : {result2['best_val_f1']}")
    print("\n✓ tuner.py self-test passed")
