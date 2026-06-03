"""
logger.py
=========
Experiment tracking for IMDb Sentiment Classification.

Logs every training run to runs_log.json — params, metrics, timing, model path.
Supports both a lightweight JSON tracker (default) and MLflow (optional).

Usage:
    from src.logger import log_run, load_runs, get_best_run, print_runs_table

    # after training a pipeline:
    log_run(
        name       = "tfidf_linearsvc",
        vectorizer = "tfidf",
        classifier = "linearsvc",
        params     = {"C": 1.0, "ngram_range": "(1,2)"},
        metrics    = {"val_f1": 0.923, "val_accuracy": 0.923, "test_f1": 0.921},
        train_time = 4.2,
        model_path = "models/tfidf_linearsvc.pkl",
    )
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RUNS_LOG_PATH = "runs_log.json"


# ─────────────────────────────────────────────────────────────────────────────
# Core JSON tracker
# ─────────────────────────────────────────────────────────────────────────────

def log_run(
    name:        str,
    vectorizer:  str,
    classifier:  str,
    params:      dict,
    metrics:     dict,
    train_time:  float,
    model_path:  Optional[str] = None,
    notes:       Optional[str] = None,
    log_path:    str = RUNS_LOG_PATH,
) -> dict:
    """
    Append one experiment run to the JSON log file.

    Args:
        name        : unique run name e.g. 'tfidf_linearsvc_tuned'
        vectorizer  : 'tfidf' | 'w2v' | 'sbert'
        classifier  : 'naive_bayes' | 'logistic_regression' | 'linearsvc' | 'xgboost'
        params      : dict of hyperparameters used
        metrics     : dict — must include 'val_f1'; optionally 'val_accuracy',
                      'test_f1', 'test_accuracy', 'roc_auc', 'inference_ms'
        train_time  : training time in seconds
        model_path  : path to saved .pkl file (if any)
        notes       : free-text annotation e.g. 'after Optuna tuning'
        log_path    : path to the JSON log file

    Returns:
        The run dict that was logged.
    """
    run = {
        "run_id":      _next_run_id(log_path),
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name":        name,
        "vectorizer":  vectorizer,
        "classifier":  classifier,
        "params":      params,
        "metrics":     metrics,
        "train_time_s": round(train_time, 2),
        "model_path":  model_path,
        "notes":       notes or "",
    }

    runs = _load_raw(log_path)
    runs.append(run)
    _save_raw(runs, log_path)

    logger.info(
        "Run logged | id: %s | name: %s | val_f1: %s",
        run["run_id"], name, metrics.get("val_f1", "—")
    )
    return run


def load_runs(log_path: str = RUNS_LOG_PATH) -> pd.DataFrame:
    """
    Load all logged runs as a flat DataFrame.

    Nested dicts (params, metrics) are flattened with prefix:
        params.C, metrics.val_f1, etc.

    Returns:
        pd.DataFrame sorted by val_f1 descending.
        Returns empty DataFrame if no runs logged yet.
    """
    runs = _load_raw(log_path)
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "run_id":       r["run_id"],
            "timestamp":    r["timestamp"],
            "name":         r["name"],
            "vectorizer":   r["vectorizer"],
            "classifier":   r["classifier"],
            "train_time_s": r["train_time_s"],
            "model_path":   r.get("model_path", ""),
            "notes":        r.get("notes", ""),
        }
        for k, v in r.get("params", {}).items():
            row[f"param_{k}"] = v
        for k, v in r.get("metrics", {}).items():
            row[f"metric_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if "metric_val_f1" in df.columns:
        df = df.sort_values("metric_val_f1", ascending=False).reset_index(drop=True)
    return df


def get_best_run(metric: str = "val_f1", log_path: str = RUNS_LOG_PATH) -> Optional[dict]:
    """
    Return the run dict with the highest value of the given metric.

    Args:
        metric : key inside the 'metrics' dict e.g. 'val_f1', 'test_f1', 'roc_auc'

    Returns:
        Run dict or None if no runs logged.
    """
    runs = _load_raw(log_path)
    if not runs:
        logger.warning("No runs found in %s", log_path)
        return None

    valid = [r for r in runs if metric in r.get("metrics", {})]
    if not valid:
        logger.warning("No runs have metric '%s'", metric)
        return None

    best = max(valid, key=lambda r: r["metrics"][metric])
    logger.info(
        "Best run | name: %s | %s: %.4f",
        best["name"], metric, best["metrics"][metric]
    )
    return best


def print_runs_table(log_path: str = RUNS_LOG_PATH, top_n: int = 20):
    """
    Print a formatted leaderboard table of all runs.

    Args:
        top_n : max rows to display
    """
    df = load_runs(log_path)
    if df.empty:
        print("No runs logged yet.")
        return

    cols = ["run_id", "name", "vectorizer", "classifier"]
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    cols += metric_cols + ["train_time_s"]

    display = df[cols].head(top_n)
    display.columns = [c.replace("metric_", "") for c in display.columns]

    print("\n" + "═" * 90)
    print("EXPERIMENT RUNS LEADERBOARD")
    print("═" * 90)
    print(display.to_string(index=False))
    print("═" * 90)

    if "val_f1" in display.columns:
        best = display.iloc[0]
        print(f"Best model: {best['name']}  (val_f1={best['val_f1']})")


def delete_run(run_id: int, log_path: str = RUNS_LOG_PATH):
    """Remove a run by run_id (useful to clean up failed experiments)."""
    runs = _load_raw(log_path)
    filtered = [r for r in runs if r["run_id"] != run_id]
    if len(filtered) == len(runs):
        logger.warning("run_id %d not found", run_id)
        return
    _save_raw(filtered, log_path)
    logger.info("Deleted run_id %d", run_id)


def clear_runs(log_path: str = RUNS_LOG_PATH):
    """Delete all runs (use with caution)."""
    _save_raw([], log_path)
    logger.info("Cleared all runs from %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow integration (optional — used if mlflow is installed)
# ─────────────────────────────────────────────────────────────────────────────

def log_run_mlflow(
    name:       str,
    params:     dict,
    metrics:    dict,
    model_path: Optional[str] = None,
    experiment: str = "imdb_sentiment",
):
    """
    Log a run to MLflow (if installed).
    Falls back silently if MLflow is not available.

    Setup:
        pip install mlflow
        mlflow ui          ← launch tracking UI at http://localhost:5000

    Args:
        name       : run name
        params     : hyperparameters dict
        metrics    : metrics dict
        model_path : path to saved .pkl (logged as artifact)
        experiment : MLflow experiment name
    """
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path)
        logger.info("MLflow run logged: %s", name)
    except ImportError:
        logger.debug("MLflow not installed — skipping MLflow logging. pip install mlflow")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_raw(runs: list, path: str):
    with open(path, "w") as f:
        json.dump(runs, f, indent=2)


def _next_run_id(path: str) -> int:
    runs = _load_raw(path)
    if not runs:
        return 1
    return max(r["run_id"] for r in runs) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (run: python -m src.logger)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    TEST_LOG = "/tmp/test_runs_log.json"

    # Clear previous test runs
    clear_runs(TEST_LOG)

    # Log 3 dummy runs
    log_run(
        name="tfidf_linearsvc", vectorizer="tfidf", classifier="linearsvc",
        params={"C": 1.0, "ngram_range": "(1,2)", "max_features": 100000},
        metrics={"val_f1": 0.923, "val_accuracy": 0.923, "roc_auc": 0.971},
        train_time=4.2, model_path="models/tfidf_linearsvc.pkl",
        log_path=TEST_LOG,
    )
    log_run(
        name="tfidf_logistic_regression", vectorizer="tfidf", classifier="logistic_regression",
        params={"C": 1.0, "ngram_range": "(1,2)", "max_features": 100000},
        metrics={"val_f1": 0.911, "val_accuracy": 0.912, "roc_auc": 0.965},
        train_time=6.1, model_path="models/tfidf_logistic_regression.pkl",
        log_path=TEST_LOG,
    )
    log_run(
        name="sbert_logistic_regression", vectorizer="sbert", classifier="logistic_regression",
        params={"C": 1.0, "model": "all-MiniLM-L6-v2"},
        metrics={"val_f1": 0.934, "val_accuracy": 0.934, "roc_auc": 0.980},
        train_time=312.0, model_path="models/sbert_logistic_regression.pkl",
        notes="SBERT baseline, no tuning",
        log_path=TEST_LOG,
    )

    # Print table
    print_runs_table(TEST_LOG)

    # Best run
    best = get_best_run("val_f1", TEST_LOG)
    print(f"\nBest run: {best['name']}  val_f1={best['metrics']['val_f1']}")

    # Load as DataFrame
    df = load_runs(TEST_LOG)
    print(f"\nDataFrame shape: {df.shape}")
    print(df[["name", "metric_val_f1", "metric_roc_auc", "train_time_s"]])

    print("\n✓ logger.py self-test passed")
