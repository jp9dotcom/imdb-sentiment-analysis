"""
Streamlit demo app for IMDb Sentiment Classification.

Features:
  - Single review prediction with confidence bar
  - Batch prediction from text input
  - Model comparison table (loads runs_log.json)
  - LIME explanation for any input text

Run:
    streamlit run app/app.py
"""

import os
import re
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IMDb Sentiment Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
RUNS_LOG = "runs_log.json"
LABEL_MAP = {0: "Negative 👎", 1: "Positive 👍"}
LABEL_COLOR = {0: "#E8534C", 1: "#4C8BF5"}
DEFAULT_MODEL = "tfidf_linearsvc_tuned"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Basic preprocessing matching Phase 1 cleaning steps."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def get_available_models() -> list:
    if not os.path.exists(MODELS_DIR):
        return [DEFAULT_MODEL]
    pkls = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return sorted(pkls) if pkls else [DEFAULT_MODEL]


def predict(pipeline, text: str) -> dict:
    cleaned = preprocess(text)
    t0 = time.perf_counter()
    label_id = int(pipeline.predict([cleaned])[0])
    ms = round((time.perf_counter() - t0) * 1000, 2)
    
    proba = None
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = float(max(proba))
        
    return {
        "label_id": label_id,
        "label": LABEL_MAP[label_id],
        "color": LABEL_COLOR[label_id],
        "confidence": confidence,
        "proba": proba,
        "ms": ms,
    }


def load_runs_log() -> pd.DataFrame:
    if not os.path.exists(RUNS_LOG):
        return pd.DataFrame()
    with open(RUNS_LOG) as f:
        runs = json.load(f)
    if not runs:
        return pd.DataFrame()
    
    rows = []
    for r in runs:
        row = {
            "name": r["name"], 
            "vectorizer": r["vectorizer"],
            "classifier": r["classifier"], 
            "train_time_s": r["train_time_s"]
        }
        row.update(r.get("metrics", {}))
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
    width=120
)
st.sidebar.title("🎬 Sentiment Classifier")
st.sidebar.markdown("---")

available_models = get_available_models()
selected_model = st.sidebar.selectbox(
    "Select Model", available_models,
    index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
)

pipeline = load_model(selected_model)

if pipeline is None:
    st.sidebar.error(f"Model `{selected_model}.pkl` not found in `{MODELS_DIR}/`.")
else:
    st.sidebar.success(f"✅ Model loaded: `{selected_model}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** IMDb NLP Sentiment Analysis")
st.sidebar.markdown("**Vectorizer:** TF-IDF (unigrams + bigrams)")
st.sidebar.markdown("**Classifier:** LinearSVC (Calibrated)")
st.sidebar.markdown("**Dataset:** 50K IMDb Reviews")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Predict",
    "📋 Batch Predict",
    "📊 Model Leaderboard",
    "🧠 LIME Explanation",
])


# ─────────────────────── TAB 1: Single Predict ───────────────────────────────

with tab1:
    st.header("Single Review Prediction")
    st.markdown("Type or paste a movie review below and hit **Predict**.")

    examples = [
        "This movie was absolutely brilliant. The acting was superb and the storyline kept me hooked throughout.",
        "Complete waste of time. Terrible acting, predictable plot, and awful dialogue. Avoid at all costs.",
        "A decent film, not great but not terrible either. Some good moments but overall forgettable.",
    ]
    example_choice = st.selectbox("Or pick an example:", ["(type your own)"] + examples)
    default_text = "" if example_choice == "(type your own)" else example_choice

    user_text = st.text_area(
        "Review text:", value=default_text, height=140,
        placeholder="Enter a movie review here..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    if predict_btn:
        if not user_text.strip():
            st.warning("Please enter a review first.")
        elif pipeline is None:
            st.error("No model loaded.")
        else:
            result = predict(pipeline, user_text)

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown(f"### Prediction")
                st.markdown(
                    f"<h2 style='color:{result['color']}'>{result['label']}</h2>",
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown("### Confidence")
                if result["confidence"]:
                    pct = result["confidence"] * 100
                    st.metric("", f"{pct:.1f}%")
                    st.progress(result["confidence"])
                else:
                    st.write("N/A")

            with col_c:
                st.markdown("### Inference Time")
                st.metric("", f"{result['ms']} ms")

            if result["proba"] is not None:
                st.markdown("---")
                st.markdown("**Class Probabilities**")
                fig, ax = plt.subplots(figsize=(5, 1.5))
                colors = [LABEL_COLOR[0], LABEL_COLOR[1]]
                bars = ax.barh(["Negative", "Positive"], result["proba"],
                               color=colors, edgecolor="white")
                for bar, val in zip(bars, result["proba"]):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=10)
                ax.set_xlim(0, 1.15)
                ax.set_xlabel("Probability")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ─────────────────────── TAB 2: Batch Predict ────────────────────────────────

with tab2:
    st.header("Batch Prediction")
    st.markdown("Enter one review per line (max 100).")

    batch_text = st.text_area(
        "Reviews (one per line):",
        height=200,
        placeholder="This film was amazing!\nTerrible movie, hated every minute.\nAverage, nothing special.",
    )

    if st.button("Run Batch Predict", type="primary"):
        if not batch_text.strip():
            st.warning("Please enter at least one review.")
        elif pipeline is None:
            st.error("No model loaded.")
        else:
            lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
            if len(lines) > 100:
                st.error("Max 100 reviews per batch.")
            else:
                results = []
                for line in lines:
                    r = predict(pipeline, line)
                    results.append({
                        "Review": line[:80] + ("..." if len(line) > 80 else ""),
                        "Prediction": r["label"],
                        "Confidence": f"{r['confidence']*100:.1f}%" if r["confidence"] else "N/A",
                        "Time (ms)": r["ms"],
                    })

                df_results = pd.DataFrame(results)
                st.success(f"Predicted {len(results)} reviews")

                pos = sum(1 for r in results if "Positive" in r["Prediction"])
                neg = len(results) - pos
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(results))
                c2.metric("Positive 👍", pos)
                c3.metric("Negative 👎", neg)

                st.dataframe(df_results, use_container_width=True)

                csv = df_results.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "batch_predictions.csv", "text/csv")


# ─────────────────────── TAB 3: Leaderboard ──────────────────────────────────

with tab3:
    st.header("Model Leaderboard")
    st.markdown("All experiments logged to `runs_log.json`.")

    df_runs = load_runs_log()
    if df_runs.empty:
        st.info("No runs logged yet. Run the master orchestrator script first.")
    else:
        # Highlight best row
        metric_cols = [c for c in df_runs.columns
                       if c not in ["name", "vectorizer", "classifier", "train_time_s"]]
        st.dataframe(
            df_runs.style.highlight_max(
                subset=["macro_f1"] if "macro_f1" in df_runs.columns else [],
                color="#d4edda"
            ),
            use_container_width=True,
        )

        # F1 bar chart
        if "macro_f1" in df_runs.columns and len(df_runs) > 1:
            fig, ax = plt.subplots(figsize=(8, max(3, len(df_runs) * 0.5)))
            colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(df_runs)))[::-1]
            ax.barh(df_runs["name"][::-1], df_runs["macro_f1"][::-1], color=colors[::-1])
            ax.set_xlabel("Macro F1")
            ax.set_title("Model Comparison — Macro F1", fontweight="bold")
            ax.set_xlim(df_runs["macro_f1"].min() - 0.02, 1.0)
            for i, (_, row) in enumerate(df_runs[::-1].iterrows()):
                ax.text(row["macro_f1"] + 0.002, i, f"{row['macro_f1']:.4f}", va="center", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ─────────────────────── TAB 4: LIME Explanation ─────────────────────────────

with tab4:
    st.header("LIME Explanation")
    st.markdown(
        "See which words drove the model's prediction. "
        "LIME perturbs the input and measures how each word affects the output."
    )

    lime_text = st.text_area(
        "Review to explain:",
        height=120,
        placeholder="Paste any review here to see word-level explanations...",
    )

    n_features = st.slider("Number of top words to show", 5, 20, 10)

    if st.button("Explain", type="primary"):
        if not lime_text.strip():
            st.warning("Please enter a review.")
        elif pipeline is None:
            st.error("No model loaded.")
        else:
            try:
                from lime.lime_text import LimeTextExplainer
                explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
                cleaned = preprocess(lime_text)

                with st.spinner("Running LIME (this takes ~10 seconds)..."):
                    exp = explainer.explain_instance(
                        cleaned,
                        pipeline.predict_proba,
                        num_features=n_features,
                        num_samples=500,
                    )

                # Prediction first
                result = predict(pipeline, lime_text)
                st.markdown(
                    f"**Prediction:** <span style='color:{result['color']};font-weight:bold'>"
                    f"{result['label']}</span> — Confidence: {result['confidence']*100:.1f}%",
                    unsafe_allow_html=True,
                )

                # Word weights table
                word_weights = exp.as_list()
                df_lime = pd.DataFrame(word_weights, columns=["Word", "Weight"])
                df_lime["Direction"] = df_lime["Weight"].apply(
                    lambda w: "→ Positive" if w > 0 else "→ Negative"
                )
                df_lime["Weight"] = df_lime["Weight"].round(4)
                df_lime = df_lime.sort_values("Weight", ascending=False)

                st.markdown("**Word Contributions:**")
                st.dataframe(
                    df_lime.style.background_gradient(subset=["Weight"], cmap="RdYlGn"),
                    use_container_width=True,
                )

                # Bar chart
                fig, ax = plt.subplots(figsize=(8, max(3, len(word_weights) * 0.4)))
                colors = ["#4C8BF5" if w > 0 else "#E8534C" for _, w in word_weights]
                words, weights = zip(*word_weights)
                ax.barh(words[::-1], weights[::-1], color=colors[::-1], edgecolor="white")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("LIME Weight")
                ax.set_title("Word-Level Contributions", fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except ImportError:
                st.error("LIME not installed. Run: `pip install lime`")
            except Exception as e:
                st.error(f"LIME failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px'>"
    "IMDb Sentiment Analysis · NeoSoft Technologies Data Science Internship · 2026"
    "</div>",
    unsafe_allow_html=True,
)