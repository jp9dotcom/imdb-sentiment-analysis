"""
vectorizers.py
==============
Feature extraction module for IMDb Sentiment Classification.

Three vectorization strategies:
  1. TF-IDF         — sparse, classical NLP baseline
  2. Word2Vec (avg) — dense static word embeddings (gensim pretrained)
  3. Sentence-BERT  — dense contextual sentence embeddings (HuggingFace)

Each strategy is wrapped as a sklearn-compatible Transformer so it plugs
directly into sklearn.pipeline.Pipeline with zero extra glue code.

Usage:
    from src.vectorizers import build_tfidf_pipeline, Word2VecTransformer, SBERTTransformer
"""

import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(
    ngram_range: tuple = (1, 2),
    max_features: int = 100_000,
    sublinear_tf: bool = True,
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """
    Return a configured TfidfVectorizer (not yet fitted).

    Args:
        ngram_range  : (1,1) for unigrams only; (1,2) adds bigrams.
        max_features : vocabulary size cap. 100K covers IMDb well.
        sublinear_tf : apply log(1+tf) — reduces impact of very frequent terms.
        min_df       : ignore terms appearing in fewer than N documents.
        max_df       : ignore terms appearing in more than X% of documents.

    Returns:
        TfidfVectorizer instance ready for fit_transform.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Word2Vec — sklearn-compatible Transformer
# ─────────────────────────────────────────────────────────────────────────────

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Converts a list of cleaned text strings into averaged Word2Vec vectors.

    Strategy: for each review, look up each token in the pretrained word
    vectors and average all found vectors. Tokens not in the vocabulary are
    silently skipped. If NO tokens are found, a zero vector is returned.

    Args:
        wv  : gensim KeyedVectors object (already loaded).
        dim : vector dimensionality — must match the loaded model (300 for
              Google News / GloVe-300).

    Example:
        import gensim.downloader as api
        wv = api.load('word2vec-google-news-300')
        transformer = Word2VecTransformer(wv=wv, dim=300)
        X = transformer.fit_transform(X_train)
    """

    def __init__(self, wv=None, dim: int = 300):
        self.wv = wv
        self.dim = dim

    def fit(self, X, y=None):
        # Nothing to fit — vectors are pretrained.
        return self

    def transform(self, X):
        """
        Args:
            X : array-like of strings (cleaned reviews).

        Returns:
            np.ndarray of shape (n_samples, dim).
        """
        if self.wv is None:
            raise ValueError(
                "Word2VecTransformer: wv is None. "
                "Load a gensim KeyedVectors object first.\n"
                "  import gensim.downloader as api\n"
                "  wv = api.load('word2vec-google-news-300')\n"
                "  transformer = Word2VecTransformer(wv=wv)"
            )

        result = []
        oov_count = 0

        for text in X:
            tokens = str(text).split()
            vecs = [self.wv[t] for t in tokens if t in self.wv]
            if vecs:
                result.append(np.mean(vecs, axis=0).astype(np.float32))
            else:
                result.append(np.zeros(self.dim, dtype=np.float32))
                oov_count += 1

        if oov_count > 0:
            logger.debug(
                "Word2VecTransformer: %d/%d samples had no in-vocabulary tokens.",
                oov_count, len(X)
            )

        return np.array(result, dtype=np.float32)


def load_word2vec(model_name: str = "word2vec-google-news-300"):
    """
    Download and return a gensim pretrained word vector model.

    Common model names:
        'word2vec-google-news-300'   — 300-d, 3M vocab, ~1.5 GB download
        'glove-wiki-gigaword-300'    — 300-d GloVe, smaller download
        'glove-wiki-gigaword-100'    — 100-d GloVe, faster for experiments

    Note:
        Models are cached after first download in ~/gensim-data/.
        Use 'glove-wiki-gigaword-100' for fast iteration, then switch to
        'word2vec-google-news-300' for final results.

    Returns:
        gensim.models.KeyedVectors
    """
    try:
        import gensim.downloader as api
        logger.info("Loading word vectors: %s  (cached after first download)", model_name)
        wv = api.load(model_name)
        logger.info("Loaded %s | vocab: %d | dim: %d", model_name, len(wv), wv.vector_size)
        return wv
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Word2Vec model '{model_name}'.\n"
            f"Error: {e}\n"
            "Ensure you have an internet connection for first-time download."
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sentence-BERT — sklearn-compatible Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SBERTTransformer(BaseEstimator, TransformerMixin):
    """
    Converts text strings into Sentence-BERT contextual embeddings.

    Uses the sentence-transformers library. The model is downloaded from
    HuggingFace on first use and cached locally (~80 MB).

    Recommended model:
        'all-MiniLM-L6-v2'  — 384-d, fast, excellent quality for
                               sentiment/classification tasks.

    Args:
        model_name : HuggingFace model identifier string.
        batch_size : sentences per batch. Increase if using GPU.
        device     : 'cpu', 'cuda', or None (auto-detect).

    Example:
        transformer = SBERTTransformer(model_name='all-MiniLM-L6-v2')
        transformer.fit(X_train)          # downloads model on first run
        X_emb = transformer.transform(X_train)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def fit(self, X, y=None):
        """Load the SBERT model (downloads once, cached in ~/.cache/huggingface)."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "SBERT ready | dim: %d | device: %s",
                self._model.get_sentence_embedding_dimension(),
                self._model.device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SBERT model '{self.model_name}'.\n"
                f"Error: {e}\n"
                "Ensure sentence-transformers is installed and HuggingFace is reachable:\n"
                "  pip install sentence-transformers"
            ) from e
        return self

    def transform(self, X):
        """
        Args:
            X : array-like of strings.

        Returns:
            np.ndarray of shape (n_samples, embedding_dim).
        """
        if self._model is None:
            raise RuntimeError(
                "SBERTTransformer: model not loaded. Call .fit() first."
            )
        logger.info(
            "Encoding %d samples with SBERT (batch_size=%d)...",
            len(X), self.batch_size
        )
        embeddings = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_pipeline(classifier, **tfidf_kwargs) -> Pipeline:
    """
    Return a Pipeline([tfidf, classifier]).

    Vectorizer is fitted inside cross-validation folds — no data leakage.

    Args:
        classifier   : any sklearn-compatible classifier instance.
        **tfidf_kwargs : forwarded to build_tfidf_vectorizer().

    Returns:
        sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer(**tfidf_kwargs)),
        ("clf",   classifier),
    ])


def build_dense_pipeline(transformer, classifier) -> Pipeline:
    """
    Return a Pipeline([dense_transformer, classifier]) for Word2Vec or SBERT.

    Note:
        For Word2Vec, pass a pre-configured Word2VecTransformer (wv already
        loaded). Its fit() is a no-op, so cross-validation is safe.
        For SBERT, the model downloads during the first pipeline.fit() call.

    Args:
        transformer : Word2VecTransformer or SBERTTransformer instance.
        classifier  : any sklearn-compatible classifier instance.

    Returns:
        sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("embedder", transformer),
        ("clf",      classifier),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (run: python -m src.vectorizers)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sample_texts = [
        "absolutely loved this film brilliant acting",
        "terrible waste of time avoid this movie",
        "outstanding cinematography and gripping storyline",
    ]

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("\n── TF-IDF ──")
    tfidf = build_tfidf_vectorizer(min_df=1)
    X_tfidf = tfidf.fit_transform(sample_texts)
    print(f"Shape  : {X_tfidf.shape}")
    print(f"Dtype  : {X_tfidf.dtype}")
    print(f"Vocab  : {len(tfidf.vocabulary_)} terms")

    # ── Word2Vec (mock) ───────────────────────────────────────────────────────
    print("\n── Word2Vec (mock wv for unit test) ──")

    class _MockWV:
        """Minimal mock of gensim KeyedVectors for offline unit testing."""
        _vocab = {"loved", "film", "brilliant", "terrible", "movie", "outstanding"}
        vector_size = 300

        def __contains__(self, word):
            return word in self._vocab

        def __getitem__(self, word):
            np.random.seed(abs(hash(word)) % (2 ** 31))
            return np.random.randn(300).astype(np.float32)

        def __len__(self):
            return len(self._vocab)

    w2v = Word2VecTransformer(wv=_MockWV(), dim=300)
    X_w2v = w2v.fit_transform(sample_texts)
    print(f"Shape  : {X_w2v.shape}")
    print(f"Dtype  : {X_w2v.dtype}")

    # ── SBERT ─────────────────────────────────────────────────────────────────
    print("\n── SBERT ──")
    print("Skipping model download in unit test (needs HuggingFace access).")
    print("On your machine:")
    print("  t = SBERTTransformer('all-MiniLM-L6-v2')")
    print("  t.fit(X_train)")
    print("  X_sbert = t.transform(X_train)   # shape: (n, 384)")

    print("\n✓ vectorizers.py self-test passed")
