import pytest
from core.semantic_shift import SemanticShiftAnalyzer

class DummyEmbeddingModel:
    """Mock embedding model for fast unit testing without downloading torch models."""
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np
        if isinstance(texts, str):
            # Deterministic pseudo-embedding based on text hash
            h = hash(texts) % 1000
            vec = np.array([float(h), 1.0, 2.0, 3.0])
            return vec / np.linalg.norm(vec)
        results = []
        for t in texts:
            h = hash(t) % 1000
            vec = np.array([float(h), 1.0, 2.0, 3.0])
            results.append(vec / np.linalg.norm(vec))
        return np.array(results)

def test_compute_sentiment_scores():
    analyzer = SemanticShiftAnalyzer()
    optimistic_text = "The company achieved record growth, profitable momentum, and outperformed sales targets."
    scores = analyzer.compute_sentiment_scores(optimistic_text)

    assert scores["positive"] > 0
    assert scores["net_tone"] > 0.0

    pessimistic_text = "The company suffered severe losses, decline, impairment, and faced regulatory litigation."
    scores_neg = analyzer.compute_sentiment_scores(pessimistic_text)
    assert scores_neg["negative"] > 0
    assert scores_neg["net_tone"] < 0.0

def test_compute_sentiment_shift():
    analyzer = SemanticShiftAnalyzer()
    t0 = "The business was stable with slight growth."
    t1 = "The business suffered substantial losses, litigation penalties, and adverse declines."

    shift = analyzer.compute_sentiment_shift(t0, t1)
    assert shift["deltas"]["negative_delta"] > 0
    assert shift["deltas"]["net_tone_delta"] < 0
    assert shift["tone_drift_direction"] == "Negative/Pessimistic"

def test_compute_vocabulary_divergence():
    analyzer = SemanticShiftAnalyzer()
    t0 = "Apple designs hardware devices such as personal computers and smartphones."
    t1 = "Apple incorporates artificial intelligence, generative models, and autonomous architectures."

    div = analyzer.compute_vocabulary_divergence(t0, t1)
    assert 0.0 < div["jaccard_distance"] <= 1.0
    assert div["novel_term_ratio"] > 0.0

def test_composite_shift_index():
    analyzer = SemanticShiftAnalyzer()
    index_low = analyzer.calculate_composite_shift_index(0.05, 0.02, 0.05, 0.10)
    index_high = analyzer.calculate_composite_shift_index(0.40, 0.80, 0.60, 0.85)

    assert 0.0 <= index_low <= 1.0
    assert 0.0 <= index_high <= 1.0
    assert index_high > index_low
