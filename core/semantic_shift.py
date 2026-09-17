import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from config import LOUGHRAN_MCDONALD_LEXICON

class SemanticShiftAnalyzer:
    """
    Quantifies semantic and narrative shifts across corporate annual reports
    for the TEEP 2026 research project:
    'Quantifying Semantic Shifts in Corporate Annual Reports Using Generative Artificial
    Intelligence and Retrieval-Augmented Generation: Predicting Operating Performance
    and Excess Returns.'
    """
    def __init__(self, spacy_nlp=None):
        self.nlp = spacy_nlp
        self.lexicon = LOUGHRAN_MCDONALD_LEXICON

    def compute_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Computes normalized Loughran-McDonald financial sentiment scores across:
        - negative, positive, uncertainty, litigious, constraining
        Scores represent term frequency per 1,000 words.
        """
        if not text.strip():
            return {cat: 0.0 for cat in self.lexicon.keys()} | {"net_tone": 0.0, "total_words": 0}

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        total_words = max(len(words), 1)
        word_counts = Counter(words)

        scores = {}
        for category, terms in self.lexicon.items():
            cat_hits = sum(word_counts.get(term, 0) for term in terms)
            scores[category] = round((cat_hits / total_words) * 1000.0, 3)

        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        scores["net_tone"] = round((pos - neg) / (pos + neg + 1e-6), 4)
        scores["total_words"] = total_words

        return scores

    def compute_sentiment_shift(self, text_t0: str, text_t1: str) -> Dict[str, Any]:
        """
        Calculates Year-over-Year sentiment drift: Delta S = S(t1) - S(t0).
        """
        scores_t0 = self.compute_sentiment_scores(text_t0)
        scores_t1 = self.compute_sentiment_scores(text_t1)

        deltas = {}
        for category in self.lexicon.keys():
            deltas[f"{category}_delta"] = round(scores_t1[category] - scores_t0[category], 3)

        deltas["net_tone_delta"] = round(scores_t1["net_tone"] - scores_t0["net_tone"], 4)

        return {
            "period_t0_scores": scores_t0,
            "period_t1_scores": scores_t1,
            "deltas": deltas,
            "tone_drift_direction": "Positive/Optimistic" if deltas["net_tone_delta"] > 0.05 else (
                "Negative/Pessimistic" if deltas["net_tone_delta"] < -0.05 else "Neutral/Stable"
            )
        }

    def compute_embedding_shift(self, text_t0: str, text_t1: str, embedding_model) -> Dict[str, float]:
        """
        Computes angular displacement and cosine distance between document representations:
        Cosine Distance Delta = 1 - Cosine Similarity.
        """
        if not text_t0.strip() or not text_t1.strip():
            return {"cosine_similarity": 0.0, "cosine_distance": 1.0, "angular_shift_rad": round(math.pi / 2, 4)}

        emb_t0 = embedding_model.encode(text_t0[:8000], convert_to_numpy=True)
        emb_t1 = embedding_model.encode(text_t1[:8000], convert_to_numpy=True)

        norm_0 = np.linalg.norm(emb_t0)
        norm_1 = np.linalg.norm(emb_t1)

        if norm_0 == 0 or norm_1 == 0:
            return {"cosine_similarity": 0.0, "cosine_distance": 1.0, "angular_shift_rad": 1.5708}

        cosine_sim = float(np.dot(emb_t0, emb_t1) / (norm_0 * norm_1))
        cosine_sim = max(min(cosine_sim, 1.0), -1.0)
        cosine_dist = float(1.0 - cosine_sim)
        angular_dist = float(math.acos(cosine_sim))

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "cosine_distance": round(cosine_dist, 4),
            "angular_shift_rad": round(angular_dist, 4)
        }

    def extract_risk_bullet_items(self, text: str) -> List[str]:
        """
        Extracts discrete risk factors / bullet items from SEC Item 1A.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        risk_items = []
        current_item = []

        for line in lines:
            is_header = (
                (line.endswith(':') or line.endswith('.')) and len(line) < 160 and
                any(kw in line.lower() for kw in ["risk", "adversely", "could", "may", "failure", "impact", "disruption", "competition", "regulation", "litigation", "intellectual property", "ai"])
            ) or re.match(r'^(?:[•\-\*]|\d+[\.\)])\s+', line)

            if is_header and current_item:
                risk_items.append(" ".join(current_item))
                current_item = [re.sub(r'^(?:[•\-\*]|\d+[\.\)])\s+', '', line)]
            else:
                current_item.append(line)

        if current_item:
            risk_items.append(" ".join(current_item))

        valid_items = [r for r in risk_items if len(r) > 60]
        return valid_items if valid_items else [text[:1000]]

    def compare_risk_factors(self, text_t0: str, text_t1: str, embedding_model) -> Dict[str, Any]:
        """
        Identifies:
        1. Newly Added Risk Disclosures in t1
        2. Removed / Retired Risk Disclosures from t0
        3. Persisting / Modified Risk Disclosures with similarity rating
        """
        items_t0 = self.extract_risk_bullet_items(text_t0)
        items_t1 = self.extract_risk_bullet_items(text_t1)

        if not items_t0 or not items_t1:
            return {"added_risks": [], "removed_risks": [], "modified_risks": [], "retention_rate": 1.0}

        embs_t0 = embedding_model.encode(items_t0, convert_to_numpy=True)
        embs_t1 = embedding_model.encode(items_t1, convert_to_numpy=True)

        embs_t0 = embs_t0 / np.linalg.norm(embs_t0, axis=1, keepdims=True)
        embs_t1 = embs_t1 / np.linalg.norm(embs_t1, axis=1, keepdims=True)

        sim_matrix = np.dot(embs_t1, embs_t0.T)

        added_risks = []
        modified_risks = []
        matched_t0_indices = set()

        for i, item_1 in enumerate(items_t1):
            max_sim_idx = int(np.argmax(sim_matrix[i]))
            max_sim_val = float(sim_matrix[i, max_sim_idx])

            if max_sim_val < 0.68:
                added_risks.append({
                    "risk_text": item_1,
                    "novelty_score": round(1.0 - max_sim_val, 3)
                })
            else:
                matched_t0_indices.add(max_sim_idx)
                if max_sim_val < 0.90:
                    modified_risks.append({
                        "prior_version": items_t0[max_sim_idx],
                        "current_version": item_1,
                        "similarity": round(max_sim_val, 3),
                        "shift_type": "Expanded / Reframed Disclosure"
                    })

        removed_risks = []
        for j, item_0 in enumerate(items_t0):
            if j not in matched_t0_indices:
                max_forward_sim = float(np.max(sim_matrix[:, j])) if len(items_t1) > 0 else 0.0
                if max_forward_sim < 0.68:
                    removed_risks.append({
                        "risk_text": item_0,
                        "omission_confidence": round(1.0 - max_forward_sim, 3)
                    })

        retention_rate = round(len(matched_t0_indices) / max(len(items_t0), 1), 3)

        return {
            "total_risks_t0": len(items_t0),
            "total_risks_t1": len(items_t1),
            "added_risks": added_risks,
            "removed_risks": removed_risks,
            "modified_risks": modified_risks,
            "retention_rate": retention_rate,
            "risk_expansion_count": len(added_risks) - len(removed_risks)
        }

    def compute_vocabulary_divergence(self, text_t0: str, text_t1: str) -> Dict[str, float]:
        """
        Computes lexical divergence between two filings.
        """
        def get_words(t):
            return re.findall(r'\b[a-z]{4,}\b', t.lower())

        words_0 = get_words(text_t0)
        words_1 = get_words(text_t1)

        set_0 = set(words_0)
        set_1 = set(words_1)

        if not set_0 or not set_1:
            return {"jaccard_distance": 1.0, "jensen_shannon_divergence": 1.0, "novel_term_ratio": 1.0}

        intersection = len(set_0 & set_1)
        union = len(set_0 | set_1)
        jaccard_dist = 1.0 - (intersection / union) if union > 0 else 1.0

        novel_terms = len(set_1 - set_0)
        novel_ratio = novel_terms / len(set_1) if len(set_1) > 0 else 0.0

        vocab = list(set_0 | set_1)
        counts_0 = Counter(words_0)
        counts_1 = Counter(words_1)

        p = np.array([counts_0.get(w, 0) + 1e-5 for w in vocab])
        q = np.array([counts_1.get(w, 0) + 1e-5 for w in vocab])
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = math.sqrt(max(0.5 * (kl_pm + kl_qm), 0.0))

        return {
            "jaccard_distance": round(jaccard_dist, 4),
            "jensen_shannon_divergence": round(float(jsd), 4),
            "novel_term_ratio": round(novel_ratio, 4),
            "vocabulary_growth_rate": round((len(set_1) - len(set_0)) / max(len(set_0), 1), 3)
        }

    def calculate_composite_shift_index(
        self,
        embedding_distance: float,
        sentiment_delta_abs: float,
        risk_added_ratio: float,
        vocabulary_divergence: float
    ) -> float:
        """
        Synthesizes a normalized composite Semantic Shift Index S_shift in [0.0, 1.0].
        """
        w_emb = 0.35
        w_sent = 0.25
        w_risk = 0.25
        w_vocab = 0.15

        norm_sent = min(abs(sentiment_delta_abs) / 2.0, 1.0)
        norm_risk = min(risk_added_ratio, 1.0)
        norm_emb = min(embedding_distance * 2.0, 1.0)
        norm_vocab = min(vocabulary_divergence, 1.0)

        composite = (
            w_emb * norm_emb +
            w_sent * norm_sent +
            w_risk * norm_risk +
            w_vocab * norm_vocab
        )
        return round(float(composite), 4)

    def analyze_reports(
        self,
        text_t0: str,
        text_t1: str,
        year_t0: str,
        year_t1: str,
        embedding_model,
        company_name: str = "Company"
    ) -> Dict[str, Any]:
        """
        Full end-to-end multi-dimensional semantic shift quantification suite.
        """
        emb_metrics = self.compute_embedding_shift(text_t0, text_t1, embedding_model)
        sent_metrics = self.compute_sentiment_shift(text_t0, text_t1)
        risk_metrics = self.compare_risk_factors(text_t0, text_t1, embedding_model)
        vocab_metrics = self.compute_vocabulary_divergence(text_t0, text_t1)

        added_ratio = len(risk_metrics["added_risks"]) / max(risk_metrics["total_risks_t1"], 1)
        shift_index = self.calculate_composite_shift_index(
            embedding_distance=emb_metrics["cosine_distance"],
            sentiment_delta_abs=abs(sent_metrics["deltas"]["net_tone_delta"]),
            risk_added_ratio=added_ratio,
            vocabulary_divergence=vocab_metrics["jaccard_distance"]
        )

        return {
            "company": company_name,
            "period_prior": year_t0,
            "period_current": year_t1,
            "composite_semantic_shift_index": shift_index,
            "shift_severity": "High Semantic Drift" if shift_index > 0.45 else (
                "Moderate Semantic Drift" if shift_index > 0.25 else "Low / Incremental Shift"
            ),
            "embedding_metrics": emb_metrics,
            "sentiment_metrics": sent_metrics,
            "risk_metrics": risk_metrics,
            "vocabulary_metrics": vocab_metrics,
            "economic_hypothesis_summary": (
                f"For {company_name} ({year_t0} -> {year_t1}), the computed Semantic Shift Index is {shift_index} "
                f"({risk_metrics['risk_expansion_count']} net risk items, Tone Drift: {sent_metrics['deltas']['net_tone_delta']}). "
                f"Under the TEEP 2026 model, high semantic shifts in MD&A/Risk sections historically correlate with "
                f"elevated operating margin volatility and information asymmetry premiums."
            )
        }
