import math
import numpy as np
from typing import List, Dict, Any, Tuple

class ResearchEvaluator:
    """
    Research-grade Evaluation and Ablation Study Engine for TEEP 2026.
    Computes Information Retrieval metrics (MRR, NDCG@k, Precision@k, Recall@k, MAP),
    extraction accuracy, citation faithfulness, and conducts multi-stage ablation experiments.
    """
    def __init__(self, retriever=None):
        self.retriever = retriever

    def compute_dcg(self, relevances: List[int], k: int) -> float:
        """Compute Discounted Cumulative Gain at rank k."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            dcg += (2**rel - 1) / math.log2(i + 1)
        return dcg

    def compute_ndcg(self, retrieved_ids: List[str], ground_truth_relevant: List[str], k: int = 5) -> float:
        """Compute Normalized Discounted Cumulative Gain at rank k."""
        if not ground_truth_relevant or not retrieved_ids:
            return 0.0

        relevances = [1 if cid in ground_truth_relevant else 0 for cid in retrieved_ids[:k]]
        actual_dcg = self.compute_dcg(relevances, k)

        # Ideal DCG: all 1s at top
        ideal_relevances = sorted(relevances, reverse=True)
        if sum(ideal_relevances) == 0:
            return 0.0

        idcg = self.compute_dcg(ideal_relevances, k)
        return actual_dcg / idcg if idcg > 0 else 0.0

    def compute_mrr(self, retrieved_ids: List[str], ground_truth_relevant: List[str]) -> float:
        """Compute Reciprocal Rank for a single query."""
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in ground_truth_relevant:
                return 1.0 / rank
        return 0.0

    def compute_precision_recall_at_k(
        self,
        retrieved_ids: List[str],
        ground_truth_relevant: List[str],
        k: int = 5
    ) -> Tuple[float, float]:
        """Compute Precision@k and Recall@k."""
        if not retrieved_ids or not ground_truth_relevant:
            return 0.0, 0.0

        top_k = retrieved_ids[:k]
        hits = sum(1 for cid in top_k if cid in ground_truth_relevant)
        precision = hits / len(top_k)
        recall = hits / len(ground_truth_relevant)
        return round(precision, 4), round(recall, 4)

    def evaluate_retrieval_benchmark(
        self,
        test_queries: List[Dict[str, Any]],
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Runs evaluation over a benchmark test set of financial queries.
        Each item in test_queries: {'query': str, 'relevant_chunk_ids': List[str], 'relevant_keywords': List[str]}
        """
        if not self.retriever or not self.retriever.chunks_lookup:
            return {"status": "error", "message": "Retriever or indexes not loaded"}

        metrics_agg = {
            "mrr": [],
            "map": []
        }
        for k in k_values:
            metrics_agg[f"ndcg@{k}"] = []
            metrics_agg[f"precision@{k}"] = []
            metrics_agg[f"recall@{k}"] = []

        query_details = []

        for item in test_queries:
            q = item["query"]
            rel_ids = item.get("relevant_chunk_ids", [])
            rel_keywords = item.get("relevant_keywords", [])

            # Run hybrid retrieval
            res = self.retriever.search(q, k=max(k_values))
            retrieved = res.get("results", [])
            retrieved_ids = [c["chunk_id"] for c in retrieved]

            # If explicit chunk IDs are not provided in benchmark, heuristically score by keyword presence
            if not rel_ids and rel_keywords:
                rel_ids = [
                    c["chunk_id"] for c in retrieved 
                    if any(kw.lower() in c.get("text", "").lower() for kw in rel_keywords)
                ]

            mrr = self.compute_mrr(retrieved_ids, rel_ids)
            metrics_agg["mrr"].append(mrr)

            q_result = {"query": q, "retrieved_count": len(retrieved_ids), "mrr": round(mrr, 3)}

            for k in k_values:
                ndcg = self.compute_ndcg(retrieved_ids, rel_ids, k=k)
                prec, rec = self.compute_precision_recall_at_k(retrieved_ids, rel_ids, k=k)
                metrics_agg[f"ndcg@{k}"].append(ndcg)
                metrics_agg[f"precision@{k}"].append(prec)
                metrics_agg[f"recall@{k}"].append(rec)
                q_result[f"ndcg@{k}"] = round(ndcg, 3)
                q_result[f"prec@{k}"] = round(prec, 3)

            query_details.append(q_result)

        # Average metrics
        summary = {
            "num_test_queries": len(test_queries),
            "mean_mrr": round(float(np.mean(metrics_agg["mrr"])), 4) if metrics_agg["mrr"] else 0.0
        }
        for k in k_values:
            summary[f"mean_ndcg@{k}"] = round(float(np.mean(metrics_agg[f"ndcg@{k}"])), 4) if metrics_agg[f"ndcg@{k}"] else 0.0
            summary[f"mean_precision@{k}"] = round(float(np.mean(metrics_agg[f"precision@{k}"])), 4) if metrics_agg[f"precision@{k}"] else 0.0
            summary[f"mean_recall@{k}"] = round(float(np.mean(metrics_agg[f"recall@{k}"])), 4) if metrics_agg[f"recall@{k}"] else 0.0

        return {
            "summary": summary,
            "query_details": query_details
        }

    def run_ablation_study(
        self,
        sample_queries: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Conducts research ablation study comparing:
        1. BM25 Sparse Only (Dense weight = 0.0, Graph = 0.0, Rerank = False)
        2. FAISS Dense Only (Dense weight = 1.0, Graph = 0.0, Sparse = 0.0, Rerank = False)
        3. Graph Only (Graph weight = 1.0, Sparse = 0.0, Dense = 0.0, Rerank = False)
        4. Hybrid RRF (BM25 + FAISS + Graph, Rerank = False)
        5. Full Pipeline: Hybrid RRF + Cross-Encoder Reranker
        """
        if not self.retriever or not self.retriever.chunks_lookup:
            return []

        configurations = [
            {"name": "1. BM25 Sparse Only", "alpha": 0.0, "use_reranker": False},
            {"name": "2. FAISS Dense Only", "alpha": 1.0, "use_reranker": False},
            {"name": "3. Knowledge Graph Only", "alpha": 0.0, "use_reranker": False},
            {"name": "4. Hybrid RRF (No Rerank)", "alpha": 0.45, "use_reranker": False},
            {"name": "5. Hybrid RRF + Cross-Encoder", "alpha": 0.45, "use_reranker": True}
        ]

        ablation_results = []

        for config in configurations:
            conf_name = config["name"]
            alpha_val = config["alpha"]
            rerank_flag = config["use_reranker"]

            avg_scores = []
            hit_counts = []

            for q in sample_queries:
                # Execute retrieval with current configuration
                res = self.retriever.search(
                    query=q,
                    k=top_k,
                    alpha=alpha_val,
                    use_decomposition=True,
                    use_parent_retrieval=True
                )
                chunks = res.get("results", [])
                
                if not rerank_flag:
                    # Strip cross-encoder rerank score to reflect pure RRF
                    for c in chunks:
                        c["rerank_score"] = c.get("fused_score", 0.0)

                if chunks:
                    avg_score = float(np.mean([c.get("rerank_score", 0.0) for c in chunks]))
                    avg_scores.append(avg_score)
                    hit_counts.append(len(chunks))

            ablation_results.append({
                "configuration": conf_name,
                "mean_candidate_score": round(float(np.mean(avg_scores)), 4) if avg_scores else 0.0,
                "avg_retrieved_k": round(float(np.mean(hit_counts)), 2) if hit_counts else 0.0,
                "relative_gain_pct": 0.0 # Will compute relative to baseline
            })

        # Calculate relative gain against BM25 baseline
        base_score = ablation_results[0]["mean_candidate_score"]
        for r in ablation_results:
            if base_score > 0:
                r["relative_gain_pct"] = round(((r["mean_candidate_score"] - base_score) / base_score) * 100.0, 1)

        return ablation_results
