from typing import List, Dict, Any, Tuple

def reciprocal_rank_fusion(
    retrieval_results: Dict[str, List[Tuple[str, float]]],
    weights: Dict[str, float] = None,
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines ranked results from multiple retrieval methods using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(doc) = Sum_{m in Methods} ( weight[m] * (1 / (k + rank_{m}(doc))) )
        
    Args:
        retrieval_results: Dict mapping search method name (e.g., 'sparse') to a list of (chunk_id, raw_score)
        weights: Dict mapping search method name to a float weight multiplier
        k: The constant parameter for RRF (default 60)
        
    Returns:
        List of dictionaries with fused scores and individual method ranks/scores for transparency.
    """
    if weights is None:
        weights = {method: 1.0 for method in retrieval_results.keys()}

    fused_scores = {}
    
    # Store raw scores and ranks for transparency
    chunk_details = {}

    for method, results in retrieval_results.items():
        weight = weights.get(method, 1.0)
        for rank, (chunk_id, raw_score) in enumerate(results, start=1):
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                chunk_details[chunk_id] = {
                    "chunk_id": chunk_id,
                    "ranks": {},
                    "scores": {}
                }
            
            # Add RRF contribution
            rrf_contribution = weight * (1.0 / (k + rank))
            fused_scores[chunk_id] += rrf_contribution
            
            # Record rank and score for transparency
            chunk_details[chunk_id]["ranks"][method] = rank
            chunk_details[chunk_id]["scores"][method] = float(raw_score)

    # Convert to list and sort by fused score descending
    fused_results = []
    for chunk_id, combined_score in fused_scores.items():
        details = chunk_details[chunk_id]
        details["combined_score"] = float(combined_score)
        
        # Fill in missing methods with None/0
        for method in retrieval_results.keys():
            if method not in details["ranks"]:
                details["ranks"][method] = None
                details["scores"][method] = 0.0
                
        fused_results.append(details)

    fused_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return fused_results
