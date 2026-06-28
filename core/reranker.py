from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Reranks candidate chunks using a local Cross-Encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2).
    Cross-encoders score the query and document chunk jointly, yielding highly accurate relevance scores.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Lazy load the Cross-Encoder model to save memory during startup."""
        if self.model is None:
            print(f"Loading Cross-Encoder model: {self.model_name}...")
            # Automatically uses GPU if available
            self.model = CrossEncoder(self.model_name)
            print("Cross-Encoder model loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunk dictionaries based on query relevance.
        
        Args:
            query: The search query string
            candidates: List of chunk dictionaries containing a "text" key
            top_k: Number of reranked results to return
            
        Returns:
            The sorted list of candidate chunks with a "rerank_score" added to each.
        """
        if not candidates:
            return []
            
        self.load_model()
        
        # Prepare inputs: list of [query, chunk_text] pairs
        pairs = [[query, candidate["text"]] for candidate in candidates]
        
        # Predict relevance scores
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Add score to each candidate
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)
            
        # Sort candidates by rerank score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return candidates[:top_k]
