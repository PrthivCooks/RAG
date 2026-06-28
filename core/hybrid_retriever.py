import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_DIR, 
    EMBEDDING_MODEL_NAME, 
    RERANK_MODEL_NAME, 
    RRF_K, 
    RRF_WEIGHTS, 
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    FINAL_TOP_K
)
from core.graph_builder import GraphBuilder
from core.fusion import reciprocal_rank_fusion
from core.reranker import CrossEncoderReranker
from core.summarizer import TextRankSummarizer

class HybridRetriever:
    """
    Main orchestration class for the hybrid retrieval engine.
    Integrates Sparse (BM25), Dense (FAISS), and Graph search,
    combines results via RRF, filters by metadata, reranks with a Cross-Encoder,
    and formats final output with highlighting, summaries, and graph suggestions.
    """
    def __init__(self, spacy_nlp):
        self.nlp = spacy_nlp
        
        # Load local embedding model
        print(f"Loading SentenceTransformer embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
        
        # Initialize sub-modules
        self.graph_builder = GraphBuilder(self.nlp)
        self.reranker = CrossEncoderReranker(RERANK_MODEL_NAME)
        self.summarizer = TextRankSummarizer(self.nlp)
        
        # In-memory storage for loaded indexes
        self.faiss_index = None
        self.bm25 = None
        self.bm25_chunk_ids = []
        self.chunks_lookup = {} # maps chunk_id -> chunk dictionary

    def load_indexes(self) -> bool:
        """
        Load indexes from disk.
        Returns True if successful, False otherwise.
        """
        try:
            # 1. Load Chunks Lookup
            chunks_path = INDEX_DIR / "chunks.pkl"
            if not chunks_path.exists():
                print(f"Error: Chunks lookup file not found at {chunks_path}")
                return False
            with open(chunks_path, "rb") as f:
                self.chunks_lookup = pickle.load(f)

            # 2. Load FAISS index
            faiss_path = INDEX_DIR / "faiss.idx"
            if not faiss_path.exists():
                print(f"Error: FAISS index file not found at {faiss_path}")
                return False
            self.faiss_index = faiss.read_index(str(faiss_path))

            # 3. Load BM25 index
            bm25_path = INDEX_DIR / "bm25.pkl"
            if not bm25_path.exists():
                print(f"Error: BM25 index file not found at {bm25_path}")
                return False
            with open(bm25_path, "rb") as f:
                self.bm25, self.bm25_chunk_ids = pickle.load(f)

            # 4. Load Knowledge Graph
            graph_path = INDEX_DIR / "graph.pkl"
            if not graph_path.exists():
                print(f"Error: Knowledge Graph file not found at {graph_path}")
                return False
            self.graph_builder.load_index(graph_path)

            print("All search indexes loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading indexes: {e}")
            return False

    def highlight_text(self, text: str, query: str) -> str:
        """
        Inject <mark> tags around terms in text that match words in query.
        Uses case-insensitive regex highlighting of non-stop words.
        """
        # Segment query into words using spaCy
        doc = self.nlp(query)
        words = [token.text.strip() for token in doc if not token.is_stop and not token.is_punct and len(token.text.strip()) > 2]
        
        # If no non-stop words, default to query text itself
        if not words and query.strip():
            words = [query.strip()]
            
        highlighted = text
        # Deduplicate and sort words by length descending to avoid nested replacement issues
        words = sorted(list(set(words)), key=len, reverse=True)
        
        for word in words:
            # Match word boundaries if possible
            pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
            # If word boundary doesn't match anything, try a simple search
            if not pattern.search(highlighted):
                pattern = re.compile(r'(' + re.escape(word) + r')', re.IGNORECASE)
                
            highlighted = pattern.sub(r'<mark style="background-color: #FFDE4D; color: #000000; padding: 2px 6px; border: 1.5px solid #000000; font-weight: 700; border-radius: 0px;">\1</mark>', highlighted)
            
        return highlighted

    def decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a complex query containing conjunctions into simpler sub-queries.
        Example: "GDPR policy and Vendor XYZ" -> ["GDPR policy", "Vendor XYZ"]
        """
        import re
        # Match splitting conjunctions
        split_markers = [" and ", " or ", " as well as ", " along with ", ", "]
        pattern = "|".join(re.escape(marker) for marker in split_markers)
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        
        sub_queries = []
        for part in parts:
            clean_part = part.strip()
            # Require a minimal query length
            if len(clean_part) > 2:
                sub_queries.append(clean_part)
                
        return sub_queries if sub_queries else [query]

    def get_relationship_walks(self, query: str, chunk_id: str) -> List[str]:
        """
        Traverses the Knowledge Graph to find connections between entities in the query and this chunk.
        """
        query_doc = self.nlp(query)
        # Find entities in the query that are also nodes in our graph
        query_entities = [self.graph_builder.clean_entity_name(ent.text) for ent in query_doc.ents]
        query_entities = [e for e in query_entities if self.graph_builder.graph.has_node(e)]
        
        # Fallback to noun chunks
        if not query_entities:
            query_entities = [self.graph_builder.clean_entity_name(chunk.text) for chunk in query_doc.noun_chunks]
            query_entities = [e for e in query_entities if self.graph_builder.graph.has_node(e)]
            
        walks = []
        if not query_entities:
            return walks
            
        for ent in query_entities:
            # Check if this entity has a direct connection to the chunk node
            if self.graph_builder.graph.has_edge(ent, chunk_id):
                # Trace relationship edges from this entity to neighboring entities in the graph
                neighbors = list(self.graph_builder.graph.successors(ent)) + list(self.graph_builder.graph.predecessors(ent))
                for n in neighbors:
                    if n != chunk_id and self.graph_builder.graph.nodes[n].get("type") == "entity":
                        if self.graph_builder.graph.has_edge(ent, n):
                            rel = self.graph_builder.graph.edges[ent, n].get("relation", "connected_to")
                            walks.append(f"{ent} -[{rel}]-> {n}")
                        elif self.graph_builder.graph.has_edge(n, ent):
                            rel = self.graph_builder.graph.edges[n, ent].get("relation", "connected_to")
                            walks.append(f"{n} -[{rel}]-> {ent}")
                            
        return list(set(walks))[:3]

    def search(
        self, 
        query: str, 
        k: int = FINAL_TOP_K, 
        alpha: float = None, 
        department_filter: str = None,
        use_decomposition: bool = True,
        use_parent_retrieval: bool = True
    ) -> Dict[str, Any]:
        """
        Executes hybrid search pipeline with optional Query Decomposition and Parent-Child retrieval.
        """
        # Ensure indexes are loaded
        if not self.chunks_lookup:
            if not self.load_indexes():
                return {"results": [], "summary": "Indexes not loaded. Please index documents first.", "see_also": []}

        # --- 1. LINGUISTIC QUERY DECOMPOSITION ---
        sub_queries = self.decompose_query(query) if use_decomposition else [query]

        sparse_accum = {}
        dense_accum = {}
        graph_accum = {}

        for q_sub in sub_queries:
            # BM25 Sparse Search
            tokenized_q = [t.text.lower() for t in self.nlp(q_sub)]
            sparse_raw = self.bm25.get_scores(tokenized_q)
            for idx, score in enumerate(sparse_raw):
                if score > 0.0:
                    cid = self.bm25_chunk_ids[idx]
                    sparse_accum[cid] = sparse_accum.get(cid, 0.0) + float(score)

            # FAISS Dense Search
            q_vector = self.embedding_model.encode(q_sub, convert_to_numpy=True)
            q_vector_norm = np.array([q_vector])
            faiss.normalize_L2(q_vector_norm)
            faiss_scores, faiss_indices = self.faiss_index.search(q_vector_norm, RETRIEVAL_TOP_K)
            for rank, idx in enumerate(faiss_indices[0]):
                if idx != -1:
                    cid = self.bm25_chunk_ids[idx]
                    score = float(faiss_scores[0][rank])
                    dense_accum[cid] = max(dense_accum.get(cid, 0.0), score)

            # Graph Retrieval
            graph_res = self.graph_builder.retrieve_by_graph(q_sub, RETRIEVAL_TOP_K)
            for cid, score in graph_res:
                graph_accum[cid] = graph_accum.get(cid, 0.0) + float(score)

        # Compile ranked results for RRF
        sparse_results = sorted(sparse_accum.items(), key=lambda x: x[1], reverse=True)[:RETRIEVAL_TOP_K]
        dense_results = sorted(dense_accum.items(), key=lambda x: x[1], reverse=True)[:RETRIEVAL_TOP_K]
        graph_results = sorted(graph_accum.items(), key=lambda x: x[1], reverse=True)[:RETRIEVAL_TOP_K]

        # --- 2. RECIPROCAL RANK FUSION ---
        retrieval_runs = {
            "sparse": sparse_results,
            "dense": dense_results,
            "graph": graph_results
        }
        
        # Configure fusion weights
        weights = RRF_WEIGHTS.copy()
        if alpha is not None:
            weights["dense"] = alpha
            weights["sparse"] = (1.0 - alpha) * 0.65
            weights["graph"] = (1.0 - alpha) * 0.35

        fused_candidates = reciprocal_rank_fusion(retrieval_runs, weights=weights, k=RRF_K)

        # --- 3. DEDUPLICATION & PARENT CONTEXT EXPANSION ---
        seen_parents = set()
        candidate_chunks = []
        
        for cand in fused_candidates:
            cid = cand["chunk_id"]
            chunk_data = self.chunks_lookup.get(cid)
            if not chunk_data:
                continue

            # Apply department filter
            if department_filter and department_filter != "All":
                chunk_dept = chunk_data["metadata"].get("department", "General")
                if chunk_dept.lower() != department_filter.lower():
                    continue

            # Check parent context
            parent_text = chunk_data.get("parent_text", chunk_data["text"])
            
            if use_parent_retrieval:
                parent_key = parent_text.strip().lower()
                if parent_key in seen_parents:
                    continue
                seen_parents.add(parent_key)

            enriched_chunk = chunk_data.copy()
            if use_parent_retrieval:
                enriched_chunk["text"] = parent_text

            enriched_chunk["fused_score"] = cand["combined_score"]
            enriched_chunk["retrieval_ranks"] = cand["ranks"]
            enriched_chunk["retrieval_scores"] = cand["scores"]
            candidate_chunks.append(enriched_chunk)

        # --- 4. RERANKING VIA CROSS-ENCODER ---
        candidates_to_rerank = candidate_chunks[:RERANK_TOP_K]
        reranked_chunks = self.reranker.rerank(query, candidates_to_rerank, top_k=k)

        if not reranked_chunks and candidate_chunks:
            reranked_chunks = candidate_chunks[:k]
            for r in reranked_chunks:
                r["rerank_score"] = r.get("fused_score", 0.0)

        # --- 5. HIGHLIGHTING & RELATION WALKS ---
        for chunk in reranked_chunks:
            chunk["highlighted_text"] = self.highlight_text(chunk["text"], query)
            # Trace graph relationship walks
            chunk["graph_walks"] = self.get_relationship_walks(query, chunk["chunk_id"])

        # --- 6. EXTRACTIVE SUMMARIZATION ---
        combined_text = "\n".join([chunk["text"] for chunk in reranked_chunks[:3]])
        summary = self.summarizer.summarize(combined_text)

        # --- 7. RECOMMENDATIONS ---
        retrieved_ids = [chunk["chunk_id"] for chunk in reranked_chunks]
        see_also = self.graph_builder.get_see_also_suggestions(query, retrieved_ids)

        return {
            "results": reranked_chunks,
            "summary": summary,
            "see_also": see_also,
            "sub_queries": sub_queries
        }
