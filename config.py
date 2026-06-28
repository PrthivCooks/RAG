import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 600       # Target chunk size in characters
DEFAULT_CHUNK_OVERLAP = 120    # Overlap between chunks
SEMANTIC_SPLIT_THRESHOLD = 0.55 # Cosine similarity threshold below which a split is triggered

# Retrieval Settings
RETRIEVAL_TOP_K = 25  # Number of raw chunks to retrieve per search method

# RRF Configuration
RRF_K = 60            # Constant for Reciprocal Rank Fusion
# Weights for sparse, dense, and graph-based retrieval scores
RRF_WEIGHTS = {
    "sparse": 0.35,   # Weight for BM25
    "dense": 0.45,    # Weight for Sentence Embeddings
    "graph": 0.20     # Weight for Knowledge Graph Entity matches
}

# Reranker Settings
RERANK_TOP_K = 15     # Number of candidate chunks to pass to the cross-encoder
FINAL_TOP_K = 5       # Final number of results to return to the user

# Extractive Summarization Configuration
SUMMARY_SENTENCES_COUNT = 3  # Number of sentences to extract for the summary
