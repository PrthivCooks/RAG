# Enterprise Hybrid Knowledge Engine (LLM-Free Advanced RAG)

A production-grade, local-first document retrieval engine that provides high-precision answers, term highlights, and entity connections for enterprise datasets without using any generative LLM APIs (no OpenAI, no Ollama, no LangChain, and no external calls).

---

## Architecture Overview

This search engine integrates three distinct retrieval paradigms and combines them to perform precise semantic searches:

1. **Sparse Indexing (BM25)**: Utilizes keyword term frequencies and statistics (`rank_bm25`). Excellent for exact IDs, compliance numbers, code names, and specific acronyms.
2. **Dense Indexing (FAISS)**: Maps semantic vector embeddings computed with `sentence-transformers/all-MiniLM-L6-v2`. Computes query similarity in dense vector space (cosine similarity), finding matches using conceptual similarity even with synonyms.
3. **Graph Indexing (Knowledge Graph)**: Employs spaCy (`en_core_web_sm`) to run Named Entity Recognition (NER) and syntactic dependency parsing (Subject-Verb-Object). Builds a graph using `NetworkX` linking entity relationships and text chunks. Resolves multi-hop context matches and recommends related topics.

```
       [User Query]
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  BM25    FAISS   Graph (NetworkX)
  (Term) (Vector) (Entity Rels)
    │       │       │
    └───────┼───────┘
            ▼
    Reciprocal Rank Fusion (RRF)
            ▼
    Metadata Filtering (Department)
            ▼
  Cross-Encoder Reranker (ms-marco)
            ▼
  ┌─────────┴─────────┐
  ▼                   ▼
Extractive Summary  "See Also" Pills
  (TextRank)        (KG Entities)
```

### Fusion & Post-Processing

- **Reciprocal Rank Fusion (RRF)**: Implement from scratch. Normalizes and blends the raw ranks of the three retrieval streams. Weights can be adjusted dynamically in the UI.
- **Cross-Encoder Reranking**: Evaluates top candidate chunks together with the user query using `cross-encoder/ms-marco-MiniLM-L-6-v2` to predict high-confidence relevance.
- **Extractive Summarization (TextRank)**: Uses TF-IDF cosine similarity of sentences from top results to build a graph, applies NetworkX PageRank, and extracts the top 3 most informative sentences chronologically to build a summarization card.
- **Snippet Highlighting**: Case-insensitive text parsing highlighting search terms with HTML `<mark>` tags.
- **"See Also" Suggestions**: Traverses the Knowledge Graph connections of retrieved results to recommend related corporate entities or files.

---

## Project Structure

```text
enterprise-hybrid-knowledge-engine/
├── data/                    # PDF, DOCX, and TXT source documents
├── indexes/                 # Serialized indexes (FAISS index, BM25, KG graph pickle)
├── core/
│   ├── __init__.py
│   ├── document_processor.py# Orchestrates text extraction and chunking
│   ├── hybrid_retriever.py  # Query pipeline entrypoint
│   ├── graph_builder.py     # spaCy entity/SVO extraction and NetworkX indexer
│   ├── reranker.py          # Cross-encoder candidate re-ranking
│   ├── fusion.py            # Reciprocal Rank Fusion implementation
│   └── summarizer.py        # TextRank extractive sentence summary builder
├── utils/
│   ├── chunking.py          # Custom recursive & semantic similarity chunking
│   └── helpers.py           # Document extractors (pypdf, python-docx)
├── app.py                   # Streamlit UI dashboard
├── ingest.py                # Ingestion execution script
├── requirements.txt         # Required Python packages
├── Dockerfile               # Container deployment configuration
├── README.md                # System documentation
└── config.py                # Tuning, weights, and path parameters
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- C++ Build Tools (required by FAISS and spaCy compiler step on Windows, though pre-built binary wheels are usually installed automatically).

### 1. Installation
Clone or navigate to the directory and run:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/macOS

# Install packages
pip install -r requirements.txt
```

*Note: The spaCy model (`en_core_web_sm`) is automatically downloaded when running `ingest.py` or starting the app.*

### 2. Add Documents
Place your corporate documents (`.pdf`, `.docx`, `.txt`) in the `data/` folder.
*(If you run ingestion with an empty `data/` folder, the script will automatically populate it with 3 sample files covering retention, refund, and vendor compliance for demonstration purposes).*

### 3. Build Indexes
Run the ingestion pipeline to extract text, chunk semantically, and generate FAISS, BM25, and Graph indexes:

```bash
python ingest.py
```

This will output the serialized indexes into the `indexes/` folder:
- `faiss.idx` (Flat Inner Product vector space)
- `bm25.pkl` (BM25 token corpus)
- `graph.pkl` (NetworkX Digraph structure)
- `chunks.pkl` (Text chunk database metadata lookup)

### 4. Launch UI
Run the Streamlit frontend dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker Deployment

You can build and deploy the application as a Docker container:

```bash
# Build the image
docker build -t hybrid-rag-engine .

# Run the container
docker run -p 8501:8501 hybrid-rag-engine
```

Access the application at `http://localhost:8501`.

---

## Tailoring Parameters (`config.py`)

You can modify settings in `config.py` to optimize performance:
- `SEMANTIC_SPLIT_THRESHOLD`: Adjust the sensitivity of the semantic chunker (default: `0.55`).
- `RRF_WEIGHTS`: Customize the contribution ratio between Sparse (`sparse`), Dense (`dense`), and Graph (`graph`) retrieval results.
- `RERANK_TOP_K`: Number of elements to pass to the heavy Cross-Encoder reranker.
- `SUMMARY_SENTENCES_COUNT`: Number of sentences in the extractive TextRank summary.

---

## Future Improvements

1. **Sub-chunk/Hierarchical Retrieval**: Retrieve small chunks for fine-grained matching but expand to their parent block context for Cross-Encoder scoring and summarization.
2. **Hybrid Embeddings**: Combine a dense transformer (like MiniLM) with a trained sparse vector embedding model (like Splade) inside FAISS for unified indexing.
3. **Graph Embeddings / GCN**: Apply graph neural networks (e.g., node2vec) to calculate vector representations of KG nodes for entity matching in dense space.
4. **Enhanced Entity Resolution**: Add clustering or TF-IDF matching to group spelling variations (e.g. "Vendor XYZ Inc", "XYZ Corp", "Vendor XYZ") into a single canonical Graph node.
