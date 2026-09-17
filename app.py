import json
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime

from config import (
    DATA_DIR, INDEX_DIR, RRF_WEIGHTS, FINAL_TOP_K, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    SEC_SECTIONS, FINANCIAL_METRICS_SCHEMA, LOUGHRAN_MCDONALD_LEXICON
)
from core.hybrid_retriever import HybridRetriever
from core.semantic_shift import SemanticShiftAnalyzer
from core.financial_extractor import FinancialMetricExtractor
from core.financial_qa import EvidenceGroundedFinancialQA
from core.evaluator import ResearchEvaluator
from finetune.prepare_finetuning import FinancialFineTuningPreparer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Enterprise Hybrid Knowledge Engine | TEEP 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end styling
st.markdown("""
<style>
    /* Force 0 rounded corners globally */
    * {
        border-radius: 0px !important;
    }

    /* Hide Streamlit header (Deploy button, options menu) */
    header, [data-testid="stHeader"] {
        display: none !important;
    }

    /* Import Space Grotesk Font for Neo-Brutalism */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    /* Global typography overrides */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Elegant Custom Card */
    .result-card {
        background-color: #121318;
        border: 3px solid #FFFFFF;
        border-radius: 0px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 5px 5px 0px 0px #2563EB;
        transition: transform 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    
    .result-card:hover {
        transform: translate(-3px, -3px);
        box-shadow: 8px 8px 0px 0px #FFDE4D;
        border-color: #FFDE4D;
    }

    /* Sub-headers and badges */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .doc-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    
    .dept-badge {
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #FFDE4D;
        color: #000000;
        padding: 4px 10px;
        border-radius: 0px;
        border: 2px solid #000000;
        text-transform: uppercase;
    }

    .meta-line {
        font-size: 0.8rem;
        color: #FFFFFF;
        opacity: 0.7;
        margin-bottom: 12px;
        font-family: monospace;
    }

    .snippet-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #FFFFFF;
        opacity: 0.95;
        margin-bottom: 16px;
    }

    /* Metrics and transparency stats */
    .stats-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 20px;
        border-top: 2px solid #FFFFFF;
        padding-top: 12px;
        font-size: 0.8rem;
    }

    .stat-box {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .stat-label {
        font-weight: 700;
        opacity: 0.8;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
    }

    .stat-value {
        font-weight: 700;
        color: #FFFFFF;
        font-family: monospace;
    }
    
    .stat-value-primary {
        font-weight: 700;
        color: #000000;
        background-color: #FFDE4D;
        padding: 2px 6px;
        border-radius: 0px;
        border: 1.5px solid #000000;
        font-family: monospace;
    }

    /* Summary Card Styling */
    .summary-card {
        background-color: #1E1B4B;
        border: 3px solid #FFFFFF;
        border-radius: 0px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 6px 6px 0px 0px #FFDE4D;
    }

    /* Suggesion Pills */
    .suggestion-pill {
        display: inline-block;
        padding: 8px 16px;
        background-color: #121318;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 0px;
        font-size: 0.85rem;
        font-weight: 700;
        cursor: pointer;
        margin-right: 10px;
        margin-bottom: 10px;
        box-shadow: 3px 3px 0px 0px #2563EB;
        transition: all 0.1s ease-in-out;
    }
    
    .suggestion-pill:hover {
        border-color: #000000;
        background-color: #FFDE4D;
        color: #000000;
        box-shadow: 1px 1px 0px 0px #000000;
        transform: translate(2px, 2px);
    }

    /* Hover Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 2px dotted #FFDE4D;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #121318;
        color: #FFFFFF;
        text-align: left;
        border: 2px solid #FFFFFF;
        border-radius: 0px;
        padding: 12px 16px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.2s, visibility 0.2s;
        font-size: 0.78rem;
        line-height: 1.5;
        box-shadow: 4px 4px 0px 0px #2563EB;
    }

    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #FFFFFF transparent transparent transparent;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* --- Override Streamlit Elements for Neo-Brutalism --- */
    
    /* Text Inputs, Selectboxes, Number Inputs */
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input, 
    div[data-baseweb="select"] {
        border: 3px solid #FFFFFF !important;
        background-color: #121318 !important;
        color: #FFFFFF !important;
        box-shadow: 4px 4px 0px 0px #2563EB !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.1s ease-in-out !important;
    }
    div[data-testid="stTextInput"] input:focus, 
    div[data-testid="stNumberInput"] input:focus {
        border-color: #FFDE4D !important;
        box-shadow: 4px 4px 0px 0px #FFDE4D !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 4px 4px 0px 0px #FFDE4D !important;
        transition: all 0.1s ease-in-out !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div.stButton > button:hover {
        background-color: #FFDE4D !important;
        color: #000000 !important;
        border-color: #000000 !important;
        box-shadow: 2px 2px 0px 0px #000000 !important;
        transform: translate(2px, 2px) !important;
    }
    div.stButton > button:active {
        transform: translate(4px, 4px) !important;
        box-shadow: 0px 0px 0px 0px !important;
    }

    /* File Uploader wrapper */
    div[data-testid="stFileUploader"] {
        background-color: #121318 !important;
        border: 3px dashed #FFFFFF !important;
        box-shadow: 5px 5px 0px 0px #2563EB !important;
        padding: 20px !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #050507 !important;
        border-right: 3px solid #FFFFFF !important;
    }

    /* Accordions / Expanders */
    div[data-testid="stExpander"] {
        background-color: #121318 !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 4px 4px 0px 0px #2563EB !important;
        margin-bottom: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get model instance
@st.cache_resource
def get_retriever():
    """Initializes and caches the hybrid search coordinator."""
    import spacy
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    retriever = HybridRetriever(nlp)
    # Attempt to load saved indexes
    retriever.load_indexes()
    return retriever

# Load Retriever
try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"Failed to initialize search engine: {e}")
    retriever = None

# Check if index files exist on disk
index_files = [INDEX_DIR / "faiss.idx", INDEX_DIR / "bm25.pkl", INDEX_DIR / "graph.pkl", INDEX_DIR / "chunks.pkl"]
indexes_exist = all(f.exists() for f in index_files) and retriever is not None and len(retriever.chunks_lookup) > 0

# Helper function to perform in-memory ingestion
def run_in_memory_ingestion(uploaded_files, clear_existing=True):
    """
    Saves uploaded files to disk and builds all indexes in-memory.
    """
    import pickle
    import faiss
    import time
    from rank_bm25 import BM25Okapi
    from core.document_processor import DocumentProcessor
    from core.graph_builder import GraphBuilder
    
    # 1. Directory maintenance
    if clear_existing:
        # Clear DATA_DIR
        for item in DATA_DIR.glob("*"):
            if item.is_file():
                item.unlink()
        # Clear INDEX_DIR
        for item in INDEX_DIR.glob("*"):
            if item.is_file():
                item.unlink()
    else:
        # If not clearing, we still clean old indexes to write fresh ones
        for item in INDEX_DIR.glob("*"):
            if item.is_file():
                item.unlink()

    # 2. Save uploaded file contents or copy disk sample files
    saved_paths = []
    for item_file in uploaded_files:
        if isinstance(item_file, (str, Path)):
            p = Path(item_file)
            target_path = DATA_DIR / p.name
            with open(p, "rb") as sf, open(target_path, "wb") as df:
                df.write(sf.read())
            saved_paths.append(target_path)
        else:
            target_path = DATA_DIR / item_file.name
            with open(target_path, "wb") as f:
                f.write(item_file.getbuffer())
            saved_paths.append(target_path)
        
    # Create streamlit progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Document Processor loading model references...")
        progress_bar.progress(10)
        
        if retriever is None:
            st.error("Failed to compile indexes: the search engine retriever could not be initialized.")
            time.sleep(3)
            progress_bar.empty()
            status_text.empty()
            return False
            
        processor = DocumentProcessor(retriever.embedding_model, retriever.nlp)
        
        status_text.text("Extracting text contents, detecting department metadata, and chunking...")
        progress_bar.progress(25)
        
        chunks = []
        for path in saved_paths:
            file_chunks = processor.process_file(
                path,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                use_semantic=True
            )
            chunks.extend(file_chunks)
            
        if not chunks:
            st.error("Extraction yielded 0 text chunks. Ensure the uploaded files contain readable text.")
            time.sleep(3)
            progress_bar.empty()
            status_text.empty()
            return False
            
        progress_bar.progress(45)
        status_text.text(f"Creating lookup registry ({len(chunks)} chunks)...")
        chunks_lookup = {c["chunk_id"]: c for c in chunks}
        with open(INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_lookup, f)
            
        progress_bar.progress(60)
        status_text.text("Tokenizing and compiling BM25 Sparse Index...")
        tokenized_corpus = []
        chunk_ids = []
        for c in chunks:
            tokens = [t.text.lower() for t in retriever.nlp(c["text"])]
            tokenized_corpus.append(tokens)
            chunk_ids.append(c["chunk_id"])
        bm25 = BM25Okapi(tokenized_corpus)
        with open(INDEX_DIR / "bm25.pkl", "wb") as f:
            pickle.dump((bm25, chunk_ids), f)
            
        progress_bar.progress(75)
        status_text.text("Computing dense Sentence-Transformer embeddings & indexing with FAISS...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = retriever.embedding_model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.idx"))
        
        progress_bar.progress(90)
        status_text.text("Running Entity Extraction & compiling Knowledge Graph relations...")
        graph_builder = GraphBuilder(retriever.nlp)
        for c in chunks:
            graph_builder.add_chunk_to_graph(c["chunk_id"], c["text"], c["metadata"])
        graph_builder.save_index(INDEX_DIR / "graph.pkl")
        
        progress_bar.progress(100)
        status_text.text("Ingestion complete! Reloading indexing engines...")
        time.sleep(1)
        
        # Clear progress UI elements
        progress_bar.empty()
        status_text.empty()
        
        # Force cache refresh and reload in-memory search structures
        st.cache_resource.clear()
        
        return True
    except Exception as e:
        st.error(f"Failed to compile indexes: {e}")
        progress_bar.empty()
        status_text.empty()
        return False

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("Engine Controls")
    
    # Ingestion Status Section
    st.header("Index Status")
    if indexes_exist:
        st.success("Search Indexes: Loaded")
        
        # Option to clear the existing dataset
        if st.button("Clear All Documents & Indexes", use_container_width=True):
            # Delete data/ and indexes/ files
            for item in DATA_DIR.glob("*"):
                if item.is_file():
                    item.unlink()
            for item in INDEX_DIR.glob("*"):
                if item.is_file():
                    item.unlink()
            st.cache_resource.clear()
            st.success("Cleaned all search indexes.")
            st.rerun()
    else:
        st.warning("Search Indexes: Empty")
        st.info("Upload files to build your knowledge engine.")

    st.markdown("---")

    # Retrieval Settings
    st.header("Search Parameters")
    
    alpha = st.slider(
        "Semantic Search Weight (Dense α)",
        min_value=0.0,
        max_value=1.0,
        value=float(RRF_WEIGHTS["dense"]),
        step=0.05,
        help="Higher alpha favors neural meaning (FAISS). Lower alpha favors keyword match (BM25) and Knowledge Graph."
    )
    
    # Scan departments from loaded index
    departments = ["All"]
    if indexes_exist and retriever and retriever.chunks_lookup:
        depts_in_data = sorted(list(set(
            chunk["metadata"].get("department", "General") 
            for chunk in retriever.chunks_lookup.values()
        )))
        departments.extend(depts_in_data)
    else:
        departments.extend(["Security", "Finance", "Operations", "Legal", "Engineering"])

    selected_dept = st.selectbox(
        "Filter by Department",
        options=departments,
        index=0
    )

    # Dynamic Financial Metadata Filters (Company, Fiscal Year, SEC Section)
    companies = ["All"]
    years = ["All"]
    sec_sections_list = ["All", "Item 1 (Business)", "Item 1A (Risk Factors)", "Item 7 (MD&A)", "Item 8 (Financial Statements)"]

    if indexes_exist and retriever and retriever.chunks_lookup:
        comps_in_data = sorted(list(set(
            chunk["metadata"].get("company", "")
            for chunk in retriever.chunks_lookup.values()
            if chunk["metadata"].get("company") and chunk["metadata"].get("company") != "Unknown Company"
        )))
        if comps_in_data:
            companies.extend(comps_in_data)

        years_in_data = sorted(list(set(
            str(chunk["metadata"].get("fiscal_year", ""))
            for chunk in retriever.chunks_lookup.values()
            if chunk["metadata"].get("fiscal_year") and chunk["metadata"].get("fiscal_year") != "N/A"
        )))
        if years_in_data:
            years.extend(years_in_data)

    selected_company = st.selectbox("Filter by Company / Ticker", options=companies, index=0)
    selected_year = st.selectbox("Filter by Fiscal Year", options=years, index=0)
    selected_sec = st.selectbox("Filter by SEC Section", options=sec_sections_list, index=0)
    
    num_results = st.number_input(
        "Max Results",
        min_value=1,
        max_value=20,
        value=FINAL_TOP_K
    )
    
    st.markdown("---")
    st.header("Advanced RAG Features")
    use_decomposition = st.checkbox(
        "Enable Query Decomposition",
        value=True,
        help="Linguistically split complex coordinate queries into simpler sub-queries and combine scores."
    )
    use_parent_retrieval = st.checkbox(
        "Enable Parent-Child Expansion",
        value=True,
        help="Match precise small child chunks but display the larger parent paragraph for summarization and results."
    )
    enable_fin_expansion = st.checkbox(
        "Financial Terminology Expansion",
        value=True,
        help="Expands queries with accounting synonyms and SEC financial taxonomy terms."
    )

    st.markdown("---")
    
    # Search stats summary
    if indexes_exist and retriever and retriever.chunks_lookup:
        st.header("Index Statistics")
        total_chunks = len(retriever.chunks_lookup)
        total_docs = len(set(chunk["metadata"]["doc_id"] for chunk in retriever.chunks_lookup.values()))
        num_nodes = retriever.graph_builder.graph.number_of_nodes()
        num_edges = retriever.graph_builder.graph.number_of_edges()
        
        st.metric(label="Documents Indexed", value=total_docs)
        st.metric(label="Total Text Chunks", value=total_chunks)
        st.metric(label="Knowledge Graph Nodes", value=num_nodes)
        st.metric(label="Knowledge Graph Relations", value=num_edges)

# --- MAIN PAGE LAYOUT ---
st.title("Enterprise Hybrid Knowledge Engine")
st.markdown(
    "*Advanced RAG system combining sparse, dense, and graph retrieval to deliver precise answers without generative LLMs.*"
)

# Active State Check: Force tabs to render always by bypassing the empty index template gate.
# The onboarding uploader and guide are now integrated directly inside the Home / About tab.
if False:
    st.markdown("Onboarding")
else:
    # All tabs rendered with Neo-Brutalist styling
    tab_about, tab_search, tab_graph, tab_filings, tab_semantic_shift, tab_metrics, tab_qa, tab_eval = st.tabs([
        "Home / About",
        "Search Engine",
        "Knowledge Graph Visualizer",
        "SEC Annual Reports",
        "Semantic Shift & Narrative Drift",
        "Financial Metrics & Audit",
        "Evidence-Grounded Q&A",
        "Research Evaluation & Ablation"
    ])
    
    with tab_about:
        # Title banner card
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 24px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #2563EB;">
            <h2 style="color: #FFDE4D; margin-top: 0; font-weight: 700; letter-spacing: -0.03em; font-family: 'Space Grotesk', sans-serif;">ENTERPRISE HYBRID KNOWLEDGE ENGINE</h2>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #FFFFFF; margin-bottom: 0; font-family: 'Space Grotesk', sans-serif;">
                Welcome to the <strong>LLM-Free Hybrid Knowledge & Retrieval Engine</strong>. This platform is designed to query, explore, and analyze complex document sets with mathematical certainty, offline accessibility, and zero hallucination risks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Workspace stats if available
        if indexes_exist:
            total_chunks = len(retriever.chunks_lookup)
            total_docs = len(set(chunk["metadata"]["doc_id"] for chunk in retriever.chunks_lookup.values()))
            num_nodes = retriever.graph_builder.graph.number_of_nodes()
            num_edges = retriever.graph_builder.graph.number_of_edges()
            
            st.markdown(f"""
            <div style="background-color: #1E1B4B; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D;">
                <h4 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">ACTIVE INDEX METRICS</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center; margin-top: 15px;">
                    <div style="background: #121318; border: 2px solid #FFFFFF; padding: 10px;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; font-family: monospace;">{total_docs}</div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Documents</div>
                    </div>
                    <div style="background: #121318; border: 2px solid #FFFFFF; padding: 10px;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; font-family: monospace;">{total_chunks}</div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Text Chunks</div>
                    </div>
                    <div style="background: #121318; border: 2px solid #FFFFFF; padding: 10px;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; font-family: monospace;">{num_nodes}</div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Graph Nodes</div>
                    </div>
                    <div style="background: #121318; border: 2px solid #FFFFFF; padding: 10px;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; font-family: monospace;">{num_edges}</div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Graph Relations</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Onboarding uploader directly on the Home / About page!
            st.markdown("""
            <div style="background-color: rgba(37, 99, 235, 0.05); border: 3px dashed #FFFFFF; padding: 30px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D;">
                <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">GET STARTED: INITIALIZE SEARCH ENGINE</h3>
                <p style="font-size: 1.05rem; opacity: 0.9; line-height: 1.6; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; margin-bottom: 15px;">
                    This system operates completely locally without preloaded documents. To begin, upload your documents (.pdf, .docx, or .txt) below. 
                    The system will instantly chunk, extract entities, construct a knowledge graph, generate embeddings, and build the indices in-memory.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload document(s) to index:",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="main_uploader_empty"
            )
            
            if uploaded_files:
                if st.button("Process and Build Indexes", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        if run_in_memory_ingestion(uploaded_files, clear_existing=True):
                            st.success("Documents indexed! Reloading environment...")
                            st.rerun()

        # Core philosophy and hybrid retrieval columns
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px;">
            <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; box-shadow: 5px 5px 0px 0px #FFDE4D; height: 100%;">
                <h3 style="color: #FFFFFF; margin-top: 0; font-weight: 700; border-bottom: 2px solid #FFFFFF; padding-bottom: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">Core Philosophy</h3>
                <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; font-family: 'Space Grotesk', sans-serif;">
                    Generative Large Language Models (LLMs) are powerful but introduce <strong>hallucination risks, security/privacy concerns, heavy hosting costs, and high latencies</strong>. 
                    <br/><br/>
                    This system is built entirely on the principle of <strong>Extractive Hybrid Retrieval</strong>. It evaluates physical text matches, conceptual embeddings, and local syntactic relationship graphs to find exactly what exists in your documents—with 100% verifiability and no generative fabrication.
                </p>
            </div>
            <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; box-shadow: 5px 5px 0px 0px #FFDE4D; height: 100%;">
                <h3 style="color: #FFFFFF; margin-top: 0; font-weight: 700; border-bottom: 2px solid #FFFFFF; padding-bottom: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">Hybrid Retrieval Mechanics</h3>
                <p style="font-size: 0.95rem; line-height: 1.5; color: #FFFFFF; opacity: 0.9; font-family: 'Space Grotesk', sans-serif;">
                    Queries are evaluated in parallel across three specialized local indexing layers, combined using <strong>Reciprocal Rank Fusion (RRF)</strong>:
                </p>
                <ul style="font-size: 0.9rem; line-height: 1.5; color: #FFFFFF; opacity: 0.9; padding-left: 20px; font-family: 'Space Grotesk', sans-serif;">
                    <li><strong>Sparse Retrieval (Rank-BM25):</strong> Employs term-frequency/inverse-document-frequency keyword relevance for precise terminology, symbols, and values.</li>
                    <li><strong>Dense Retrieval (FAISS + MiniLM-L6-v2):</strong> Encodes chunk contents into a dense vector space to find semantic/conceptual equivalents.</li>
                    <li><strong>Knowledge Graph (SpaCy + NetworkX):</strong> Matches Named Entities (NER) and Subject-Verb-Object (SVO) tuples to map semantic connections.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Pipeline flowchart card (Terminal look)
        st.markdown("""
        <div style="background-color: #050507; border: 3px solid #FFFFFF; font-family: monospace; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #2563EB; color: #00FF66;">
            <h3 style="color: #FFFFFF; margin-top: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700; border-bottom: 2px solid #FFFFFF; padding-bottom: 8px; letter-spacing: -0.02em;">Under the Hood: Pipeline Architecture</h3>
            <pre style="margin: 0; line-height: 1.4; font-size: 0.85rem; color: #00FF66; overflow-x: auto; font-family: monospace;">
[1. DOCUMENT UPLOAD]
       │
       ▼ (PyPDF / Docx / Text Extractors)
[2. SEMANTIC CHUNKING] ──► (Cosine-similarity chunk boundaries)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
[3a. SPARSE INDEX]        [3b. DENSE INDEX]        [3c. KNOWLEDGE GRAPH]
  (Rank-BM25 Okapi)        (MiniLM-L6-v2 Embs)       (SpaCy NER + SVO POS)
       │                         │                         │
  `bm25.pkl`                `faiss.idx`               `graph.pkl`
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 │
                            [4. QUERY]
                                 │
       ▼ (Query Decomposition to multiple sub-queries)
[5. TRIPLE RETRIEVAL SCORING]
  (Sparse matches + Dense similarity + Graph node traversal)
       │
       ▼ (Reciprocal Rank Fusion - RRF)
[6. MS-MARCO CROSS-ENCODER RERANKING]
       │
       ▼ (Graph-Guided Summarization & Highlighting)
[7. STARK NEO-BRUTALIST PRESENTATION]
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Technology grid card
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 24px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D;">
            <h3 style="color: #FFFFFF; margin-top: 0; font-weight: 700; border-bottom: 2px solid #FFFFFF; padding-bottom: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; margin-bottom: 15px;">Built with State-of-the-Art Local Libraries</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.9rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; font-family: 'Space Grotesk', sans-serif;">
                <div>
                    <strong>NLP & Graph Modeling:</strong>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        <li><code>spaCy</code> (en_core_web_sm) for syntactic dependencies, entity classification, and verb phrases.</li>
                        <li><code>NetworkX</code> for entity co-occurrence modeling and graph queries.</li>
                    </ul>
                    <strong>Vector & Sparse Retrieval:</strong>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        <li><code>FAISS</code> (Facebook AI Similarity Search) CPU-index for high-performance retrieval.</li>
                        <li><code>Sentence-Transformers</code> (all-MiniLM-L6-v2) local text embeddings.</li>
                        <li><code>Rank-BM25</code> for keyword term-frequency retrieval.</li>
                    </ul>
                </div>
                <div>
                    <strong>Re-ranking & Summary:</strong>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        <li><code>MS-Marco Cross-Encoder</code> for calculating high-fidelity chunk relevance scores relative to query intents.</li>
                        <li>Custom extractive summaries utilizing text chunk node density and similarity.</li>
                    </ul>
                    <strong>User Interface & Visualizer:</strong>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        <li><code>Streamlit</code> for the interactive Neo-Brutalist interface.</li>
                        <li><code>Vis.js Network</code> rendering interactive nodes, chunks, and categories directly in the browser canvas.</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How to Use Card
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 24px; box-shadow: 5px 5px 0px 0px #2563EB;">
            <h3 style="color: #FFFFFF; margin-top: 0; font-weight: 700; border-bottom: 2px solid #FFFFFF; padding-bottom: 8px; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; margin-bottom: 15px;">Interactive Quick-Start</h3>
            <ol style="font-size: 0.95rem; line-height: 1.7; color: #FFFFFF; opacity: 0.9; padding-left: 20px; font-family: 'Space Grotesk', sans-serif;">
                <li>Navigate to the <strong>Search Engine</strong> tab, enter any query, or click one of the suggested search pills dynamically loaded from your index's main entities.</li>
                <li>Examine search matches displaying precise matched sentences, confidence ratings, and decomposition transparency stats.</li>
                <li>Go to the <strong>Knowledge Graph Visualizer</strong> tab to explore the physical structure map (showing chunks and how concepts link to them) or filter strictly to pure concept-to-concept linkages. Drag, zoom, and explore connections interactively!</li>
                <li>Tweak retrieval options in the sidebar to test different search weights, max chunk limits, or toggle advanced features like query splitting.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab_search:
        if not indexes_exist:
            st.markdown("""
            <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D; text-align: center; margin-top: 20px;">
                <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">No Documents Indexed Yet</h3>
                <p style="font-size: 1rem; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; margin-bottom: 0;">
                    The search engine requires documents to perform matching. Please go to the <strong>Home / About</strong> tab and upload some files to begin!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Let's check what documents exist in this index to present them
            if retriever and retriever.chunks_lookup:
                indexed_filenames = sorted(list(set(
                    chunk["metadata"]["filename"] for chunk in retriever.chunks_lookup.values()
                )))
                
                with st.expander(f"Currently Indexed Documents ({len(indexed_filenames)} files)", expanded=False):
                    for fname in indexed_filenames:
                        st.markdown(f"- `{fname}`")
                    
                    # Allow updating/uploading more documents
                    st.markdown("---")
                    uploaded_more = st.file_uploader(
                        "Add more documents or replace current index:",
                        type=["pdf", "docx", "txt"],
                        accept_multiple_files=True,
                        key="main_uploader_append"
                    )
                    if uploaded_more:
                        replace_choice = st.radio("Upload Strategy:", ["Replace existing index (Delete current documents)", "Append to existing index (Add documents)"], index=0)
                        if st.button("Process Uploaded Files", use_container_width=True):
                            clear_flag = (replace_choice == "Replace existing index (Delete current documents)")
                            if run_in_memory_ingestion(uploaded_more, clear_existing=clear_flag):
                                st.success("Indexes updated!")
                                st.rerun()

            # Generate Dynamic Sample Queries based on Knowledge Graph entities if possible
            suggested_queries = []
            if retriever and hasattr(retriever, 'graph_builder') and retriever.graph_builder.graph.number_of_nodes() > 0:
                # Get nodes that are entities (GPE, ORG, PERSON, PRODUCT, LAW)
                nodes_with_degree = sorted(
                    [(n, d, retriever.graph_builder.graph.nodes[n].get("entity_type", "Entity")) 
                     for n, d in retriever.graph_builder.graph.degree() 
                     if retriever.graph_builder.graph.nodes[n].get("type") == "entity" 
                     and retriever.graph_builder.graph.nodes[n].get("entity_type") in ["ORG", "GPE", "PERSON", "PRODUCT", "LAW"]],
                    key=lambda x: x[1],
                    reverse=True
                )
                # Select up to 4 interesting suggestions
                seen_names = set()
                for name, degree, type_val in nodes_with_degree:
                    clean_name = name.strip()
                    # Avoid single characters, numbers, and generic date strings
                    if len(clean_name) > 2 and clean_name.lower() not in seen_names and not clean_name.isdigit():
                        seen_names.add(clean_name.lower())
                        suggested_queries.append(f"Tell me about {clean_name}")
                        if len(suggested_queries) >= 4:
                            break

            # If we couldn't extract enough entities, provide default suggestions or generic prompts
            if len(suggested_queries) < 2:
                suggested_queries = [
                    "What are the main requirements discussed?",
                    "Key dates and compliance policy",
                    "Overview of the uploaded documents"
                ]

            st.markdown("**Suggested Queries (based on index entities):**")
            
            if "search_query" not in st.session_state:
                st.session_state["search_query"] = ""

            cols = st.columns(len(suggested_queries))
            for i, sample in enumerate(suggested_queries):
                if cols[i].button(sample, key=f"sample_{i}", use_container_width=True):
                    st.session_state["search_query"] = sample
                    st.rerun()

            # Large Search Bar Input
            query_input = st.text_input(
                "Enter natural language query:",
                value=st.session_state["search_query"],
                placeholder="Type search terms or entity queries here...",
                label_visibility="collapsed"
            )

            # Reset state after read
            if query_input != st.session_state["search_query"]:
                st.session_state["search_query"] = query_input

            # Run Search Execution
            if query_input:
                with st.spinner("Processing search query..."):
                    # Execute search
                    results_dict = retriever.search(
                        query=query_input,
                        k=num_results,
                        alpha=alpha,
                        department_filter=selected_dept,
                        company_filter=selected_company,
                        year_filter=selected_year,
                        section_filter=selected_sec,
                        use_decomposition=use_decomposition,
                        use_parent_retrieval=use_parent_retrieval,
                        enable_financial_expansion=enable_fin_expansion
                    )
                    
                    results = results_dict["results"]
                    summary = results_dict["summary"]
                    see_also = results_dict["see_also"]
                    sub_queries = results_dict.get("sub_queries", [query_input])

                    if not results:
                        st.info("No matching results found for the query. Try adjusting your filters or search terms.")
                    else:
                        # 1. Display Extractive Summary Card
                        st.subheader("Extractive Summary (TextRank)")
                        st.markdown(f"""
                        <div class="summary-card">
                            <div style="font-weight:600; font-size:1.15rem; color:#2563EB; margin-bottom:8px;">Key Summary Points</div>
                            <div style="line-height:1.6; font-size:0.98rem; opacity:0.95;">
                                {summary if summary else "Could not generate summary from top documents."}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # If query decomposed, show a caption
                        if len(sub_queries) > 1:
                            st.caption(f"Query decomposed and processed as: {', '.join([f'**\"{q}\"**' for q in sub_queries])}")
                        
                        # Create layouts for Results and suggestions
                        res_col, sugg_col = st.columns([3, 1])
                        
                        with res_col:
                            st.subheader(f"Search Results ({len(results)} matches)")
                            
                            # Loop and render cards
                            for idx, chunk in enumerate(results):
                                # Extract basic items
                                filename = chunk["metadata"]["filename"]
                                filepath = chunk["metadata"]["filepath"]
                                dept = chunk["metadata"]["department"]
                                doc_date = chunk["metadata"]["doc_date"]
                                size = chunk["metadata"]["file_size_bytes"]
                                text_highlighted = chunk["highlighted_text"]
                                
                                # Financial badges
                                fin_badges = ""
                                if chunk["metadata"].get("is_financial_report"):
                                    c_tick = chunk["metadata"].get("ticker", "GEN")
                                    c_yr = chunk["metadata"].get("fiscal_year", "")
                                    c_sec = chunk["metadata"].get("section", "BODY")
                                    fin_badges = f'<span class="dept-badge" style="background-color:#38BDF8; margin-left:6px;">{c_tick} FY{c_yr}</span> <span class="dept-badge" style="background-color:#A855F7; margin-left:6px;">{c_sec}</span>'

                                fused_score = chunk["fused_score"]
                                rerank_score = chunk.get("rerank_score", 0.0)
                                
                                ranks = chunk["retrieval_ranks"]
                                scores = chunk["retrieval_scores"]
                                
                                walks_html = ""
                                
                                # Render HTML card
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="card-header">
                                        <span class="doc-title">{filename}</span>
                                        <div>
                                            <span class="dept-badge">{dept}</span>
                                            {fin_badges}
                                        </div>
                                    </div>
                                    <div class="meta-line">
                                        Document Date: {doc_date} | File Size: {size} bytes | Path: <code style="font-size:0.75rem;">{filepath}</code>
                                    </div>
                                    <div class="snippet-text">
                                        {text_highlighted}
                                    </div>
                                    <div class="stats-grid">
                                        <div class="stat-box tooltip">
                                            <span class="stat-label">Cross-Encoder:</span>
                                            <span class="stat-value-primary">{rerank_score:.4f}</span>
                                            <span class="tooltiptext">
                                                <b>Cross-Encoder Reranker Score</b><br>
                                                Calculates deep semantic query-document relevance using a neural network. 
                                                A score of {rerank_score:.4f} represents high contextual alignment.
                                            </span>
                                        </div>
                                        <div class="stat-box tooltip">
                                            <span class="stat-label">RRF:</span>
                                            <span class="stat-value">{fused_score:.4f}</span>
                                            <span class="tooltiptext">
                                                <b>Reciprocal Rank Fusion (RRF)</b><br>
                                                Blends and normalizes Sparse, Dense, and Graph search ranks. 
                                                RRF score is calculated as sum(weight / (60 + rank)).
                                            </span>
                                        </div>
                                        <div class="stat-box tooltip">
                                            <span class="stat-label">Sparse (BM25):</span>
                                            <span class="stat-value">{scores['sparse']:.4f} (Rank {ranks['sparse'] or 'N/A'})</span>
                                            <span class="tooltiptext">
                                                <b>Sparse Search (BM25)</b><br>
                                                Evaluates keyword matching statistics (term frequency / document frequency). 
                                                A raw score of {scores['sparse']:.4f} (Rank {ranks['sparse'] or 'N/A'}) reflects exact term matches.
                                            </span>
                                        </div>
                                        <div class="stat-box tooltip">
                                            <span class="stat-label">Dense (FAISS):</span>
                                            <span class="stat-value">{scores['dense']:.4f} (Rank {ranks['dense'] or 'N/A'})</span>
                                            <span class="tooltiptext">
                                                <b>Dense Search (FAISS)</b><br>
                                                Measures vector similarity in semantic space. 
                                                Cosine similarity score: {scores['dense']:.4f} (Rank {ranks['dense'] or 'N/A'}), matching concept meanings beyond literal words.
                                            </span>
                                        </div>
                                        <div class="stat-box tooltip">
                                            <span class="stat-label">Graph:</span>
                                            <span class="stat-value">{scores['graph']:.1f} (Rank {ranks['graph'] or 'N/A'})</span>
                                            <span class="tooltiptext">
                                                <b>Knowledge Graph Search</b><br>
                                                Finds intersections of extracted entity nodes and relations in the graph. 
                                                Match score: {scores['graph']:.1f} (Rank {ranks['graph'] or 'N/A'}), resolving multi-hop structural connections.
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with sugg_col:
                            st.subheader("See Also")
                            st.markdown("Related topics & entities extracted from the Knowledge Graph:")
                            
                            if not see_also:
                                st.info("No related terms found in graph context.")
                            else:
                                for item in see_also:
                                    term_name = item["name"]
                                    term_type = item["type"]
                                    
                                    # Clicking this button triggers search for this term
                                    if st.button(f"{term_name} ({term_type})", key=f"sugg_{term_name}", use_container_width=True):
                                        st.session_state["search_query"] = term_name
                                        st.rerun()
                                        
                            st.markdown("---")
                            
                            # 3. Download results as JSON
                            st.subheader("Export Results")
                            
                            # Format results to serializable structure
                            export_data = {
                                "query": query_input,
                                "timestamp": datetime.now().isoformat(),
                                "summary": summary,
                                "results": [
                                    {
                                        "chunk_id": chunk["chunk_id"],
                                        "doc_id": chunk["doc_id"],
                                        "filename": chunk["metadata"]["filename"],
                                        "department": chunk["metadata"]["department"],
                                        "doc_date": chunk["metadata"]["doc_date"],
                                        "text": chunk["text"],
                                        "fused_score": chunk["fused_score"],
                                        "rerank_score": chunk.get("rerank_score", 0.0)
                                    }
                                    for chunk in results
                                ]
                            }
                            
                            st.download_button(
                                label="Download Search Results (JSON)",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"hybrid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )

    with tab_graph:
        if not indexes_exist:
            st.markdown("""
            <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D; text-align: center; margin-top: 20px;">
                <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">No Documents Indexed Yet</h3>
                <p style="font-size: 1rem; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; margin-bottom: 0;">
                    The knowledge graph requires documents to show entities. Please go to the <strong>Home / About</strong> tab and upload some files to begin!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.subheader("Interactive Knowledge Graph Visualizer")
            st.markdown(
                "Explore the document collection's semantic relationships, co-occurrences, and structural associations extracted by the parser. "
                "Drag nodes to rearrange, scroll to zoom, and hover over elements for entity types and relation paths."
            )

            graph = retriever.graph_builder.graph
            if graph.number_of_nodes() == 0:
                st.info("The Knowledge Graph is currently empty. Please upload and index documents in the Search Engine tab.")
            else:
                col_g1, col_g2, col_g3 = st.columns([1, 1, 1])
                with col_g1:
                    # Find all unique documents represented in the graph chunks
                    all_files = sorted(list(set(
                        attrs.get("filename", "Unknown") 
                        for n, attrs in graph.nodes(data=True) 
                        if attrs.get("type") == "chunk"
                    )))
                    
                    selected_file = st.selectbox(
                        "Filter Graph by Document Source",
                        options=["All Documents"] + all_files,
                        key="graph_file_filter"
                    )
                with col_g2:
                    viz_mode = st.selectbox(
                        "Visualization Mode",
                        options=[
                            "Document Structure Map (Chunks & Entities)",
                            "Semantic Concept Network (Entities & Relations)"
                        ],
                        index=0,
                        help="Structure mode shows how concepts map to text segments. Semantic mode shows pure connections between concepts (no abstract Chunks)."
                    )
                with col_g3:
                    max_nodes = st.slider(
                        "Max Nodes to Display",
                        min_value=10,
                        max_value=150,
                        value=60,
                        step=5,
                        help="Limit the number of visible nodes to avoid cluttered layouts."
                    )

                # Node filtering logic based on Mode and Source file
                valid_nodes = set()
                
                if viz_mode == "Semantic Concept Network (Entities & Relations)":
                    # Pure Entity-to-Entity Network
                    if selected_file == "All Documents":
                        valid_nodes = {n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "entity"}
                    else:
                        # Chunks belonging to this file
                        valid_chunks = {n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "chunk" and attrs.get("filename") == selected_file}
                        # Entities appearing in these chunks
                        for chunk in valid_chunks:
                            for u, v in graph.in_edges(chunk):
                                if graph.nodes[u].get("type") == "entity":
                                    valid_nodes.add(u)
                            for u, v in graph.out_edges(chunk):
                                if graph.nodes[v].get("type") == "entity":
                                    valid_nodes.add(v)
                                    
                    filtered_graph = graph.subgraph(valid_nodes)
                    
                    # Sort entity nodes by degree within the filtered graph
                    entity_degrees = sorted([(n, filtered_graph.degree(n)) for n in filtered_graph.nodes()], key=lambda x: x[1], reverse=True)
                    top_nodes = [n for n, d in entity_degrees[:max_nodes]]
                    viz_graph = filtered_graph.subgraph(top_nodes)
                    
                else:
                    # Chunks + Entities Structural Map
                    if selected_file == "All Documents":
                        valid_nodes = set(graph.nodes())
                    else:
                        valid_chunks = {n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "chunk" and attrs.get("filename") == selected_file}
                        valid_nodes.update(valid_chunks)
                        for chunk in valid_chunks:
                            for u, v in graph.in_edges(chunk):
                                valid_nodes.add(u)
                            for u, v in graph.out_edges(chunk):
                                valid_nodes.add(v)

                    filtered_graph = graph.subgraph(valid_nodes)

                    # Sort entity nodes by degree
                    entity_nodes = [n for n, attrs in filtered_graph.nodes(data=True) if attrs.get("type") == "entity"]
                    entity_degrees = sorted([(n, filtered_graph.degree(n)) for n in entity_nodes], key=lambda x: x[1], reverse=True)
                    top_entities = [n for n, d in entity_degrees[:max_nodes]]

                    display_nodes = set(top_entities)
                    for ent in top_entities:
                        for neighbor in filtered_graph.neighbors(ent):
                            if filtered_graph.nodes[neighbor].get("type") == "chunk":
                                display_nodes.add(neighbor)
                        for u, v in filtered_graph.in_edges(ent):
                            if filtered_graph.nodes[u].get("type") == "chunk":
                                display_nodes.add(u)

                    viz_graph = filtered_graph.subgraph(display_nodes)

                # Format to Vis.js structures
                vis_nodes = []
                for n, attrs in viz_graph.nodes(data=True):
                    node_type = attrs.get("type", "entity")
                    if node_type == "chunk":
                        # Retrieve chunk_index from retriever chunks lookup if available
                        chunk_idx = 0
                        if retriever and retriever.chunks_lookup and n in retriever.chunks_lookup:
                            chunk_idx = retriever.chunks_lookup[n].get("metadata", {}).get("chunk_index", 0)
                        
                        label = f"Chunk {chunk_idx}"
                        group = "chunk"
                        title = f"<b>Document Chunk {chunk_idx}</b><br>File: {attrs.get('filename')}<br>ID: {n}"
                    else:
                        label = n
                        group = attrs.get("entity_type", "Entity")
                        title = f"<b>Entity:</b> {n}<br>Type: {group}"

                    vis_nodes.append({
                        "id": n,
                        "label": label,
                        "group": group,
                        "title": title
                    })

                vis_edges = []
                for u, v, attrs in viz_graph.edges(data=True):
                    relation = attrs.get("relation", "connected")
                    # Exclude edge labeling of "appears_in" to prevent clutter, keeping it for semantic relationships
                    edge_label = relation if relation != "appears_in" else ""
                    vis_edges.append({
                        "from": u,
                        "to": v,
                        "label": edge_label,
                        "title": relation
                    })

                # Style options for Dark Neo-Brutalist design
                vis_options = {
                    "nodes": {
                        "shape": "box",
                        "margin": 10,
                        "font": {
                            "color": "#FFFFFF",
                            "size": 14,
                            "face": "Space Grotesk, sans-serif",
                            "bold": {
                                "color": "#FFFFFF",
                                "size": 14,
                                "face": "Space Grotesk, sans-serif"
                            }
                        },
                        "borderWidth": 3,
                        "shadow": {
                            "enabled": True,
                            "color": "#2563EB",
                            "size": 0,
                            "x": 4,
                            "y": 4
                        }
                    },
                    "edges": {
                        "width": 2,
                        "color": {
                            "color": "#FFFFFF",
                            "highlight": "#FFDE4D",
                            "hover": "#FFDE4D"
                        },
                        "arrows": {
                            "to": {
                                "enabled": True,
                                "scaleFactor": 0.8
                            }
                        },
                        "font": {
                            "color": "#A0A0A0",
                            "size": 10,
                            "face": "monospace",
                            "strokeWidth": 0
                        },
                        "smooth": {
                            "type": "continuous"
                        }
                    },
                    "groups": {
                        "chunk": {
                            "color": {
                                "background": "#1E1B4B",
                                "border": "#FFFFFF",
                                "highlight": {
                                    "background": "#FFDE4D",
                                    "border": "#000000"
                                }
                            },
                            "font": {
                                "color": "#FFFFFF"
                            },
                            "shadow": {
                                "color": "#FFDE4D"
                            }
                        },
                        "ORG": { "color": { "background": "#121318", "border": "#38BDF8", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "PERSON": { "color": { "background": "#121318", "border": "#F43F5E", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "GPE": { "color": { "background": "#121318", "border": "#10B981", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "LOC": { "color": { "background": "#121318", "border": "#10B981", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "PRODUCT": { "color": { "background": "#121318", "border": "#A855F7", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "DATE": { "color": { "background": "#121318", "border": "#6B7280", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "LAW": { "color": { "background": "#121318", "border": "#A855F7", "highlight": { "background": "#FFDE4D", "border": "#000000" } } },
                        "CO_OCCURRENCE": { "color": { "background": "#121318", "border": "#FFFFFF", "highlight": { "background": "#2563EB", "border": "#FFFFFF" } } }
                    },
                    "physics": {
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {
                            "gravitationalConstant": -120,
                            "centralGravity": 0.005,
                            "springLength": 200,
                            "springConstant": 0.05
                        },
                        "stabilization": {
                            "iterations": 150,
                            "updateInterval": 25
                        }
                    },
                    "interaction": {
                        "hover": True,
                        "zoomView": True,
                        "dragView": True
                    }
                }

                # Embed Vis.js network graph
                import streamlit.components.v1 as components
                import json

                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                    <style type="text/css">
                        #mynetwork {{
                            width: 100%;
                            height: 600px;
                            background-color: #0A0A0C;
                            border: 3px solid #FFFFFF;
                            box-shadow: 6px 6px 0px 0px #2563EB;
                        }}
                        body {{
                            margin: 0;
                            padding: 0;
                            background-color: #0A0A0C;
                            color: #FFFFFF;
                            font-family: 'Space Grotesk', sans-serif;
                        }}
                        /* Neo-brutalist custom styled tooltips */
                        div.vis-tooltip {{
                            background-color: #121318 !important;
                            color: #FFFFFF !important;
                            border: 2px solid #FFFFFF !important;
                            border-radius: 0px !important;
                            font-family: monospace !important;
                            padding: 8px 12px !important;
                            box-shadow: 4px 4px 0px 0px #2563EB !important;
                        }}
                    </style>
                </head>
                <body>
                    <div id="mynetwork"></div>
                    <script type="text/javascript">
                        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
                        var edges = new vis.DataSet({json.dumps(vis_edges)});
                        var container = document.getElementById('mynetwork');
                        var data = {{
                            nodes: nodes,
                            edges: edges
                        }};
                        var options = {json.dumps(vis_options)};
                        var network = new vis.Network(container, data, options);
                    </script>
                </body>
                </html>
                """
                
                components.html(html_template, height=650)
                st.caption("Tip: You can use zoom (scroll) and pan (drag background) to explore dense parts of the document graph structure.")

    # =========================================================================
    # TAB 4: SEC ANNUAL REPORTS & FILINGS MANAGEMENT
    # =========================================================================
    with tab_filings:
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #2563EB;">
            <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">SEC ANNUAL REPORTS & CORPORATE FILINGS</h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; margin-bottom: 0;">
                Ingest, parse, and segment multi-year corporate annual reports (Form 10-K and 10-Q).
                The engine extracts SEC Item boundaries (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements)
                and builds section-isolated index representations for cross-year semantic comparison.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.markdown("#### Load Verified Research Benchmark Filings")
            st.info("Pre-packaged multi-year Form 10-Ks for Apple Inc. (FY2023 vs FY2024) and Microsoft Corp (FY2023 vs FY2024).")
            if st.button("Load TEEP 2026 Sample 10-K Filings (AAPL & MSFT)", use_container_width=True):
                sample_dir = Path(__file__).resolve().parent / "data_test" / "financial_samples"
                sample_files = list(sample_dir.glob("*.txt"))
                if sample_files:
                    with st.spinner("Processing and indexing multi-year financial 10-Ks..."):
                        if run_in_memory_ingestion(sample_files, clear_existing=True):
                            st.success("Loaded and indexed multi-year filings!")
                            st.rerun()
                else:
                    st.error("Sample directory data_test/financial_samples/ not found.")

        with col_f2:
            st.markdown("#### Upload Custom Annual Reports")
            uploaded_sec = st.file_uploader(
                "Upload Form 10-K / 10-Q (.pdf, .docx, .txt):",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="sec_filings_uploader"
            )
            if uploaded_sec:
                if st.button("Process & Append SEC Reports", use_container_width=True):
                    with st.spinner("Ingesting corporate reports..."):
                        if run_in_memory_ingestion(uploaded_sec, clear_existing=False):
                            st.success("Reports indexed successfully!")
                            st.rerun()

        # Overview of currently indexed filings
        st.markdown("---")
        st.subheader("Indexed Corporate Filings Overview")
        if indexes_exist and retriever and retriever.chunks_lookup:
            # Group chunks by doc_id
            doc_groups = {}
            for cid, cdata in retriever.chunks_lookup.items():
                did = cdata.get("doc_id", "default")
                if did not in doc_groups:
                    doc_groups[did] = []
                doc_groups[did].append(cdata)

            filing_cards = []
            for did, chunks in doc_groups.items():
                first_meta = chunks[0]["metadata"]
                ticker = first_meta.get("ticker", "GEN")
                company = first_meta.get("company", "Company")
                year = first_meta.get("fiscal_year", "N/A")
                form = first_meta.get("form_type", "Document")
                fname = first_meta.get("filename", "")
                sections = sorted(list(set(c["metadata"].get("section", "BODY") for c in chunks)))

                st.markdown(f"""
                <div style="background-color: #121318; border: 2px solid #FFFFFF; padding: 16px; margin-bottom: 12px; box-shadow: 4px 4px 0px 0px #2563EB;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:1.1rem; color:#FFFFFF;">{company} ({ticker}) — {form} FY{year}</span>
                        <span class="dept-badge" style="background-color:#38BDF8;">{len(chunks)} CHUNKS</span>
                    </div>
                    <div style="font-size:0.82rem; color:#A0A0A0; margin-top:6px; font-family:monospace;">
                        File: {fname} | Identified Sections: {', '.join(sections)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents indexed yet. Click 'Load TEEP 2026 Sample 10-K Filings' above to initialize with benchmark data.")

    # =========================================================================
    # TAB 5: SEMANTIC SHIFT & NARRATIVE DRIFT (TEEP 2026 PRIMARY RESEARCH)
    # =========================================================================
    with tab_semantic_shift:
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D;">
            <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">TEEP 2026: QUANTIFYING SEMANTIC SHIFTS & NARRATIVE DRIFT</h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; margin-bottom: 0;">
                Empirical framework for measuring year-over-year textual displacement in corporate annual reports.
                Calculates cosine embedding distances, Loughran-McDonald financial tone shifts, risk disclosure additions/deletions,
                and synthesizes the composite <strong>Semantic Shift Index (S_shift)</strong> for predicting operating performance changes and excess returns.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not indexes_exist or not retriever:
            st.warning("Please load or index annual reports first to conduct semantic shift analysis.")
        else:
            # Find companies with at least 2 filings/years
            comp_years_map = {}
            for cid, cdata in retriever.chunks_lookup.items():
                meta = cdata.get("metadata", {})
                comp = meta.get("company", "Unknown")
                yr = str(meta.get("fiscal_year", "N/A"))
                if comp != "Unknown" and yr != "N/A":
                    if comp not in comp_years_map:
                        comp_years_map[comp] = set()
                    comp_years_map[comp].add(yr)

            available_companies = [c for c, yrs in comp_years_map.items() if len(yrs) >= 2]
            if not available_companies:
                available_companies = list(comp_years_map.keys()) if comp_years_map else ["Apple Inc."]

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                target_comp = st.selectbox("Select Target Company:", options=available_companies, index=0)
            with col_s2:
                sec_choice = st.selectbox("Filing Section to Analyze:", options=[
                    "Item 1A (Risk Factors)",
                    "Item 7 (MD&A)",
                    "Item 1 (Business)",
                    "Full Filing (Aggregate)"
                ], index=0)
            
            comp_yrs_sorted = sorted(list(comp_years_map.get(target_comp, ["2023", "2024"])))
            with col_s3:
                year_t0 = st.selectbox("Base Year (t0):", options=comp_yrs_sorted, index=0)
            with col_s4:
                year_t1 = st.selectbox("Comparison Year (t1):", options=comp_yrs_sorted, index=min(1, len(comp_yrs_sorted)-1))

            if st.button("Execute Multi-Dimensional Semantic Shift Analysis", use_container_width=True):
                with st.spinner(f"Computing semantic drift for {target_comp} ({year_t0} -> {year_t1})..."):
                    # Map section choice to code
                    sec_code_map = {
                        "Item 1A (Risk Factors)": "ITEM_1A",
                        "Item 7 (MD&A)": "ITEM_7",
                        "Item 1 (Business)": "ITEM_1",
                        "Full Filing (Aggregate)": None
                    }
                    target_sec_code = sec_code_map[sec_choice]

                    # Gather text chunks for t0 and t1
                    text_t0_chunks = []
                    text_t1_chunks = []
                    for cid, cdata in retriever.chunks_lookup.items():
                        meta = cdata.get("metadata", {})
                        if meta.get("company") == target_comp:
                            c_yr = str(meta.get("fiscal_year", ""))
                            c_sec = meta.get("section", "").upper()
                            
                            if target_sec_code is None or target_sec_code in c_sec:
                                if c_yr == str(year_t0):
                                    text_t0_chunks.append(cdata["text"])
                                elif c_yr == str(year_t1):
                                    text_t1_chunks.append(cdata["text"])

                    text_t0 = "\n\n".join(text_t0_chunks)
                    text_t1 = "\n\n".join(text_t1_chunks)

                    if not text_t0 or not text_t1:
                        st.error(f"Insufficient text found for {target_comp} comparing {year_t0} and {year_t1}. Ensure both years are indexed.")
                    else:
                        analyzer = SemanticShiftAnalyzer(retriever.nlp)
                        analysis = analyzer.analyze_reports(
                            text_t0=text_t0,
                            text_t1=text_t1,
                            year_t0=str(year_t0),
                            year_t1=str(year_t1),
                            embedding_model=retriever.embedding_model,
                            company_name=target_comp
                        )

                        # Render Key KPIs
                        shift_val = analysis["composite_semantic_shift_index"]
                        severity = analysis["shift_severity"]
                        cos_dist = analysis["embedding_metrics"]["cosine_distance"]
                        tone_delta = analysis["sentiment_metrics"]["deltas"]["net_tone_delta"]
                        risk_count = analysis["risk_metrics"]["risk_expansion_count"]

                        st.markdown("---")
                        st.subheader(f"Quantified Drift Results: {target_comp} ({year_t0} vs {year_t1})")

                        # KPI Cards Neo-Brutalist Grid
                        st.markdown(f"""
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;">
                            <div style="background:#121318; border:3px solid #FFDE4D; padding:18px; box-shadow:4px 4px 0px 0px #FFDE4D;">
                                <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#A0A0A0;">Composite Shift Index</div>
                                <div style="font-size:2.2rem; font-weight:700; color:#FFDE4D; font-family:monospace; margin:4px 0;">{shift_val}</div>
                                <div style="font-size:0.8rem; font-weight:700; color:#FFFFFF;">{severity}</div>
                            </div>
                            <div style="background:#121318; border:3px solid #2563EB; padding:18px; box-shadow:4px 4px 0px 0px #2563EB;">
                                <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#A0A0A0;">Cosine Angular Distance</div>
                                <div style="font-size:2.2rem; font-weight:700; color:#38BDF8; font-family:monospace; margin:4px 0;">{cos_dist}</div>
                                <div style="font-size:0.8rem; font-weight:700; color:#FFFFFF;">Similarity: {analysis['embedding_metrics']['cosine_similarity']}</div>
                            </div>
                            <div style="background:#121318; border:3px solid #10B981; padding:18px; box-shadow:4px 4px 0px 0px #10B981;">
                                <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#A0A0A0;">Net Tone Drift (Delta)</div>
                                <div style="font-size:2.2rem; font-weight:700; color:#10B981; font-family:monospace; margin:4px 0;">{tone_delta:+.4f}</div>
                                <div style="font-size:0.8rem; font-weight:700; color:#FFFFFF;">Direction: {analysis['sentiment_metrics']['tone_drift_direction']}</div>
                            </div>
                            <div style="background:#121318; border:3px solid #F43F5E; padding:18px; box-shadow:4px 4px 0px 0px #F43F5E;">
                                <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#A0A0A0;">Net Risk Item Shift</div>
                                <div style="font-size:2.2rem; font-weight:700; color:#F43F5E; font-family:monospace; margin:4px 0;">{risk_count:+d}</div>
                                <div style="font-size:0.8rem; font-weight:700; color:#FFFFFF;">+{len(analysis['risk_metrics']['added_risks'])} New / -{len(analysis['risk_metrics']['removed_risks'])} Retired</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Loughran-McDonald Sentiment Breakdown
                        col_sent1, col_sent2 = st.columns([1, 1])
                        with col_sent1:
                            st.markdown("#### Loughran-McDonald Lexicon Sentiment Shifts")
                            sent_table = []
                            for cat in ["negative", "positive", "uncertainty", "litigious", "constraining"]:
                                s0 = analysis["sentiment_metrics"]["period_t0_scores"].get(cat, 0.0)
                                s1 = analysis["sentiment_metrics"]["period_t1_scores"].get(cat, 0.0)
                                d = analysis["sentiment_metrics"]["deltas"].get(f"{cat}_delta", 0.0)
                                sent_table.append({
                                    "Tone Category": cat.capitalize(),
                                    f"FY{year_t0} (per 1k words)": s0,
                                    f"FY{year_t1} (per 1k words)": s1,
                                    "Tone Shift (Delta)": f"{d:+.3f}"
                                })
                            st.table(sent_table)

                        with col_sent2:
                            st.markdown("#### Lexical & Information Divergence")
                            st.markdown(f"""
                            - **Vocabulary Jaccard Distance**: `{analysis['vocabulary_metrics']['jaccard_distance']}`
                            - **Jensen-Shannon Divergence (JSD)**: `{analysis['vocabulary_metrics']['jensen_shannon_divergence']}`
                            - **Novel Terminology Introduction Rate**: `{analysis['vocabulary_metrics']['novel_term_ratio'] * 100:.1f}%`
                            - **Risk Disclosure Retention Rate**: `{analysis['risk_metrics']['retention_rate'] * 100:.1f}%`
                            """)
                            st.markdown("""
                            <div style="background:#1E1B4B; border:2px solid #FFFFFF; padding:14px; margin-top:10px;">
                                <strong style="color:#FFDE4D;">Research Significance (TEEP 2026):</strong><br>
                                <span style="font-size:0.85rem; color:#FFFFFF; opacity:0.9;">
                                High linguistic divergence (JSD > 0.35) combined with negative tone escalation signifies non-routine corporate disclosure restructuring,
                                which empirically anticipates heightened earnings volatility in subsequent quarters.
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Risk Factors Drift Detail
                        st.markdown("---")
                        st.subheader("Risk Disclosures Drift Analysis (Item 1A)")
                        
                        added_r = analysis["risk_metrics"]["added_risks"]
                        mod_r = analysis["risk_metrics"]["modified_risks"]
                        
                        if added_r:
                            st.markdown(f"**Newly Introduced Risk Factors in FY{year_t1} ({len(added_r)} detected):**")
                            for ar in added_r:
                                st.markdown(f"- 🔴 **[Novelty: {ar['novelty_score']:.2f}]** {ar['risk_text']}")

                        if mod_r:
                            st.markdown(f"**Significantly Modified / Reframed Disclosures ({len(mod_r)} detected):**")
                            for mr in mod_r:
                                with st.expander(f"Modified Risk (Similarity: {mr['similarity']:.2f})"):
                                    st.markdown(f"**FY{year_t0} Prior Version:** {mr['prior_version']}")
                                    st.markdown(f"**FY{year_t1} Current Version:** {mr['current_version']}")

                        # Economic hypothesis summary
                        st.markdown("---")
                        st.markdown(f"""
                        <div style="background-color: #050507; border: 3px solid #2563EB; padding: 18px; box-shadow: 4px 4px 0px 0px #FFDE4D; color: #FFFFFF;">
                            <h4 style="color: #FFDE4D; margin-top: 0;">TEEP 2026 QUANTITATIVE FINANCE MODEL PREDICTION</h4>
                            <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 0;">
                                {analysis['economic_hypothesis_summary']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 6: FINANCIAL METRICS EXTRACTION & SOURCE AUDIT
    # =========================================================================
    with tab_metrics:
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #2563EB;">
            <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">FINANCIAL INFORMATION EXTRACTION & SOURCE TRACEABILITY</h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; margin-bottom: 0;">
                Automated extraction of primary financial indicators (Revenue, Operating Income, Margins, Cash Flow, Debt).
                Every figure is paired with an exact <strong>Chunk ID, SEC Section, and Verbatim Evidence Quote</strong>
                and evaluated by the mathematical consistency validation engine.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not indexes_exist or not retriever:
            st.warning("Please load or index annual reports to view extracted metrics.")
        else:
            extractor = FinancialMetricExtractor()
            corpus_chunks = list(retriever.chunks_lookup.values())
            extracted_records = extractor.extract_from_corpus(corpus_chunks)

            if not extracted_records:
                st.info("No standardized financial metrics were extracted. Ensure annual reports contain Income Statement / Balance Sheet disclosures.")
            else:
                # Group records by (ticker, fiscal_year)
                companies_years = sorted(list(set((rec["ticker"], rec["fiscal_year"]) for rec in extracted_records.values())))
                
                # Selection row
                c_sel1, c_sel2 = st.columns(2)
                with c_sel1:
                    sel_tick = st.selectbox("Select Company / Ticker:", options=sorted(list(set(cy[0] for cy in companies_years))), index=0)
                
                avail_yrs = sorted([cy[1] for cy in companies_years if cy[0] == sel_tick])
                with c_sel2:
                    sel_yr = st.selectbox("Select Fiscal Year:", options=avail_yrs, index=len(avail_yrs)-1)

                # Filter records for selected company and year
                current_metrics_dict = {
                    rec["metric_key"]: rec["raw_value"]
                    for rec in extracted_records.values()
                    if rec["ticker"] == sel_tick and rec["fiscal_year"] == sel_yr
                }

                st.subheader(f"Extracted Indicators: {sel_tick} (FY{sel_yr})")
                
                # Display metrics table
                table_rows = []
                for rec in extracted_records.values():
                    if rec["ticker"] == sel_tick and rec["fiscal_year"] == sel_yr:
                        table_rows.append({
                            "Financial Metric": rec["display_name"],
                            "Category": rec["category"],
                            "Extracted Value": rec["formatted_value"],
                            "SEC Section": rec["section"],
                            "Filing Source": rec["filename"],
                            "Chunk ID": rec["chunk_id"]
                        })

                st.table(table_rows)

                # Mathematical Sanity Engine Validations
                st.markdown("---")
                st.subheader("Mathematical Sanity & Accounting Equation Checks")
                validations = extractor.validate_financial_consistency(current_metrics_dict)
                
                v_cols = st.columns(len(validations) if validations else 1)
                for i, v in enumerate(validations):
                    with v_cols[i]:
                        status_color = "#10B981" if v["status"] in ["VALID", "COMPUTED"] else "#F43F5E"
                        st.markdown(f"""
                        <div style="background:#121318; border:2px solid {status_color}; padding:14px; box-shadow:3px 3px 0px 0px {status_color};">
                            <div style="font-size:0.75rem; font-weight:700; color:{status_color};">{v['status']}</div>
                            <div style="font-size:0.85rem; font-weight:700; color:#FFFFFF; margin:6px 0;">{v['rule']}</div>
                            <div style="font-size:1.1rem; font-weight:700; color:#FFDE4D; font-family:monospace;">{v['calculated']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Temporal Year-over-Year Comparison if multiple years exist
                if len(avail_yrs) >= 2:
                    st.markdown("---")
                    st.subheader(f"Temporal YoY Metric Trajectory: {avail_yrs[0]} vs {avail_yrs[-1]}")
                    m0 = {rec["metric_key"]: rec["raw_value"] for rec in extracted_records.values() if rec["ticker"] == sel_tick and rec["fiscal_year"] == avail_yrs[0]}
                    m1 = {rec["metric_key"]: rec["raw_value"] for rec in extracted_records.values() if rec["ticker"] == sel_tick and rec["fiscal_year"] == avail_yrs[-1]}
                    comp_table = extractor.compute_temporal_comparison(m0, m1, avail_yrs[0], avail_yrs[-1])
                    
                    if comp_table:
                        yoy_rows = []
                        for ct in comp_table:
                            yoy_rows.append({
                                "Metric": ct["display_name"],
                                f"FY{avail_yrs[0]}": f"${ct['value_t0']:,.0f}" if ct['value_t0'] > 100 else f"${ct['value_t0']:.2f}",
                                f"FY{avail_yrs[-1]}": f"${ct['value_t1']:,.0f}" if ct['value_t1'] > 100 else f"${ct['value_t1']:.2f}",
                                "Dollar Delta": f"${ct['delta_abs']:+,.0f}" if abs(ct['delta_abs']) > 100 else f"${ct['delta_abs']:+.2f}",
                                "Growth (%)": f"{ct['growth_pct']:+.2f}%",
                                "Trajectory": ct["trend"]
                            })
                        st.table(yoy_rows)

                # Verbatim Source Audit Trail
                st.markdown("---")
                with st.expander("Verbatim Evidence Quotes & Citation Audit Trail", expanded=False):
                    for rec in extracted_records.values():
                        if rec["ticker"] == sel_tick and rec["fiscal_year"] == sel_yr:
                            st.markdown(f"**{rec['display_name']} ({rec['formatted_value']})** — Section: `{rec['section']}` | Chunk: `{rec['chunk_id']}`")
                            st.markdown(f"> *\"{rec['evidence_quote']}\"*")
                            st.markdown("")

    # =========================================================================
    # TAB 7: EVIDENCE-GROUNDED FINANCIAL Q&A
    # =========================================================================
    with tab_qa:
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #FFDE4D;">
            <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">EVIDENCE-GROUNDED FINANCIAL Q&A & RESEARCH DOSSIER</h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; margin-bottom: 0;">
                Audited answers with full evidence grounding. Outputs strictly segregate
                <strong>Extracted Facts (Verified Disclosures)</strong> from <strong>Analytical Interpretations (Temporal Observations)</strong>
                with clickable section-level citations and exportable research dossiers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not indexes_exist or not retriever:
            st.warning("Please index documents to use the Financial Q&A engine.")
        else:
            qa_engine = EvidenceGroundedFinancialQA(retriever)

            # Suggested Research Prompts
            sample_fin_prompts = [
                "How did Apple's operating margin change from FY2023 to FY2024 and what were the primary MD&A drivers?",
                "What new risk factors related to artificial intelligence and regulations did Apple and Microsoft disclose?",
                "Compare Microsoft's capital expenditures and cloud revenue growth between 2023 and 2024."
            ]
            
            st.markdown("**Sample TEEP 2026 Research Queries:**")
            p_cols = st.columns(len(sample_fin_prompts))
            for i, p_txt in enumerate(sample_fin_prompts):
                if p_cols[i].button(p_txt, key=f"fin_prompt_{i}", use_container_width=True):
                    st.session_state["fin_query"] = p_txt
                    st.rerun()

            if "fin_query" not in st.session_state:
                st.session_state["fin_query"] = sample_fin_prompts[0]

            qa_query = st.text_input("Enter financial query:", value=st.session_state["fin_query"])

            if st.button("Generate Evidence-Grounded Analysis", use_container_width=True) and qa_query:
                with st.spinner("Retrieving annual report evidence and synthesizing grounded response..."):
                    qa_response = qa_engine.analyze_and_answer(
                        query=qa_query,
                        company=selected_company,
                        years=[selected_year] if selected_year != "All" else None,
                        section=selected_sec,
                        top_k=5
                    )

                    # 1. Facts Section Card
                    st.markdown("---")
                    st.markdown("""
                    <div style="background-color:#121318; border-left:6px solid #10B981; border:2px solid #FFFFFF; border-left-width:8px; padding:18px; margin-bottom:18px; box-shadow:4px 4px 0px 0px #10B981;">
                        <h4 style="color:#10B981; margin-top:0; font-family:'Space Grotesk', sans-serif;">EXTRACTED FACTS (DIRECT FILING DISCLOSURES)</h4>
                    """, unsafe_allow_html=True)

                    if qa_response["facts"]:
                        for fact in qa_response["facts"]:
                            st.markdown(f"- **{fact['statement']}** &nbsp; <span style='font-size:0.75rem; background:#38BDF8; color:#000; padding:2px 6px; font-weight:700;'>{fact['citation_tag']}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No direct numerical or policy statements matched.")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # 2. Analytical Interpretations Card
                    st.markdown("""
                    <div style="background-color:#121318; border-left:6px solid #FFDE4D; border:2px solid #FFFFFF; border-left-width:8px; padding:18px; margin-bottom:18px; box-shadow:4px 4px 0px 0px #FFDE4D;">
                        <h4 style="color:#FFDE4D; margin-top:0; font-family:'Space Grotesk', sans-serif;">ANALYTICAL INTERPRETATIONS & TEMPORAL SYNTHESIS</h4>
                    """, unsafe_allow_html=True)

                    for interp in qa_response["interpretations"]:
                        st.markdown(f"• **{interp['observation']}**")
                        st.caption(f"Methodological Basis: {interp['basis']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # 3. Citation Footnotes
                    st.subheader("Verifiable Source Citations")
                    for cit in qa_response["citations"]:
                        st.markdown(f"- **{cit['label']}** — Filing: `{cit['filename']}` | Chunk ID: `{cit['chunk_id']}` | Cross-Encoder Score: `{cit['rerank_score']}`")

                    # Export Dossier
                    st.markdown("---")
                    dossier_json = json.dumps(qa_response, indent=2, default=str)
                    st.download_button(
                        label="Download Research Dossier (JSON)",
                        data=dossier_json,
                        file_name="TEEP2026_Research_Dossier.json",
                        mime="application/json"
                    )

    # =========================================================================
    # TAB 8: RESEARCH EVALUATION & ABLATIONS
    # =========================================================================
    with tab_eval:
        st.markdown("""
        <div style="background-color: #121318; border: 3px solid #FFFFFF; padding: 20px; margin-bottom: 24px; box-shadow: 5px 5px 0px 0px #2563EB;">
            <h3 style="color: #FFDE4D; margin-top: 0; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">RESEARCH EVALUATION & ABLATION SUITE (TEEP 2026)</h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: #FFFFFF; opacity: 0.9; margin-bottom: 0;">
                Benchmarking retrieval performance with Mean Reciprocal Rank (MRR), NDCG@k, Precision@k,
                and multi-stage ablation experiments. Prepares domain-specific fine-tuning datasets and configurations for LoRA / Unsloth.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not indexes_exist or not retriever:
            st.warning("Please index documents to run evaluation benchmarks.")
        else:
            evaluator = ResearchEvaluator(retriever)

            # Benchmark Queries
            benchmark_queries = [
                {"query": "Apple operating income and revenue for fiscal year 2024", "relevant_keywords": ["391,035", "123,216", "operating income", "net sales"]},
                {"query": "Artificial intelligence risk factors and regulatory scrutiny", "relevant_keywords": ["artificial intelligence", "DMA", "generative models", "risk"]},
                {"query": "Microsoft Cloud revenue and capital expenditures datacenter", "relevant_keywords": ["44,477", "137 billion", "Intelligent Cloud", "capital expenditures"]},
                {"query": "Free cash flow and operating activities cash flow", "relevant_keywords": ["operating activities", "free cash flow", "property, plant and equipment"]}
            ]

            col_e1, col_e2 = st.columns([1, 1])
            with col_e1:
                st.markdown("#### Retrieval Performance Evaluation")
                if st.button("Run IR Benchmark Suite", use_container_width=True):
                    with st.spinner("Evaluating retrieval metrics across benchmark queries..."):
                        eval_results = evaluator.evaluate_retrieval_benchmark(benchmark_queries)
                        summary = eval_results["summary"]
                        
                        st.markdown(f"""
                        <div style="background:#121318; border:2px solid #10B981; padding:16px; margin:12px 0;">
                            <div style="font-weight:700; color:#10B981; margin-bottom:8px;">BENCHMARK SUMMARY ({summary['num_test_queries']} Test Queries)</div>
                            <div>• <strong>Mean Reciprocal Rank (MRR)</strong>: <code>{summary['mean_mrr']}</code></div>
                            <div>• <strong>Mean NDCG@3</strong>: <code>{summary['mean_ndcg@3']}</code> | <strong>NDCG@5</strong>: <code>{summary['mean_ndcg@5']}</code></div>
                            <div>• <strong>Mean Precision@5</strong>: <code>{summary['mean_precision@5']}</code> | <strong>Recall@5</strong>: <code>{summary['mean_recall@5']}</code></div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_e2:
                st.markdown("#### Multi-Stage Ablation Studies")
                if st.button("Run Multi-Stage Ablation Experiments", use_container_width=True):
                    with st.spinner("Conducting comparative ablations (Sparse vs Dense vs Graph vs RRF vs Cross-Encoder)..."):
                        sample_qs = [b["query"] for b in benchmark_queries]
                        ablations = evaluator.run_ablation_study(sample_qs, top_k=5)
                        
                        st.table(ablations)
                        st.caption("Relative gain is measured relative to the BM25 Sparse baseline.")

            # Fine-tuning preparation box
            st.markdown("---")
            st.subheader("Domain-Specific Model Fine-Tuning Readiness (LoRA / Unsloth)")
            st.markdown("""
            Generate instruction-tuning datasets in Alpaca or ChatML format from your indexed annual reports
            and generate ready-to-run parameter-efficient fine-tuning (QLoRA) training scripts.
            """)

            col_ft1, col_ft2 = st.columns(2)
            with col_ft1:
                if st.button("Export RAG Chunks to Alpaca Dataset", use_container_width=True):
                    preparer = FinancialFineTuningPreparer()
                    out_path = preparer.generate_instruction_dataset(list(retriever.chunks_lookup.values()), format_type="alpaca")
                    st.success(f"Generated Alpaca dataset: {out_path}")
            
            with col_ft2:
                if st.button("Generate QLoRA / Unsloth Training Script", use_container_width=True):
                    preparer = FinancialFineTuningPreparer()
                    script_path = preparer.generate_unsloth_training_script()
                    st.success(f"Generated training script: {script_path}")

