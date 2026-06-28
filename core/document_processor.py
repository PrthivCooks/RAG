import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from utils.helpers import extract_document_text, extract_metadata
from utils.chunking import semantic_chunking, recursive_chunking, hierarchical_chunking

class DocumentProcessor:
    """
    Orchestrates the loading, text extraction, metadata extraction,
    and custom chunking of enterprise documents.
    """
    def __init__(self, embedding_model, spacy_nlp):
        self.embedding_model = embedding_model
        self.spacy_nlp = spacy_nlp

    def generate_doc_id(self, filepath: str) -> str:
        """Generate a deterministic unique ID for a document based on its path."""
        return hashlib.md5(filepath.encode("utf-8")).hexdigest()

    def process_file(
        self, 
        file_path: Path, 
        chunk_size: int, 
        chunk_overlap: int, 
        use_semantic: bool = True,
        use_hierarchical: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a single file: extract text, extract metadata, chunk it,
        and build structured chunk objects.
        """
        text = extract_document_text(file_path)
        if not text.strip():
            print(f"Warning: Extracted text is empty for {file_path}")
            return []

        # Extract metadata
        meta = extract_metadata(file_path, text)
        doc_id = self.generate_doc_id(meta["filepath"])
        
        processed_chunks = []
        
        if use_hierarchical:
            # We construct parent-child relationships. 
            # Parent size is roughly 4x child size for broader context.
            parent_size = chunk_size * 4
            parent_overlap = chunk_overlap * 4
            
            hier_chunks = hierarchical_chunking(
                text=text,
                embedding_model=self.embedding_model,
                spacy_nlp=self.spacy_nlp,
                parent_size=parent_size,
                parent_overlap=parent_overlap,
                child_size=chunk_size,
                child_overlap=chunk_overlap,
                use_semantic=use_semantic
            )
            
            for idx, item in enumerate(hier_chunks):
                chunk_uuid = str(uuid.uuid4())
                chunk_metadata = meta.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["doc_id"] = doc_id
                
                processed_chunks.append({
                    "chunk_id": chunk_uuid,
                    "doc_id": doc_id,
                    "text": item["child_text"],
                    "parent_text": item["parent_text"],
                    "parent_index": item["parent_index"],
                    "metadata": chunk_metadata
                })
        else:
            if use_semantic:
                chunks = semantic_chunking(
                    text=text,
                    embedding_model=self.embedding_model,
                    spacy_nlp=self.spacy_nlp,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                chunks = recursive_chunking(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            for idx, chunk_text in enumerate(chunks):
                chunk_uuid = str(uuid.uuid4())
                chunk_metadata = meta.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["doc_id"] = doc_id
                
                processed_chunks.append({
                    "chunk_id": chunk_uuid,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "parent_text": chunk_text,
                    "parent_index": idx,
                    "metadata": chunk_metadata
                })
                
        return processed_chunks

    def process_directory(
        self, 
        directory_path: Path, 
        chunk_size: int, 
        chunk_overlap: int, 
        use_semantic: bool = True,
        use_hierarchical: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan directory for supported files (PDF, DOCX, TXT) and process them.
        """
        all_chunks = []
        if not directory_path.exists():
            print(f"Directory {directory_path} does not exist. Creating it.")
            directory_path.mkdir(parents=True, exist_ok=True)
            return all_chunks

        supported_extensions = [".pdf", ".docx", ".txt"]
        
        # Walk directory
        for item in directory_path.rglob("*"):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                print(f"Processing document: {item.name}")
                file_chunks = self.process_file(item, chunk_size, chunk_overlap, use_semantic, use_hierarchical)
                all_chunks.extend(file_chunks)
                
        return all_chunks
