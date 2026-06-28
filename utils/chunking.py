import numpy as np
import spacy
from typing import List, Dict, Any
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SEMANTIC_SPLIT_THRESHOLD

def recursive_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Fallback recursive chunker that splits text by structural separators
    (paragraphs, double newlines, single newlines, sentences, words)
    while maintaining overlap.
    """
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    
    # Simple recursive-like splitting logic
    def split_text(txt: str, size: int, overlap: int) -> List[str]:
        if len(txt) <= size:
            return [txt]
            
        # Find best separator
        separator = ""
        for sep in separators:
            if sep in txt:
                separator = sep
                break
                
        parts = txt.split(separator) if separator else list(txt)
        result = []
        current_chunk = []
        current_len = 0
        
        for part in parts:
            part_len = len(part) + len(separator) if current_chunk else len(part)
            if current_len + part_len > size:
                if current_chunk:
                    chunk_str = separator.join(current_chunk)
                    result.append(chunk_str)
                    
                    # Backtrack for overlap
                    # Find how many parts we need to keep to satisfy overlap
                    overlap_parts = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        p_len = len(p) + len(separator) if overlap_parts else len(p)
                        if overlap_len + p_len <= overlap:
                            overlap_parts.insert(0, p)
                            overlap_len += p_len
                        else:
                            break
                    current_chunk = overlap_parts
                    current_len = overlap_len
                
                # If a single part exceeds chunk size, split it directly
                if part_len > size:
                    # force split
                    sub_parts = [part[i:i+size] for i in range(0, len(part), size - overlap)]
                    result.extend(sub_parts)
                    current_chunk = []
                    current_len = 0
                    continue
            
            current_chunk.append(part)
            current_len += part_len
            
        if current_chunk:
            result.append(separator.join(current_chunk))
        return result

    return split_text(text, chunk_size, chunk_overlap)

def semantic_chunking(
    text: str, 
    embedding_model, 
    spacy_nlp,
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    threshold: float = SEMANTIC_SPLIT_THRESHOLD
) -> List[str]:
    """
    Advanced semantic chunker.
    1. Splits text into sentences using spaCy.
    2. Generates embeddings for each sentence using a local SentenceTransformer.
    3. Computes cosine similarity between consecutive sentence embeddings.
    4. Group sentences together, breaking when similarity falls below a threshold,
       or when total character length exceeds 1.5 * chunk_size.
    5. Merges or splits groups to align with targets.
    """
    if not text.strip():
        return []
        
    if len(text) <= chunk_size:
        return [text]

    # Step 1: Sentence segmentation via spaCy
    doc = spacy_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if len(sentences) < 2:
        return [text]

    # Step 2: Generate sentence embeddings
    # We embed sentences in a single batch for speed
    try:
        embeddings = embedding_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        print(f"Embedding failed in semantic chunker: {e}. Falling back to recursive chunking.")
        return recursive_chunking(text, chunk_size, chunk_overlap)

    # Step 3: Compute adjacent similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i]
        vec2 = embeddings[i + 1]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            sim = 0.0
        similarities.append(sim)

    # Step 4: Group sentences into semantic sections based on similarity threshold
    semantic_groups = []
    current_group = [sentences[0]]
    current_group_len = len(sentences[0])
    
    for i in range(len(similarities)):
        next_sentence = sentences[i + 1]
        sim = similarities[i]
        
        # If similarity is low, or if adding next sentence makes it too large, split
        if sim < threshold or (current_group_len + len(next_sentence) > chunk_size * 1.5):
            semantic_groups.append(" ".join(current_group))
            current_group = [next_sentence]
            current_group_len = len(next_sentence)
        else:
            current_group.append(next_sentence)
            current_group_len += len(next_sentence) + 1 # +1 for space
            
    if current_group:
        semantic_groups.append(" ".join(current_group))

    # Step 5: Merge and align semantic groups to meet size target
    # Some semantic blocks might be tiny, we merge them, respecting overlaps where appropriate
    chunks = []
    current_chunk = []
    current_chunk_len = 0
    
    for group in semantic_groups:
        if not group.strip():
            continue
            
        # If single group is larger than chunk_size * 1.5, we recursively chunk it
        if len(group) > chunk_size * 1.5:
            # First dump current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_len = 0
            
            # Split this large group recursively
            sub_chunks = recursive_chunking(group, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue

        if current_chunk_len + len(group) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # To maintain context overlap, we can carry over the last sentences or blocks
                # We extract the last few sentences of the previous chunk for overlap
                overlap_text = ""
                # Simple character-based overlap carryover
                prev_text = " ".join(current_chunk)
                if len(prev_text) > chunk_overlap:
                    overlap_text = prev_text[-chunk_overlap:]
                    # align to word boundary
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx:].strip()
                
                if overlap_text:
                    current_chunk = [overlap_text, group]
                    current_chunk_len = len(overlap_text) + len(group) + 1
                else:
                    current_chunk = [group]
                    current_chunk_len = len(group)
            else:
                current_chunk = [group]
                current_chunk_len = len(group)
        else:
            current_chunk.append(group)
            current_chunk_len += len(group) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def hierarchical_chunking(
    text: str,
    embedding_model,
    spacy_nlp,
    parent_size: int = 1200,
    parent_overlap: int = 240,
    child_size: int = 300,
    child_overlap: int = 60,
    use_semantic: bool = True
) -> List[Dict[str, Any]]:
    """
    Creates a hierarchical Parent-Child chunking structure.
    1. Splits the document text into larger parent chunks using recursive chunking.
    2. For each parent chunk, splits it into smaller child chunks (using semantic or recursive chunking).
    3. Returns a list of dictionaries mapping child text to parent text.
    """
    from typing import List, Dict, Any
    # Generate parent chunks
    parent_chunks = recursive_chunking(text, chunk_size=parent_size, chunk_overlap=parent_overlap)
    
    hierarchical_chunks = []
    for p_idx, p_text in enumerate(parent_chunks):
        # Generate child chunks inside parent text
        if use_semantic:
            children = semantic_chunking(
                text=p_text,
                embedding_model=embedding_model,
                spacy_nlp=spacy_nlp,
                chunk_size=child_size,
                chunk_overlap=child_overlap
            )
        else:
            children = recursive_chunking(p_text, chunk_size=child_size, chunk_overlap=child_overlap)
            
        for c_text in children:
            if c_text.strip():
                hierarchical_chunks.append({
                    "child_text": c_text,
                    "parent_text": p_text,
                    "parent_index": p_idx
                })
                
    return hierarchical_chunks
