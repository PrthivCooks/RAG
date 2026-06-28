import re
import pickle
import spacy
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

class GraphBuilder:
    """
    Extracts Named Entities (NER) and syntactic relations (Subject-Verb-Object)
    using spaCy, constructs a NetworkX Knowledge Graph, and supports graph-based search.
    """
    def __init__(self, spacy_nlp):
        self.nlp = spacy_nlp
        self.graph = nx.DiGraph()

    def clean_entity_name(self, name: str) -> str:
        """Clean and normalize entity text."""
        name = re.sub(r'\s+', ' ', name).strip()
        # Remove leading/trailing punctuation but keep internal
        name = name.strip('.,;:"\'()[]{}')
        return name

    def extract_entities_and_relations(self, text: str) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, str]]]:
        """
        Use spaCy to perform:
        1. Named Entity Recognition (NER)
        2. Relationship extraction based on Subject-Verb-Object (SVO) structures
        """
        import re
        doc = self.nlp(text)
        entities = []
        seen_entities = set()

        # Define noise words to filter out common fragments and symbols
        noise_set = {
            "al", "et", "etc", "fig", "table", "figure", "fig.", "figure.", "tab", "tab.",
            "page", "section", "equation", "eq", "eq.", "i.e.", "e.g.", "al.", "et.", "author", "authors"
        }

        # Extract NER entities
        allowed_types = {"ORG", "PERSON", "GPE", "LOC", "PRODUCT", "DATE", "LAW"}
        for ent in doc.ents:
            cleaned_name = self.clean_entity_name(ent.text)
            if not cleaned_name:
                continue
            cleaned_lower = cleaned_name.lower()
            # Filter: must be > 1 char, not in noise set, not a number, and not purely symbols/punctuation
            if (len(cleaned_name) > 1 and 
                cleaned_lower not in noise_set and 
                not cleaned_name.isdigit() and
                not re.match(r'^[\d\W]+$', cleaned_name) and
                ent.label_ in allowed_types):
                
                ent_key = (cleaned_name.lower(), ent.label_)
                if ent_key not in seen_entities:
                    seen_entities.add(ent_key)
                    entities.append({
                        "name": cleaned_name,
                        "type": ent.label_
                    })

        relations = []
        # SVO extraction logic
        for sent in doc.sents:
            # Find root verbs
            verbs = [token for token in sent if token.pos_ == "VERB" and token.dep_ == "ROOT"]
            if not verbs:
                # Fallback to any verb if no root verb is identified
                verbs = [token for token in sent if token.pos_ == "VERB"]
                
            for verb in verbs:
                subj = None
                obj = None
                
                # Look for subject and object in children of the verb
                for child in verb.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = child
                    elif child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                        obj = child
                
                if subj and obj:
                    # Resolve compound nouns (e.g. "Vendor" + "XYZ" -> "Vendor XYZ")
                    def get_compound(token):
                        parts = []
                        # Look at left and right children for compounds
                        for left in token.lefts:
                            if left.dep_ == "compound":
                                parts.append(left.text)
                        parts.append(token.text)
                        for right in token.rights:
                            if right.dep_ == "compound":
                                parts.append(right.text)
                        return " ".join(parts)
                    
                    subj_text = self.clean_entity_name(get_compound(subj))
                    obj_text = self.clean_entity_name(get_compound(obj))
                    relation_text = verb.lemma_.lower()
                    
                    if (len(subj_text) > 1 and len(obj_text) > 1 and relation_text and
                        subj_text.lower() not in noise_set and obj_text.lower() not in noise_set and
                        not subj_text.isdigit() and not obj_text.isdigit() and
                        not re.match(r'^[\d\W]+$', subj_text) and not re.match(r'^[\d\W]+$', obj_text)):
                        relations.append((subj_text, relation_text, obj_text))

        # Add co-occurrence relations for entities in the same sentence if no SVO relations found
        for sent in doc.sents:
            s_ents = [self.clean_entity_name(e.text) for e in sent.ents if e.label_ in allowed_types]
            s_ents = list(set([
                e for e in s_ents 
                if len(e) > 1 and e.lower() not in noise_set and not e.isdigit() and not re.match(r'^[\d\W]+$', e)
            ]))
            if len(s_ents) >= 2:
                for i in range(len(s_ents)):
                    for j in range(i + 1, len(s_ents)):
                        relations.append((s_ents[i], "associated_with", s_ents[j]))

        return entities, relations

    def add_chunk_to_graph(self, chunk_id: str, text: str, doc_metadata: dict):
        """
        Extract entities/relations from chunk and construct/update the NetworkX Graph.
        """
        entities, relations = self.extract_entities_and_relations(text)
        
        # Add the chunk node
        self.graph.add_node(chunk_id, type="chunk", doc_id=doc_metadata.get("doc_id", ""), filename=doc_metadata.get("filename", ""))
        
        # Add entities as nodes and link to the chunk
        for ent in entities:
            ent_name = ent["name"]
            ent_type = ent["type"]
            
            # Add entity node if it doesn't exist
            if not self.graph.has_node(ent_name):
                self.graph.add_node(ent_name, type="entity", entity_type=ent_type)
            
            # Draw edge from entity to chunk
            self.graph.add_edge(ent_name, chunk_id, relation="appears_in")
            
        # Add relation edges between entities
        for subj, rel_type, obj in relations:
            # Add nodes if they don't exist
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, type="entity", entity_type="CO_OCCURRENCE")
            if not self.graph.has_node(obj):
                self.graph.add_node(obj, type="entity", entity_type="CO_OCCURRENCE")
                
            self.graph.add_edge(subj, obj, relation=rel_type)

    def build_graph_index(self, chunks: List[Dict[str, Any]]):
        """Build the graph from a collection of processed chunks."""
        self.graph.clear()
        for c in chunks:
            self.add_chunk_to_graph(c["chunk_id"], c["text"], c["metadata"])
            
    def save_index(self, file_path: Path):
        """Serialize the NetworkX graph using pickle."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Knowledge Graph index saved successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def load_index(self, file_path: Path):
        """Deserialize the NetworkX graph using pickle."""
        if not file_path.exists():
            raise FileNotFoundError(f"Graph index file not found at {file_path}")
        with open(file_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Knowledge Graph index loaded successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def retrieve_by_graph(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieves relevant chunk IDs based on query entities and graph connections.
        Steps:
        1. Extract entities from query.
        2. Find entity nodes matching the extracted query entities.
        3. Traverse edges to discover connected chunks.
        4. Score chunks by frequency of connections, edge types, and entity overlaps.
        """
        doc = self.nlp(query)
        query_entities = [self.clean_entity_name(e.text) for e in doc.ents]
        
        # Fallback keyword matching for entities if spaCy NER doesn't trigger on a short query
        if not query_entities:
            # Look at capitalized words or nouns
            nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
            query_entities = [self.clean_entity_name(n) for n in nouns]
            
        query_entities = list(set([e for e in query_entities if len(e) > 1]))
        
        chunk_scores = {}
        matched_nodes = []
        
        # Find matches in graph
        for q_ent in query_entities:
            # Check for exact matches
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get("type") == "entity":
                    # Check substring match or exact match
                    if q_ent.lower() in node.lower() or node.lower() in q_ent.lower():
                        matched_nodes.append(node)
                        
        matched_nodes = list(set(matched_nodes))
        
        # Traverse graph to find linked chunks
        for ent_node in matched_nodes:
            # 1-hop chunks (entities appearing in chunks)
            for neighbor in self.graph.neighbors(ent_node):
                if self.graph.nodes[neighbor].get("type") == "chunk":
                    chunk_scores[neighbor] = chunk_scores.get(neighbor, 0.0) + 1.0
                    
            # 2-hop chunks (related entities that appear in chunks)
            for related_node in self.graph.neighbors(ent_node):
                if self.graph.nodes[related_node].get("type") == "entity":
                    # This is another entity, find chunks it appears in
                    for hop2_neighbor in self.graph.neighbors(related_node):
                        if self.graph.nodes[hop2_neighbor].get("type") == "chunk":
                            # Give lower weight for 2-hop connections
                            chunk_scores[hop2_neighbor] = chunk_scores.get(hop2_neighbor, 0.0) + 0.35

        # Sort and return
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:top_k]

    def get_see_also_suggestions(self, query: str, retrieved_chunk_ids: List[str], max_suggestions: int = 5) -> List[Dict[str, str]]:
        """
        Suggest related entity terms to show in the "See Also" section of the UI.
        Traverses nodes adjacent to the query entities and retrieved chunks.
        """
        doc = self.nlp(query)
        query_entities = [self.clean_entity_name(e.text).lower() for e in doc.ents]
        
        related_entities = {}
        
        # Collect entities present in the retrieved chunks
        for chunk_id in retrieved_chunk_ids:
            if not self.graph.has_node(chunk_id):
                continue
            # Look at incoming edges (entities pointing to this chunk)
            for u, v, attrs in self.graph.in_edges(chunk_id, data=True):
                if self.graph.nodes[u].get("type") == "entity":
                    ent_name = u
                    ent_type = self.graph.nodes[u].get("entity_type", "General")
                    
                    # Ignore if in query
                    if ent_name.lower() in query_entities:
                        continue
                        
                    related_entities[ent_name] = related_entities.get(ent_name, 0) + 1
                    
        # Sort by co-occurrence frequency in retrieved chunks
        sorted_entities = sorted(related_entities.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        for ent_name, count in sorted_entities:
            ent_type = self.graph.nodes[ent_name].get("entity_type", "General")
            if ent_type != "CO_OCCURRENCE" and len(ent_name) > 2:
                suggestions.append({
                    "name": ent_name,
                    "type": ent_type
                })
            if len(suggestions) >= max_suggestions:
                break
                
        return suggestions
