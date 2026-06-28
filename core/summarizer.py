import numpy as np
import networkx as nx
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from config import SUMMARY_SENTENCES_COUNT

class TextRankSummarizer:
    """
    Extractive summarization using TextRank.
    Segments text, vectorizes sentences using TF-IDF, creates a similarity graph,
    and applies PageRank to identify and extract the most informative sentences.
    """
    def __init__(self, spacy_nlp):
        self.nlp = spacy_nlp

    def summarize(self, text: str, num_sentences: int = SUMMARY_SENTENCES_COUNT) -> str:
        """
        Extract the top N sentences from the text using TextRank.
        
        Args:
            text: The combined document text to summarize.
            num_sentences: The target number of sentences in the summary.
            
        Returns:
            A string containing the extracted summary.
        """
        if not text.strip():
            return ""

        # Step 1: Segment sentences using spaCy
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        try:
            # Step 2: Vectorize sentences using TF-IDF
            # Ignore sentences that are too short to avoid noisy PageRank links
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Step 3: Compute sentence-to-sentence cosine similarity matrix
            # Matrix multiplication of TF-IDF yields cosine similarities since vectorizer outputs normalized vectors
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            # Step 4: Build NetworkX similarity graph
            graph = nx.Graph()
            n = len(sentences)
            graph.add_nodes_from(range(n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    weight = similarity_matrix[i][j]
                    # Only add edge if there is some overlap
                    if weight > 0.01:
                        graph.add_edge(i, j, weight=weight)
            
            # Step 5: Compute PageRank scores
            if graph.number_of_edges() > 0:
                scores = nx.pagerank(graph, weight='weight')
            else:
                # If there are no edges, fallback to uniform scores
                scores = {i: 1.0/n for i in range(n)}
                
            # Step 6: Select top sentences
            ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_sentence_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
            
            # Sort top indices chronologically to preserve document narrative flow
            top_sentence_indices.sort()
            
            summary = [sentences[idx] for idx in top_sentence_indices]
            return " ".join(summary)
            
        except Exception as e:
            print(f"Error during TextRank summarization: {e}. Returning first few sentences.")
            return " ".join(sentences[:num_sentences])
