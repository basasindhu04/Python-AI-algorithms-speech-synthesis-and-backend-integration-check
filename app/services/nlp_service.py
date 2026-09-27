import time
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, similarities

class NLPService:
    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        """
        Tokenize and normalize text string.
        """
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [token.strip() for token in clean_text.split() if len(token.strip()) > 1]
        return tokens

    @classmethod
    def compute_tfidf_similarity(
        cls,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Gensim-powered TF-IDF vector similarity engine.
        """
        if not documents:
            return []

        doc_texts = [doc["text"] for doc in documents]
        tokenized_docs = [cls._simple_tokenize(text) for text in doc_texts]
        
        # Build Gensim Dictionary and Corpus
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
        
        if len(dictionary) == 0:
            return [
                {
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "similarity_score": 0.0
                }
                for doc in documents[:top_k]
            ]

        # Fit TF-IDF Model
        tfidf_model = models.TfidfModel(corpus)
        index = similarities.MatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))

        # Query vectorization
        query_tokens = cls._simple_tokenize(query)
        query_bow = dictionary.doc2bow(query_tokens)
        query_tfidf = tfidf_model[query_bow]

        # Calculate cosine similarity vector
        sims = index[query_tfidf]

        # Rank documents
        ranked_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "doc_id": documents[idx]["doc_id"],
                "text": documents[idx]["text"],
                "similarity_score": round(float(sims[idx]), 4)
            })

        return results

    @classmethod
    def compute_vector_average_similarity(
        cls,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Vector Average similarity using token hashing / feature vectors & NumPy.
        """
        if not documents:
            return []

        def text_to_vec(tokens: List[str], dim: int = 100) -> np.ndarray:
            vec = np.zeros(dim, dtype=np.float64)
            if not tokens:
                return vec
            for token in tokens:
                # Hash token into deterministic pseudo-random embedding vector
                hash_val = hash(token)
                np.random.seed(abs(hash_val) % (2**32))
                vec += np.random.randn(dim)
            return vec / len(tokens)

        query_tokens = cls._simple_tokenize(query)
        query_vec = text_to_vec(query_tokens)
        q_norm = np.linalg.norm(query_vec)

        scores = []
        for doc in documents:
            doc_tokens = cls._simple_tokenize(doc["text"])
            doc_vec = text_to_vec(doc_tokens)
            d_norm = np.linalg.norm(doc_vec)

            sim = 0.0
            if q_norm > 0 and d_norm > 0:
                sim = float(np.dot(query_vec, doc_vec) / (q_norm * d_norm))
            scores.append((doc["doc_id"], doc["text"], sim))

        scores.sort(key=lambda x: x[2], reverse=True)
        top_scores = scores[:top_k]

        return [
            {
                "doc_id": doc_id,
                "text": text,
                "similarity_score": round(float(sim), 4)
            }
            for doc_id, text, sim in top_scores
        ]

    @classmethod
    def analyze_similarity(
        cls,
        query: str,
        documents: List[Dict[str, str]],
        method: str = "tfidf",
        top_k: int = 5
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if method == "tfidf":
            results = cls.compute_tfidf_similarity(query, documents, top_k)
        elif method == "vector_average":
            results = cls.compute_vector_average_similarity(query, documents, top_k)
        else:
            raise ValueError(f"Unsupported NLP similarity method: {method}")

        end_time = time.perf_counter()
        return {
            "query": query,
            "method_used": method,
            "results": results,
            "execution_time_ms": (end_time - start_time) * 1000
        }
